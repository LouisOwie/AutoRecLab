import os
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.data_variants import SplitData
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


SPLIT_SEEDS = [11, 23, 37, 53, 71]
TRAINING_SEED = 2026
TEST_FRACTION = 0.20
CUTOFF = 10

# Explicitly record the verified standard effective ImplicitMFConfig values.
ALS_CONFIG = {
    "embedding_size": 64,
    "epochs": 10,
    "regularization": 0.1,
    "weight": 40,
    "use_ratings": False,
    "user_embeddings": True,
}
ALGORITHM_PREFIXES = {
    "ALS": "LensKit.ImplicitMFScorer",
    "Pop": "LensKit.PopScorer",
}


class SummaryRecord(TypedDict):
    algorithm: str
    dataset: str
    n_split_seeds: int
    fixed_training_seed: int
    metric: str
    mean: float
    sample_std: float
    min: float
    max: float
    range: float
    coefficient_of_variation: float


def make_implicit_five_core(canonical_path: str | Path) -> pd.DataFrame:
    """Create ratings > 3 implicit interactions and iteratively enforce 5-core."""
    interactions = pd.read_csv(canonical_path)
    interactions = interactions.loc[interactions["rating"] > 3].copy()

    while True:
        user_counts = interactions["user"].value_counts()
        item_counts = interactions["item"].value_counts()
        pruned = interactions.loc[
            interactions["user"].isin(user_counts[user_counts >= 5].index)
            & interactions["item"].isin(item_counts[item_counts >= 5].index)
        ].copy()
        if len(pruned) == len(interactions):
            break
        interactions = pruned

    # OmniRec's LensKit runner identifies implicit feedback by the absence of rating.
    return interactions.drop(columns=["rating"]).reset_index(drop=True)


def user_holdout_80_20(
    interactions: pd.DataFrame, split_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic, disjoint two-way per-user split with nonempty user test sets."""
    rng = np.random.default_rng(split_seed)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, user_rows in interactions.groupby("user", sort=False):
        row_indices = user_rows.index.to_numpy()
        test_count = max(1, int(np.ceil(TEST_FRACTION * len(row_indices))))
        if test_count >= len(row_indices):
            raise ValueError("A user has insufficient interactions for train and test.")

        test_indices = rng.choice(row_indices, size=test_count, replace=False)
        is_test = user_rows.index.isin(test_indices)
        test_parts.append(user_rows.loc[is_test])
        train_parts.append(user_rows.loc[~is_test])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    validation = interactions.iloc[0:0].copy()
    return train, validation, test


def audit_split(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> dict[str, float | int]:
    """Fail fast unless this is an exhaustive two-way per-user partition."""
    if not validation.empty:
        raise RuntimeError("Validation must be empty for a two-way 80/20 holdout.")

    train_keys = set(map(tuple, train[["user", "item"]].to_numpy()))
    test_keys = set(map(tuple, test[["user", "item"]].to_numpy()))
    if train_keys & test_keys:
        raise RuntimeError("Train and test interactions overlap.")

    test_per_user = test.groupby("user").size()
    train_per_user = train.groupby("user").size()
    if test_per_user.empty or (test_per_user < 1).any():
        raise RuntimeError("At least one evaluated user has no test interaction.")
    if set(test_per_user.index) != set(train_per_user.index):
        raise RuntimeError("Every evaluated user must retain both train and test data.")

    total = len(train) + len(test)
    return {
        "train_interactions": len(train),
        "validation_interactions": len(validation),
        "test_interactions": len(test),
        "evaluated_users": int(test_per_user.size),
        "minimum_test_interactions_per_user": int(test_per_user.min()),
        "actual_test_fraction": float(len(test) / total),
    }


def extract_metrics_at_10(evaluator: Evaluator) -> list[dict[str, float | str]]:
    """Extract one NDCG@10 and one Precision@10 value for ALS and Pop."""
    result_frames = list(evaluator.get_results().values())
    if not result_frames:
        raise RuntimeError("OmniRec returned no evaluation results.")

    results = pd.concat(result_frames, ignore_index=True)
    records: list[dict[str, float | str]] = []
    for algorithm, prefix in ALGORITHM_PREFIXES.items():
        algorithm_results = results.loc[results["algorithm"].astype(str).str.startswith(prefix)]
        record: dict[str, float | str] = {"algorithm": algorithm}
        for metric_name, output_name in (("NDCG", "ndcg_at_10"), ("Precision", "precision_at_10")):
            selected = algorithm_results.loc[
                (algorithm_results["name"] == metric_name)
                & (algorithm_results["k"] == CUTOFF),
                "value",
            ]
            if len(selected) != 1:
                raise RuntimeError(
                    f"Expected exactly one {metric_name}@{CUTOFF} value for {algorithm}; "
                    f"found {len(selected)}. Results:\n{results}"
                )
            record[output_name] = float(selected.iloc[0])
        records.append(record)
    return records


def make_comparison_plot(per_seed: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path) -> None:
    """Plot both algorithms and label every observed split-seed value."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for algorithm, group in per_seed.groupby("algorithm", sort=False):
        group = group.sort_values("split_seed")
        ax.plot(group["split_seed"], group[metric], marker="o", linewidth=1.8, label=algorithm)
        for seed, value in zip(group["split_seed"], group[metric]):
            ax.annotate(f"{value:.4f}", (seed, value), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("data-split seed")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(SPLIT_SEEDS)
    ax.grid(alpha=0.3)
    ax.legend(title="algorithm")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    working_path = Path(working_dir)

    raw_dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    canonical_path = raw_dataset.meta.canon_pth
    if canonical_path is None:
        raise RuntimeError("MovieLens100K did not provide a canonical dataset path.")
    implicit_interactions = make_implicit_five_core(canonical_path)
    core_users = int(implicit_interactions["user"].nunique())
    core_items = int(implicit_interactions["item"].nunique())
    print(
        "Preprocessing complete: ratings > 3 converted to implicit and iterative 5-core "
        f"filtering retained {core_users} users, {core_items} items, and "
        f"{len(implicit_interactions)} interactions"
    )
    print(f"Effective ALS configuration: {ALS_CONFIG}; fixed training seed={TRAINING_SEED}")

    seed_records: list[dict[str, float | int | str]] = []
    audit_records: list[dict[str, float | int]] = []
    original_cwd = os.getcwd()
    try:
        for split_seed in SPLIT_SEEDS:
            print(f"\n===== Split seed {split_seed}: per-user 80/20 holdout =====")
            set_random_state(split_seed)
            train, validation, test = user_holdout_80_20(implicit_interactions, split_seed)
            audit = audit_split(train, validation, test)
            audit["split_seed"] = split_seed
            audit_records.append(audit)
            print(
                f"split_seed={split_seed} train={audit['train_interactions']} "
                f"test={audit['test_interactions']} test_fraction={audit['actual_test_fraction']:.4f} "
                f"users={audit['evaluated_users']}"
            )

            # Both algorithms receive this identical SplitData; its train frame is the
            # sole training input passed by OmniRec to the runner.
            split_dataset = RecSysDataSet(SplitData(train, validation, test), raw_dataset.meta)
            seed_dir = working_path / f"split_seed_{split_seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(seed_dir)
            try:
                set_random_state(TRAINING_SEED)
                plan = ExperimentPlan(plan_name=f"ml100k_implicit_als_pop_split_{split_seed}")
                plan.add_algorithm(LensKit.ImplicitMFScorer, ALS_CONFIG)
                plan.add_algorithm(LensKit.PopScorer)
                evaluator = Evaluator(NDCG(CUTOFF), Precision(CUTOFF))
                run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
                run_records = extract_metrics_at_10(evaluator)
            finally:
                os.chdir(original_cwd)

            for record in run_records:
                record["split_seed"] = split_seed
                record["training_seed"] = TRAINING_SEED
                seed_records.append(record)
                print(
                    f"algorithm={record['algorithm']} split_seed={split_seed} "
                    f"NDCG@{CUTOFF}={record['ndcg_at_10']:.6f} "
                    f"Precision@{CUTOFF}={record['precision_at_10']:.6f}"
                )
    finally:
        os.chdir(original_cwd)

    per_seed = pd.DataFrame(seed_records).sort_values(["algorithm", "split_seed"]).reset_index(drop=True)
    split_audit = pd.DataFrame(audit_records).sort_values("split_seed").reset_index(drop=True)
    per_seed.to_csv(working_path / "algorithm_split_seed_results.csv", index=False)
    # Retain the original output name while making it a complete long-form table.
    per_seed.to_csv(working_path / "als_split_seed_results.csv", index=False)
    split_audit.to_csv(working_path / "split_audit.csv", index=False)

    summary_records: list[SummaryRecord] = []
    for algorithm, group in per_seed.groupby("algorithm", sort=False):
        algorithm_name = str(algorithm)
        for metric, metric_label in (("ndcg_at_10", f"NDCG@{CUTOFF}"), ("precision_at_10", f"Precision@{CUTOFF}")):
            values = group[metric]
            mean_value = float(values.mean())
            std_value = float(values.std(ddof=1))
            summary_records.append({
                "algorithm": algorithm_name,
                "dataset": "MovieLens100K",
                "n_split_seeds": len(group),
                "fixed_training_seed": TRAINING_SEED,
                "metric": metric_label,
                "mean": mean_value,
                "sample_std": std_value,
                "min": float(values.min()),
                "max": float(values.max()),
                "range": float(values.max() - values.min()),
                "coefficient_of_variation": float(std_value / mean_value) if mean_value else float("nan"),
            })
    summary = pd.DataFrame(summary_records)
    summary.to_csv(working_path / "algorithm_split_seed_summary.csv", index=False)
    summary.to_csv(working_path / "als_split_seed_summary.csv", index=False)

    print("\nSplit audit:")
    print(split_audit.to_string(index=False))
    print("\nComplete per-seed results:")
    print(per_seed.to_string(index=False))
    print("\nAcross-split-seed descriptive statistics:")
    print(summary.to_string(index=False))

    # Preserve the prototype's ALS-only NDCG output.
    als_results = per_seed.loc[per_seed["algorithm"] == "ALS"].sort_values("split_seed")
    als_mean = float(als_results["ndcg_at_10"].mean())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(als_results["split_seed"], als_results["ndcg_at_10"], marker="o", linewidth=1.8, label="ALS NDCG@10")
    ax.axhline(als_mean, color="tab:red", linestyle="--", label=f"mean = {als_mean:.4f}")
    ax.set_xlabel("data-split seed")
    ax.set_ylabel(f"NDCG@{CUTOFF}")
    ax.set_title("MovieLens100K implicit ALS: split-seed sensitivity")
    ax.set_xticks(SPLIT_SEEDS)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    prototype_plot_path = working_path / "als_ndcg_at_10_by_split_seed.png"
    fig.savefig(prototype_plot_path, dpi=150)
    plt.close(fig)

    ndcg_plot_path = working_path / "ndcg_at_10_by_algorithm_and_split_seed.png"
    precision_plot_path = working_path / "precision_at_10_by_algorithm_and_split_seed.png"
    make_comparison_plot(per_seed, "ndcg_at_10", f"NDCG@{CUTOFF}",
                         "MovieLens100K implicit: NDCG@10 split-seed sensitivity", ndcg_plot_path)
    make_comparison_plot(per_seed, "precision_at_10", f"Precision@{CUTOFF}",
                         "MovieLens100K implicit: Precision@10 split-seed sensitivity", precision_plot_path)

    report_lines = [
        "MovieLens100K implicit split-seed sensitivity experiment",
        "=" * 58,
        f"Preprocessing: ratings > 3 were converted to implicit interactions, then iterative 5-core filtering retained {core_users} users, {core_items} items, and {len(implicit_interactions)} interactions.",
        f"Splits: user-based disjoint 80/20 holdouts with seeds {SPLIT_SEEDS}; each user retains at least one train and one test interaction. The same SplitData object was evaluated by ALS and Pop for each seed.",
        f"Algorithms: OmniRec LensKit.ImplicitMFScorer with fixed configuration {ALS_CONFIG}; OmniRec LensKit.PopScorer with its default configuration. OmniRec random state was reset to training seed {TRAINING_SEED} before every two-algorithm run.",
        f"Evaluation: OmniRec implicit top-N recommendation/evaluation path with NDCG@{CUTOFF} and Precision@{CUTOFF}; training uses only each split's train frame.",
        "",
        "Descriptive seed-sensitivity summary (five splits; no statistical-significance inference):",
    ]
    for row in summary_records:
        report_lines.append(
            f"- {row['algorithm']} {row['metric']}: mean={row['mean']:.6f}, sample SD={row['sample_std']:.6f}, "
            f"range={row['range']:.6f} ({row['min']:.6f} to {row['max']:.6f}), CV={row['coefficient_of_variation']:.2%}."
        )
    report_lines.extend([
        "",
        "Outputs:",
        "- algorithm_split_seed_results.csv: complete algorithm-by-seed metric table.",
        "- algorithm_split_seed_summary.csv: algorithm/metric descriptive statistics.",
        "- split_audit.csv: split integrity and size audit.",
        "- als_ndcg_at_10_by_split_seed.png: retained prototype ALS NDCG plot.",
        "- ndcg_at_10_by_algorithm_and_split_seed.png and precision_at_10_by_algorithm_and_split_seed.png: labeled ALS-vs-Pop comparison plots.",
    ])
    report_path = working_path / "experiment_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nSaved complete per-seed metrics to: {working_path / 'algorithm_split_seed_results.csv'}")
    print(f"Saved split audit to: {working_path / 'split_audit.csv'}")
    print(f"Saved descriptive statistics to: {working_path / 'algorithm_split_seed_summary.csv'}")
    print(f"Saved retained prototype plot to: {prototype_plot_path}")
    print(f"Saved NDCG comparison plot to: {ndcg_plot_path}")
    print(f"Saved Precision comparison plot to: {precision_plot_path}")
    print(f"Saved reproducibility report to: {report_path}")


if __name__ == "__main__":
    main()
