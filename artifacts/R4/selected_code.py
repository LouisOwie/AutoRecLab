import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.data_variants import RawData, SplitData
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.preprocess.base import Preprocessor
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.recsys_data_set import RecSysDataSet as OmniRecDataSet
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


class TwoWayUserHoldout(Preprocessor[RawData, SplitData]):
    """Exact OmniRec SplitData train/test-only user holdout."""

    def __init__(self, test_size: float, seed: int) -> None:
        super().__init__()
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be strictly between 0 and 1")
        self.test_size = test_size
        self.seed = seed

    def _process(self, dataset: OmniRecDataSet[RawData]) -> OmniRecDataSet[SplitData]:
        df = dataset._data.df.copy()
        train_parts, test_parts = [], []
        for _, user_df in df.groupby("user", sort=False):
            train_df, test_df = train_test_split(
                user_df, test_size=self.test_size, random_state=self.seed, shuffle=True
            )
            train_parts.append(train_df)
            test_parts.append(test_df)
        train = pd.concat(train_parts, axis=0).sort_index()
        test = pd.concat(test_parts, axis=0).sort_index()
        validation = df.iloc[0:0].copy()
        if len(train) + len(test) != len(df) or set(train.index) & set(test.index):
            raise RuntimeError("Holdout is not exhaustive and disjoint")
        return dataset.replace_data(SplitData(train, validation, test))


def pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def extract_metrics(evaluator: Evaluator, seed: int) -> dict[str, float]:
    result_sets = evaluator.get_results()
    if not result_sets:
        raise RuntimeError(f"No OmniRec results for seed {seed}")
    frame = pd.concat(list(result_sets.values()), ignore_index=True)
    out = {}
    for algorithm, label in (("ImplicitMFScorer", "ALS"), ("PopScorer", "Pop")):
        subset = frame[frame["algorithm"].astype(str).str.contains(algorithm)]
        for metric in ("NDCG", "Precision"):
            hit = subset[(subset["name"].astype(str).str.lower() == metric.lower()) & (subset["k"] == 10)]
            if hit.empty:
                raise RuntimeError(f"Missing {label} {metric}@10 for seed {seed}:\n{frame}")
            out[f"{label} {metric}@10"] = float(hit.iloc[-1]["value"])
    return out


def ci95(values: pd.Series) -> tuple[float, float]:
    mean = float(values.mean())
    half = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return mean - half, mean + half


def main() -> None:
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    working_path = Path(working_dir)
    seeds = [11, 22, 33, 44, 55]
    fixed_configs = {"ALS": {}, "Pop": {}}

    print("Dataset: MovieLens100K")
    print("Preprocessing: ratings >= 4 -> implicit; iterative 5-core filtering")
    print("Split: seeded user-based 80/20 train/test holdout; validation is empty")
    print(f"Algorithms: {list(fixed_configs)}; metrics: nDCG@10, Precision@10")
    print(f"Seeds: {seeds}")

    raw = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    raw_df = raw._data.df.copy()
    before = {"interactions": len(raw_df), "users": raw_df["user"].nunique(), "items": raw_df["item"].nunique()}
    preprocessed = Pipe(MakeImplicit(4), CorePruning(5)).process(raw)
    pre_df = preprocessed._data.df.copy()
    user_min = int(pre_df.groupby("user").size().min())
    item_min = int(pre_df.groupby("item").size().min())
    if user_min < 5 or item_min < 5:
        raise RuntimeError(f"5-core verification failed: user_min={user_min}; item_min={item_min}")
    after = {"interactions": len(pre_df), "users": pre_df["user"].nunique(), "items": pre_df["item"].nunique()}
    print("\n=== Dataset counts and 5-core verification ===")
    print(f"Before: {before}; after: {after}; user_min={user_min}; item_min={item_min}")

    meta = raw.meta
    provenance = {
        "dataset": "MovieLens100K",
        "source": str(getattr(meta, "canon_pth", "unknown")),
        "raw_dir": str(getattr(meta, "raw_dir", "unknown")),
        "metadata_name": str(getattr(meta, "name", "MovieLens100K")),
        "omnirec_version": pkg_version("omnirec"),
        "omnirec_runner_version": pkg_version("omnirec-runner"),
        "python": sys.version,
        "platform": platform.platform(),
        "before_counts": before,
        "after_counts": after,
        "core": 5,
        "implicit_threshold": 4,
        "split": "per-user shuffled 80/20, empty validation",
        "evaluation": "OmniRec user-averaged nDCG@10 and Precision@10 over held-out positives; runner top-10 candidate lists exclude training positives",
        "candidate_convention": "all catalog items, with each user's training positives excluded by OmniRec recommendation workflow",
        "model_configs": fixed_configs,
        "training_random_state": "set_random_state(seed) before split and run",
        "seeds": seeds,
        "rerun_tolerance": 1e-10,
    }
    (working_path / "experiment_metadata.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")

    rows = []
    for seed in seeds:
        print(f"\n=== Seed {seed} ===", flush=True)
        set_random_state(seed)
        split_dataset = TwoWayUserHoldout(0.20, seed).process(preprocessed)
        train = split_dataset._data.get("train")
        val = split_dataset._data.get("val")
        test = split_dataset._data.get("test")
        train_users, test_users = set(train["user"]), set(test["user"])
        if train_users != test_users or set(train.index) & set(test.index):
            raise RuntimeError(f"Invalid evaluated-user overlap/disjointness for seed {seed}")
        split_counts = {"train": len(train), "val": len(val), "test": len(test)}
        print(f"Split counts: {split_counts}; evaluated users: {len(test_users)}", flush=True)

        plan = ExperimentPlan(plan_name=f"MovieLens100K_Implicit_ALS_Pop_seed_{seed}")
        plan.add_algorithm(LensKit.ImplicitMFScorer, fixed_configs["ALS"])
        plan.add_algorithm(LensKit.PopScorer, fixed_configs["Pop"])
        evaluator = Evaluator(NDCG([10]), Precision([10]))
        run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
        metrics = extract_metrics(evaluator, seed)
        for algorithm in ("ALS", "Pop"):
            rows.append({
                "dataset": "MovieLens100K", "algorithm": algorithm, "seed": seed,
                **split_counts, "train_users": len(train_users), "test_users": len(test_users),
                "configuration": json.dumps(fixed_configs[algorithm], sort_keys=True),
                "training_random_state": seed,
                "NDCG@10": metrics[f"{algorithm} NDCG@10"],
                "Precision@10": metrics[f"{algorithm} Precision@10"],
            })
            print(f"{algorithm}: nDCG@10={metrics[f'{algorithm} NDCG@10']:.6f}, Precision@10={metrics[f'{algorithm} Precision@10']:.6f}", flush=True)

    results = pd.DataFrame(rows).sort_values(["algorithm", "seed"])
    raw_path = working_path / "movielens100k_implicit_seed_raw_results.csv"
    results.to_csv(raw_path, index=False)

    summary_rows = []
    melted = results.melt(
        id_vars=["algorithm"],
        value_vars=["NDCG@10", "Precision@10"],
        var_name="metric",
        value_name="value",
    )
    for _, group in melted.groupby(["algorithm", "metric"]):
        algorithm = str(group["algorithm"].iloc[0])
        metric = str(group["metric"].iloc[0])
        values = group["value"]
        lo, hi = ci95(values)
        summary_rows.append({"algorithm": algorithm, "metric": metric, "mean": values.mean(), "std": values.std(ddof=1), "median": values.median(), "min": values.min(), "max": values.max(), "ci95_low": lo, "ci95_high": hi})
    summary = pd.DataFrame(summary_rows)
    summary_path = working_path / "movielens100k_implicit_seed_summary.csv"
    summary.to_csv(summary_path, index=False)

    wide = results.pivot(index="seed", columns="algorithm", values=["NDCG@10", "Precision@10"])
    paired_rows = []
    for metric in ["NDCG@10", "Precision@10"]:
        diff = wide[(metric, "ALS")] - wide[(metric, "Pop")]
        lo, hi = ci95(diff)
        paired_rows.append({"metric": metric, "comparison": "ALS-minus-Pop", "mean_difference": diff.mean(), "std": diff.std(ddof=1), "median": diff.median(), "min": diff.min(), "max": diff.max(), "ci95_low": lo, "ci95_high": hi, "interpretation": "Exploratory paired five-seed comparison; no significance claim."})
    paired = pd.DataFrame(paired_rows)
    paired_path = working_path / "movielens100k_implicit_paired_differences.csv"
    paired.to_csv(paired_path, index=False)
    print("\n=== Five-seed summaries ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n=== Exploratory paired ALS-minus-Pop differences ===")
    print(paired.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    for metric, filename, title in [("NDCG@10", "movielens100k_ndcg10_als_vs_pop.png", "MovieLens100K implicit nDCG@10: ALS vs Pop"), ("Precision@10", "movielens100k_precision10_als_vs_pop.png", "MovieLens100K implicit Precision@10: ALS vs Pop")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for algorithm, marker in [("ALS", "o"), ("Pop", "s")]:
            group = results[results["algorithm"] == algorithm].sort_values("seed")
            s = summary[(summary["algorithm"] == algorithm) & (summary["metric"] == metric)].iloc[0]
            ax.plot(group["seed"], group[metric], marker=marker, label=algorithm)
            ax.errorbar([group["seed"].mean()], [s["mean"]], yerr=[[s["mean"] - s["ci95_low"]], [s["ci95_high"] - s["mean"]]], fmt="none", capsize=5)
        ax.set_title(title); ax.set_xlabel("split random seed"); ax.set_ylabel(metric); ax.grid(True, alpha=0.3); ax.legend(title="Algorithm")
        fig.tight_layout(); fig.savefig(working_path / filename, dpi=180); plt.close(fig)

    print(f"\nSaved raw results to {raw_path}")
    print(f"Saved summaries to {summary_path} and paired differences to {paired_path}")


if __name__ == "__main__":
    main()
