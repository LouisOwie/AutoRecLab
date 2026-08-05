import json
import os
from pathlib import Path
from typing import Iterable, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.data_variants import RawData, SplitData
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.preprocess.base import Preprocessor
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state

SPLIT_SEEDS = [11, 22, 33, 44, 55]
TRAINING_SEED = 2027
TEST_FRACTION = 0.20
CUTOFF = 10
ALGORITHMS = {"ALS": LensKit.ImplicitMFScorer, "Pop": LensKit.PopScorer}


class TwoWayUserHoldout(Preprocessor[RawData, SplitData]):
    """OmniRec preprocessor for a user-based two-way 80/20 holdout."""

    def __init__(self, test_fraction: float, seed: int) -> None:
        super().__init__()
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must be strictly between 0 and 1")
        self.test_fraction = test_fraction
        self.seed = seed

    def _process(self, dataset: RecSysDataSet[RawData]) -> RecSysDataSet[SplitData]:
        frame = dataset._data.df.copy()
        rng = np.random.default_rng(self.seed)
        train_parts, test_parts = [], []
        for _, user_frame in frame.groupby("user", sort=False):
            if len(user_frame) < 2:
                raise ValueError("Every user needs at least two interactions")
            n_test = max(
                1,
                min(int(round(len(user_frame) * self.test_fraction)), len(user_frame) - 1),
            )
            test_index = set(
                rng.choice(user_frame.index.to_numpy(), size=n_test, replace=False).tolist()
            )
            test_parts.append(user_frame.loc[sorted(test_index)])
            train_parts.append(user_frame.drop(index=list(test_index)))
        train = pd.concat(train_parts, ignore_index=True)
        test = pd.concat(test_parts, ignore_index=True)
        validation = frame.iloc[0:0].copy()
        result = dataset.replace_data(SplitData(train, validation, test))
        if len(train) + len(test) != len(frame) or len(validation) != 0:
            raise RuntimeError("two-way user holdout invariant failed")
        return result


def extract_metrics(evaluator: Evaluator, seed: int) -> dict[str, dict[str, float]]:
    tables = evaluator.get_results()
    if not tables:
        raise RuntimeError(f"No OmniRec evaluation results for split seed {seed}")
    table = pd.concat(tables.values(), ignore_index=True)
    result = {}
    for label, enum_value in ALGORITHMS.items():
        algorithm_name = str(enum_value)
        rows = table[
            table["algorithm"].astype(str).str.startswith(algorithm_name)
            & table["name"].isin({"NDCG", "Precision"})
            & (table["k"] == CUTOFF)
        ]
        if len(rows) != 2:
            raise RuntimeError(
                f"Expected two metrics for {label}, seed {seed}; found {len(rows)}"
            )
        result[label] = {
            str(row["name"]): float(row["value"])
            for _, row in rows.iterrows()
        }
    return result


def build_split(seed: int):
    set_random_state(seed)
    raw = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    raw_count = raw.num_interactions()
    implicit = MakeImplicit(4).process(raw)
    implicit_count = implicit.num_interactions()
    core = CorePruning(5).process(implicit)
    core_count = core.num_interactions()
    split = TwoWayUserHoldout(TEST_FRACTION, seed).process(core)
    return split, {
        "raw_interactions": int(raw_count),
        "implicit_interactions": int(implicit_count),
        "five_core_interactions": int(core_count),
    }


def paired_analysis(results: pd.DataFrame) -> dict:
    output = {}
    for metric in ["nDCG@10", "Precision@10"]:
        wide = results[results["metric"] == metric].pivot(
            index="split_seed", columns="algorithm", values="value"
        )
        differences = (wide["ALS"] - wide["Pop"]).astype(float)
        sd = differences.std(ddof=1)
        output[metric] = {
            "als_minus_pop_by_seed": {str(k): float(v) for k, v in differences.items()},
            "mean_difference": float(differences.mean()),
            "sample_std_difference": float(sd),
            "minimum_difference": float(differences.min()),
            "maximum_difference": float(differences.max()),
            "paired_t_like_statistic": (
                float(differences.mean() / (sd / np.sqrt(len(differences))))
                if sd > 0
                else None
            ),
        }
    output["interpretation"] = (
        "Descriptive paired comparison across five common random splits; "
        "inference is limited by the small number of paired splits."
    )
    return output


def make_plot(results: pd.DataFrame, metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for algorithm, group in results[results["metric"] == metric].groupby("algorithm"):
        group = group.sort_values("split_seed")
        ax.plot(
            group["split_seed"],
            group["value"],
            marker="o",
            linewidth=2,
            label=algorithm,
        )
    ax.set_xlabel("User-holdout split seed")
    ax.set_ylabel(metric)
    ax.set_title(f"Split-seed sensitivity: {metric}")
    ax.set_xticks(SPLIT_SEEDS)
    ax.grid(alpha=0.3)
    ax.legend(title="Algorithm")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_per_run_table(results: pd.DataFrame) -> pd.DataFrame:
    """Pivot long metric records into one complete row per algorithm and seed."""
    per_run = (
        results.pivot_table(
            index=["algorithm", "split_seed"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    required = ["algorithm", "split_seed", "nDCG@10", "Precision@10"]
    missing = [column for column in required if column not in per_run.columns]
    if missing:
        raise RuntimeError(f"Per-run table is missing required columns: {missing}")
    diagnostic = results[
        [
            "algorithm",
            "split_seed",
            "train_interactions",
            "validation_interactions",
            "test_interactions",
        ]
    ].drop_duplicates(["algorithm", "split_seed"])
    per_run = per_run.merge(diagnostic, on=["algorithm", "split_seed"], how="left")
    return per_run.sort_values(["algorithm", "split_seed"]).reset_index(drop=True)


def main() -> None:
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    working = Path(working_dir)
    original_cwd = Path.cwd()
    if len(set(SPLIT_SEEDS)) != 5:
        raise ValueError("SPLIT_SEEDS must contain exactly five distinct seeds")

    records = []
    preprocessing_counts = None
    for seed in SPLIT_SEEDS:
        print(f"\n=== ALS + Pop: split seed {seed} ===", flush=True)
        dataset, counts = build_split(seed)
        if preprocessing_counts is None:
            preprocessing_counts = counts
        elif counts != preprocessing_counts:
            raise RuntimeError("Preprocessing counts changed across split seeds")
        train = dataset._data.get("train")
        validation = dataset._data.get("val")
        test = dataset._data.get("test")
        if len(validation) != 0:
            raise RuntimeError("Validation must be empty in this experiment")

        seed_dir = working / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        dataset.save(seed_dir / "split_dataset")
        os.chdir(seed_dir)
        try:
            set_random_state(TRAINING_SEED)
            plan = ExperimentPlan(f"full_experiment_seed_{seed}")
            plan.add_algorithm(LensKit.ImplicitMFScorer, {})
            plan.add_algorithm(LensKit.PopScorer, {})
            evaluator = Evaluator(NDCG([CUTOFF]), Precision([CUTOFF]))
            run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
            metric_values = extract_metrics(evaluator, seed)
        finally:
            os.chdir(original_cwd)

        for algorithm, metrics in metric_values.items():
            for metric_name, value in [
                ("nDCG@10", metrics["NDCG"]),
                ("Precision@10", metrics["Precision"]),
            ]:
                records.append(
                    {
                        "algorithm": algorithm,
                        "split_seed": seed,
                        "metric": metric_name,
                        "value": value,
                        "train_interactions": len(train),
                        "validation_interactions": len(validation),
                        "test_interactions": len(test),
                    }
                )
            print(
                f"seed={seed}, {algorithm} nDCG@10={metrics['NDCG']:.8f}, "
                f"Precision@10={metrics['Precision']:.8f}",
                flush=True,
            )

    results = pd.DataFrame(records).sort_values(
        ["metric", "algorithm", "split_seed"]
    )
    per_run_results = make_per_run_table(results)
    results.to_csv(working / "experiment_results_long.csv", index=False)
    per_run_results.to_csv(working / "per_run_results.csv", index=False)

    summary = {}
    grouped_results = cast(
        Iterable[tuple[tuple[str, str], pd.DataFrame]],
        results.groupby(["algorithm", "metric"]),
    )
    for (algorithm, metric), group in grouped_results:
        values = group["value"].astype(float)
        summary[f"{algorithm}__{metric}"] = {
            "algorithm": algorithm,
            "metric": metric,
            "n_seeds": int(len(values)),
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    settings = {
        "stage": "final",
        "dataset": "MovieLens100K",
        "preprocessing": "retain ratings >= 4 (ratings > 3), then iterative 5-core filtering",
        "preprocessing_counts": preprocessing_counts,
        "split": "independent random user-based two-way holdout",
        "train_fraction": 0.80,
        "test_fraction": TEST_FRACTION,
        "validation": "none; empty SplitData validation field",
        "split_seeds": SPLIT_SEEDS,
        "fixed_training_seed": TRAINING_SEED,
        "algorithms": {
            "ALS": {
                "omnirec_identifier": str(LensKit.ImplicitMFScorer),
                "configuration": "standard OmniRec defaults; fixed {}",
            },
            "Pop": {
                "omnirec_identifier": str(LensKit.PopScorer),
                "configuration": "standard OmniRec defaults; fixed {}",
            },
        },
        "metrics": ["nDCG@10", "Precision@10"],
        "evaluation_protocol": "top-10 implicit recommendations exclude split-training interactions and score held-out positive interactions for the same users",
    }
    (working / "experiment_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    (working / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    paired = paired_analysis(results)
    (working / "paired_comparison.json").write_text(
        json.dumps(paired, indent=2), encoding="utf-8"
    )

    print("\n=== Per-run results (one row per algorithm and split seed) ===", flush=True)
    print(
        per_run_results[
            ["algorithm", "split_seed", "nDCG@10", "Precision@10"]
        ].to_string(index=False),
        flush=True,
    )
    print("\n=== Across-seed summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("\n=== Paired ALS-versus-Pop analysis ===", flush=True)
    print(json.dumps(paired, indent=2), flush=True)
    print("\n=== Protocol and preprocessing ===", flush=True)
    print(json.dumps(settings, indent=2), flush=True)
    make_plot(results, "nDCG@10", working / "seed_variability_ndcg10.png")
    make_plot(results, "Precision@10", working / "seed_variability_precision10.png")
    print(f"Saved final experiment artifacts under {working}", flush=True)


if __name__ == "__main__":
    main()
