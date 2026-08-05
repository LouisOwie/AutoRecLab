import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.data_variants import SplitData
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.filter import RatingFilter
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


SEEDS = [11, 23, 37, 41, 59]
ALGORITHMS = {
    "ALS": (LensKit.ImplicitMFScorer, {
        "embedding_size": 64,
        "epochs": 10,
        "regularization": 0.1,
        "user_embeddings": True,
        "use_ratings": False,
        "weight": 1.0,
    }),
    "Pop": (LensKit.PopScorer, {}),
}


def extract_metrics(evaluator):
    tables = evaluator.get_results()
    if not tables:
        raise RuntimeError("OmniRec returned no evaluation result tables")
    frames = [table for table in tables.values() if table is not None and not table.empty]
    if not frames:
        raise RuntimeError("OmniRec returned only empty evaluation result tables")
    results = pd.concat(frames, ignore_index=True)
    selected = results[(results["k"] == 10) & results["name"].isin(["NDCG", "Precision"])]
    if set(selected["name"]) != {"NDCG", "Precision"}:
        raise RuntimeError(f"Missing NDCG@10 or Precision@10 in OmniRec output:\n{results}")
    return {
        "NDCG@10": float(selected.loc[selected["name"] == "NDCG", "value"].iloc[-1]),
        "Precision@10": float(selected.loc[selected["name"] == "Precision", "value"].iloc[-1]),
    }


def run_one_seed(base_dataset, seed, algorithm_name, algorithm, config):
    set_random_state(seed)
    initial_split = UserHoldout(
        validation_size=0.001,
        test_size=0.20,
    ).process(base_dataset)

    # UserHoldout's documented API requires validation_size.  Merge that tiny
    # partition into train so the actual experiment has train/test only.
    train = initial_split._data.get("train")
    validation = initial_split._data.get("val")
    test = initial_split._data.get("test")
    split_dataset = initial_split.replace_data(
        SplitData(
            pd.concat([train, validation], ignore_index=True),
            train.iloc[0:0].copy(),
            test,
        )
    )

    plan = ExperimentPlan(plan_name=f"MovieLens100K-{algorithm_name}-seed-{seed}")
    if config:
        plan.add_algorithm(algorithm, dict(config))
    else:
        plan.add_algorithm(algorithm)
    evaluator = Evaluator(NDCG([10]), Precision([10]))

    print(f"Running seed={seed}: OmniRec {algorithm_name}")
    run_omnirec(split_dataset, plan, evaluator)
    metrics = extract_metrics(evaluator)
    row = {"seed": seed, "algorithm": algorithm_name, **metrics}
    print(
        f"seed={seed}, algorithm={algorithm_name}, "
        f"NDCG@10={metrics['NDCG@10']:.6f}, "
        f"Precision@10={metrics['Precision@10']:.6f}"
    )
    return row


def summarize(results):
    rows = []
    for (algorithm, metric), group in results.melt(
        id_vars=["seed", "algorithm"],
        value_vars=["NDCG@10", "Precision@10"],
        var_name="metric",
        value_name="value",
    ).groupby(["algorithm", "metric"], sort=True):
        values = group["value"].to_numpy(dtype=float)
        best = group.loc[group["value"].idxmax(), "seed"]
        worst = group.loc[group["value"].idxmin(), "seed"]
        rows.append({
            "algorithm": algorithm,
            "metric": metric,
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
            "best_seed": int(best),
            "worst_seed": int(worst),
            "range": float(np.ptp(values)),
        })
    return pd.DataFrame(rows)


def print_interpretations(stats):
    print("\nSeed-sensitivity interpretation")
    for _, row in stats.iterrows():
        print(
            f"{row['algorithm']} {row['metric']}: mean={row['mean']:.6f}, "
            f"SD={row['std']:.6f}, range={row['range']:.6f}; "
            f"best seed={int(row['best_seed'])}, worst seed={int(row['worst_seed'])}."
        )
    print("These five split realizations describe seed sensitivity; they do not establish statistically reliable differences between algorithms.")


def main():
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)

    raw_dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    base_dataset = Pipe(
        RatingFilter(lower=4),
        MakeImplicit(4),
        CorePruning(5),
    ).process(raw_dataset)

    filtered_df = base_dataset._data.df
    dataset_counts = {
        "users": int(filtered_df["user"].nunique()),
        "items": int(filtered_df["item"].nunique()),
        "interactions": int(base_dataset.num_interactions()),
    }
    print(
        "After implicit conversion and 5-core pruning: "
        f"users={dataset_counts['users']}, items={dataset_counts['items']}, "
        f"interactions={dataset_counts['interactions']}"
    )

    rows = []
    for algorithm_name, (algorithm, config) in ALGORITHMS.items():
        for seed in SEEDS:
            rows.append(run_one_seed(base_dataset, seed, algorithm_name, algorithm, config))
    results = pd.DataFrame(rows).sort_values(["algorithm", "seed"]).reset_index(drop=True)
    stats = summarize(results)

    print("\nPer-run metrics")
    print(results.to_string(index=False))
    print("\nSeed summary by algorithm and metric")
    print(stats.to_string(index=False))
    print_interpretations(stats)

    results_path = os.path.join(working_dir, "prototype_als_pop_metrics_by_seed.csv")
    stats_path = os.path.join(working_dir, "prototype_als_pop_metrics_summary.csv")
    counts_path = os.path.join(working_dir, "prototype_als_pop_dataset_counts.json")
    config_path = os.path.join(working_dir, "prototype_als_pop_config.json")
    plot_path = os.path.join(working_dir, "prototype_als_pop_metrics_by_seed.png")
    interpretation_path = os.path.join(working_dir, "prototype_als_pop_seed_interpretation.txt")

    results.to_csv(results_path, index=False)
    stats.to_csv(stats_path, index=False)
    with open(counts_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_counts, handle, indent=2)
    config = {
        "dataset": "MovieLens100K",
        "algorithms": {name: algorithm.value for name, (algorithm, _) in ALGORITHMS.items()},
        "preprocessing": [
            "RatingFilter(lower=4), equivalent to rating > 3",
            "MakeImplicit(4)",
            "CorePruning(5)",
        ],
        "split": {
            "type": "user-wise random holdout",
            "test_size": 0.20,
            "validation_size": 0.0,
            "note": "UserHoldout creates its required tiny validation partition; it is merged into train, leaving no validation interactions.",
        },
        "seeds": SEEDS,
        "metrics": ["NDCG@10", "Precision@10"],
        "dataset_counts_after_preprocessing": dataset_counts,
        "execution": "OmniRec run_omnirec",
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    interpretation_lines = []
    for _, row in stats.iterrows():
        interpretation_lines.append(
            f"{row['algorithm']} {row['metric']}: best seed {int(row['best_seed'])}, "
            f"worst seed {int(row['worst_seed'])}, range {row['range']:.6f}."
        )
    interpretation_lines.append("Five split realizations quantify seed sensitivity but do not support reliable between-algorithm significance claims.")
    with open(interpretation_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(interpretation_lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, metric in zip(axes, ["NDCG@10", "Precision@10"]):
        for algorithm_name, group in results.groupby("algorithm"):
            ax.plot(group["seed"], group[metric], marker="o", linewidth=2, label=algorithm_name)
        ax.set_title(f"{metric} by data-split seed")
        ax.set_xlabel("Data-split random seed")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"Saved metrics to {results_path}")
    print(f"Saved summary to {stats_path}")
    print(f"Saved dataset counts to {counts_path}")
    print(f"Saved configuration to {config_path}")
    print(f"Saved interpretation to {interpretation_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
