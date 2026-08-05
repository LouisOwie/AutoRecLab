# Experiment Summary

## User Request
Run an experiment on MovieLens100K (implicit feedback) comparing two algorithms (ALS and Pop). Preprocess by (a) converting ratings > 3 to implicit interactions and (b) applying iterative 5-core filtering. Create 5 different random seeds for user-based 80/20 holdout splits. For each algorithm and seed train with fixed training seed and default hyperparameters. Measure nDCG@10 and Precision@10 and provide a short statistical analysis across seeds.

## What Was Run
- Dataset: MovieLens100K.
- Preprocessing:
  - Convert ratings > 3 (i.e., retain ratings >= 4) into implicit interactions (MakeImplicit(4)).
  - Iterative 5-core pruning (CorePruning(5)).
- Splitting: user-based two-way holdout (80% train / 20% test per user) using five split seeds: 11, 22, 33, 44, 55. No validation set.
- Algorithms: LensKit.ImplicitMFScorer (ALS) and LensKit.PopScorer (Pop) using OmniRec default configurations; training RNG seeded with TRAINING_SEED = 2027.
- Metrics: nDCG@10 and Precision@10; evaluation excludes items present in the split training for the same users and scores held-out positives.
- Repeats: one run per algorithm per split seed (5 splits × 2 algorithms = 10 runs).
- Artifacts: per-run results and summary JSONs saved under working directory; paired comparison computed across the five common splits.

Preprocessing counts (from experiment output):
- raw_interactions: 100000
- implicit_interactions (ratings > 3): 55375
- five_core_interactions: 54413

## Key Results
The captured experiment output includes the paired (ALS − Pop) comparisons across the five common splits. The full per-algorithm per-seed absolute metric values were not present in the provided output extract, but the paired statistics below are available and factual.

| Metric       | mean (ALS − Pop) | sample std (diff) | min diff | max diff | paired t-like statistic |
|--------------|------------------:|-------------------:|---------:|---------:|------------------------:|
| nDCG@10      | 0.022015345949876884 | 0.0023929545101876093 | 0.018061769221094065 | 0.024030284181943945 | 20.571979067098898 |
| Precision@10 | 0.027739872068230274 | 0.0038219281653619577 | 0.02132196162046908  | 0.03027718550106609  | 16.229567105387016 |

Per-seed ALS − Pop differences recorded in the experiment output:
- nDCG@10 differences by split seed:
  - seed 11: 0.023164077219407164
  - seed 22: 0.018061769221094065
  - seed 33: 0.024030284181943945
  - seed 44: 0.02152074352855285
  - seed 55: 0.023299855598386404
- Precision@10 differences by split seed:
  - seed 11: 0.030170575692963747
  - seed 22: 0.02132196162046908
  - seed 33: 0.03027718550106609
  - seed 44: 0.027078891257995757
  - seed 55: 0.02985074626865672

Additional notes:
- The experiment produced per-run CSVs (experiment_results_long.csv and per_run_results.csv) and JSON summaries, but the absolute per-algorithm means, sample stds, minima and maxima for each metric are not fully present in the provided output excerpt. Therefore the table above focuses on the paired differences that are explicitly available.

## Limitations
- The provided output extract does not include the complete per-algorithm per-seed absolute metric values or the printed "Across-seed summary" content in full; only the paired comparison JSON and preprocessing counts are fully present. As a result, I report only the paired (ALS − Pop) statistics that are explicitly shown in the output.
- The statistical comparison is descriptive and based on five paired splits (n = 5). The code computes a "paired_t_like_statistic" from the sample mean and sample standard deviation of the differences, but formal inference is limited by the small number of splits; the experiment’s own interpretation text notes this limitation.
- The training RNG was fixed (TRAINING_SEED = 2027), so the reported variability isolates split-seed effects (not training randomness).

## Conclusion
- Across the five user-holdout splits, ALS (LensKit.ImplicitMFScorer) consistently outperformed the popularity baseline (LensKit.PopScorer) on both metrics.
  - Mean advantage (ALS − Pop): ~0.022 nDCG@10 and ~0.028 Precision@10.
  - Differences across seeds were small but very consistent (sample std of differences ≈ 0.0024 for nDCG@10 and ≈ 0.0038 for Precision@10).
  - The experiment’s paired t-like statistics are large (≈20.57 for nDCG@10, ≈16.23 for Precision@10), reflecting very consistent positive differences across the five splits; however, inference is limited by the small number of splits.
- If you need absolute per-algorithm means, standard deviations, and min/max across seeds (rather than paired differences), provide the full experiment output (or the per_run_results.csv / experiment_summary.json produced by the run) and I will extract those exact values and update the table.