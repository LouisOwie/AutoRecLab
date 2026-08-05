# Experiment Summary

## User Request
Run an experiment to quantify how much data-split random seeds affect recommender accuracy for two algorithms (ALS and Pop) on MovieLens100K with implicit feedback. Preprocess with 5-core filtering and convert ratings > 3 to implicit interactions. Use 5 different random seeds for user-wise 80/20 holdout splits. Train ALS and Pop with the code's standard hyperparameters. Measure NDCG@10 and Precision@10 and perform a short statistical analysis across seeds.

## What Was Run
- Dataset: MovieLens100K, preprocessed by:
  - RatingFilter(lower=4) (equivalent to rating > 3)
  - MakeImplicit(4)
  - CorePruning(5) (5-core)
- Resulting dataset counts after preprocessing (from run output):
  - users = 938
  - items = 1008
  - interactions = 54,413
- Split: user-wise random holdout (UserHoldout) with test_size=0.20; a tiny validation partition created by UserHoldout was merged into train so the experiment used train/test only.
- Seeds tested: [11, 23, 37, 41, 59]
- Algorithms:
  - ALS: LensKit.ImplicitMFScorer with embedding_size=64, epochs=10, regularization=0.1, user_embeddings=True, use_ratings=False, weight=1.0
  - Pop: LensKit.PopScorer (default config)
- Metrics: NDCG@10 and Precision@10 (ranking metrics)
- Execution: For each (algorithm, seed) the model was trained and evaluated; per-seed metrics and a summary across seeds were computed and printed/saved.

## Key Results

- Per-run metrics (excerpted from experiment output):
  - ALS NDCG@10 by seed: 0.185467, 0.183600, 0.186705, 0.185726, 0.182073
  - ALS Precision@10 by seed: 0.166951, 0.160235, 0.167271, 0.166098, 0.164286
  - Pop NDCG@10 by seed: 0.137367, 0.142582, 0.141096, 0.137838, 0.140790
  - Pop Precision@10 by seed: 0.118550, 0.126759, 0.122601, 0.121642, 0.125267

- Summary statistics across the five seeds (from the run's "Seed summary by algorithm and metric"):

| algorithm | metric        | mean     | std      | min      | max      | best_seed | worst_seed | range    |
|-----------|---------------|----------:|---------:|---------:|---------:|----------:|-----------:|---------:|
| ALS       | NDCG@10       | 0.184714 | 0.001856 | 0.182073 | 0.186705 | 37        | 59        | 0.004633 |
| ALS       | Precision@10  | 0.164968 | 0.002889 | 0.160235 | 0.167271 | 37        | 23        | 0.007036 |
| Pop       | NDCG@10       | 0.139935 | 0.002240 | 0.137367 | 0.142582 | 23        | 11        | 0.005214 |
| Pop       | Precision@10  | 0.122964 | 0.003204 | 0.118550 | 0.126759 | 23        | 11        | 0.008209 |

- Direct factual observations from the output:
  - Across all five seeds, ALS achieved higher NDCG@10 and Precision@10 than Pop (see per-run metrics and means above).
  - Seed-to-seed variation (range and sample standard deviation) is small in absolute terms:
    - NDCG@10 range: ALS = 0.004633, Pop = 0.005214
    - Precision@10 range: ALS = 0.007036, Pop = 0.008209
  - The code prints the interpretation: "These five split realizations describe seed sensitivity; they do not establish statistically reliable differences between algorithms."

## Limitations
- Only five random seeds were tested. The sample size is small for formal statistical inference.
- The experiment presents descriptive statistics (mean, std, min, max, range) only; no hypothesis test (e.g., paired t-test or nonparametric test) comparing algorithms across seeds was performed in the provided output.
- Results are specific to the preprocessing choices, hyperparameters used (ALS configuration shown in code), and Random seeds listed; no hyperparameter search or cross-dataset generalization was performed.
- All information and numbers reported above are taken directly from the provided code and experiment output. If additional analyses (statistical tests, confidence intervals, more seeds) are required, they were not present in the supplied output.

## Conclusion
- The provided experiment ran ALS and Pop on MovieLens100K (implicit, 5-core) over five user-wise 80/20 splits (seeds 11, 23, 37, 41, 59).
- ALS outperformed Pop on both metrics in these runs (mean NDCG@10: 0.184714 vs 0.139935; mean Precision@10: 0.164968 vs 0.122964).
- Data-split seed caused small but measurable variation in metrics across the five realizations (ranges up to ~0.0082 in Precision@10).
- The experiment's authors note that these five realizations quantify seed sensitivity but do not establish statistically reliable between-algorithm significance. For formal significance claims or more robust seed-sensitivity estimates, increase the number of seeds and run appropriate statistical tests (not present in the current output).

Saved outputs referenced in the run (paths shown in the experiment output) include the per-run CSV, summary CSV, config, interpretation text, and a plot of metrics by seed.