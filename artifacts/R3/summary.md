# Experiment Summary

## User Request

Quantify the effect of random data-split seeds on recommender accuracy for ALS and Pop using MovieLens100K implicit feedback, five user-based 80/20 holdout splits, and the metrics NDCG@10 and Precision@10.

## What Was Run

- **Preprocessing:** Ratings greater than 3 were converted to implicit interactions, followed by iterative 5-core filtering.
  - Retained **938 users**, **1,008 items**, and **54,413 interactions**.
- **Split seeds:** 11, 23, 37, 53, and 71.
- **Splitting:** Disjoint per-user holdout splits. The observed split for seed 11 contained 43,158 training and 11,255 test interactions (20.68% test); each evaluated user retained train and test interactions.
- **Algorithms:**
  - **ALS:** LensKit `ImplicitMFScorer`, with embedding size 64, 10 epochs, regularization 0.1, weight 40, implicit ratings disabled, and user embeddings enabled.
  - **Pop:** LensKit `PopScorer` with its default configuration.
- **Evaluation:** NDCG@10 and Precision@10.
- The training random seed was fixed at **2026** for every run, so the reported variation is across the five data-split seeds.

## Key Results

| Algorithm | Metric | Mean | Sample SD | Min–Max | Range | CV |
|---|---:|---:|---:|---:|---:|---:|
| ALS | NDCG@10 | 0.162859 | 0.004358 | 0.155384–0.166520 | 0.011136 | 2.68% |
| ALS | Precision@10 | 0.148977 | 0.002389 | 0.145096–0.150746 | 0.005650 | 1.60% |
| Pop | NDCG@10 | 0.137836 | 0.004264 | 0.131296–0.142512 | 0.011216 | 3.09% |
| Pop | Precision@10 | 0.121130 | 0.003272 | 0.116205–0.124520 | 0.008316 | 2.70% |

Per-split results:

| Split seed | ALS NDCG@10 | ALS Precision@10 | Pop NDCG@10 | Pop Precision@10 |
|---:|---:|---:|---:|---:|
| 11 | 0.155384 | 0.145096 | 0.140031 | 0.120896 |
| 23 | 0.163469 | 0.148294 | 0.138935 | 0.124520 |
| 37 | 0.165229 | 0.150746 | 0.142512 | 0.123667 |
| 53 | 0.163693 | 0.150746 | 0.131296 | 0.116205 |
| 71 | 0.166520 | 0.150000 | 0.136405 | 0.120362 |

- ALS had higher mean accuracy than Pop on both metrics across all five splits.
- Split-seed variability was modest in absolute terms for both methods, but Pop had higher relative variability than ALS for both metrics based on the coefficient of variation.
- The observed NDCG@10 range was nearly identical for ALS (0.011136) and Pop (0.011216), while Pop had a larger Precision@10 range.

## Limitations

- The statistical analysis is descriptive only: mean, sample standard deviation, range, and coefficient of variation across five split seeds. No statistical-significance test or confidence interval was reported.
- Results apply only to this preprocessing procedure, the five specified splits, cutoff 10, and the stated algorithm configurations.
- The experiment fixes the training seed, so it does not measure variability due to model-training randomness.

## Conclusion

Data-split seed affected measured accuracy for both algorithms. ALS was consistently stronger on average, with mean NDCG@10 of 0.162859 and Precision@10 of 0.148977, compared with Pop’s 0.137836 and 0.121130. Across the five splits, ALS also showed lower relative seed sensitivity than Pop, particularly for Precision@10.