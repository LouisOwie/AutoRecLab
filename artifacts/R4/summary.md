# Experiment Summary

## User Request

Quantify the effect of data-split random seeds on recommender accuracy using ALS and Pop on MovieLens100K with implicit feedback, measuring nDCG@10 and Precision@10 across five seeded user-based 80/20 splits.

## What Was Run

- **Dataset:** MovieLens100K
- **Implicit conversion:** Ratings ≥ 4 were converted to implicit interactions.
- **Filtering:** Iterative 5-core filtering.
- **Preprocessed data:** 54,413 interactions, 938 users, and 1,008 items.
- **Split:** Per-user shuffled 80/20 train/test holdout with an empty validation set.
- **Seeds:** 11, 22, 33, 44, and 55.
- **Algorithms:** LensKit ALS (`ImplicitMFScorer`) and popularity ranking (`PopScorer`), using empty configuration dictionaries.
- **Evaluation:** User-averaged nDCG@10 and Precision@10 over held-out positives. Training positives were excluded from recommendation candidate lists.

## Key Results

| Algorithm | Metric | Mean | Std. dev. | Median | Min–Max | 95% CI for mean |
|---|---:|---:|---:|---:|---:|---:|
| ALS | nDCG@10 | 0.159656 | 0.002882 | 0.158524 | 0.157247–0.163825 | 0.157130–0.162183 |
| ALS | Precision@10 | 0.148380 | 0.002752 | 0.147548 | 0.145736–0.152878 | 0.145968–0.150791 |
| Pop | nDCG@10 | 0.136706 | 0.001027 | 0.136601 | 0.135136–0.137841 | 0.135807–0.137606 |
| Pop | Precision@10 | 0.119531 | 0.000688 | 0.119510 | 0.118550–0.120469 | 0.118927–0.120134 |

### Exploratory paired comparison

| Metric | Mean ALS − Pop | Std. dev. | Difference range | 95% CI |
|---|---:|---:|---:|---:|
| nDCG@10 | 0.022950 | 0.003037 | 0.019906–0.027224 | 0.020288–0.025612 |
| Precision@10 | 0.028849 | 0.002965 | 0.025267–0.033369 | 0.026250–0.031447 |

Across the five seeds, ALS had higher mean accuracy than Pop for both metrics. Seed variation was measurable: ALS had a larger absolute standard deviation than Pop for both nDCG@10 and Precision@10. The paired comparisons are exploratory and do not establish statistical significance.

## Limitations

- Only five split seeds were tested, so the statistical analysis is limited.
- The output reports confidence intervals and paired differences but no formal hypothesis test; therefore, no significance claim can be made.
- The algorithm configuration dictionaries were empty. The output does not expose the underlying LensKit default hyperparameter values, so “standard hyperparameters” cannot be specified more precisely from the provided materials.
- The raw per-seed results were saved to the reported CSV path, but the summary above focuses on the aggregate values shown in the experiment output.

## Conclusion

For this MovieLens100K implicit-feedback experiment, ALS achieved higher average nDCG@10 and Precision@10 than Pop across all five split seeds. Random split seeds affected the measured accuracy, with greater observed variability for ALS than Pop in absolute terms. The five-seed results quantify this variation descriptively, but they are insufficient for a definitive statistical significance conclusion.