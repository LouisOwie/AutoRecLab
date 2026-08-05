# AutoRecLab Evaluation: Comprehensive Run-by-Run Analysis

## 1. What Was Tested

**AutoRecLab** is an agentic research system that autonomously designs, writes, and executes recommender system experiments. Given a natural-language prompt, it generates Python code to preprocess data, train models, measure metrics, and produce statistical analyses. All 4 runs were provided an **identical prompt** and executed under the same tree-search hyperparameters. The sole experimental variable was the **LLM backend configuration** — specifically, which models were assigned to the Planner, Coder, Reviewer, and Summarizer agent roles.

---

## 2. The Prompt (Identical Across All Runs)

> *"I'd like to run an experiment to quantify how much data split random seeds affect recommender system accuracy. Please test two algorithms: ALS and Pop. Run this on the MovieLens100K dataset with implicit feedback. First, preprocess the dataset with 5-core filtering and convert any ratings greater than 3 to implicit interactions. Generate 5 different random seeds for data splitting. For each algorithm and seed, do a user-based 80/20 holdout split and train the models using standard hyperparameters. Measure nDCG@10 and Precision@10 and conduct a short statistical analysis across seeds."*

---

## 3. Tree-Search & Execution Configuration (Identical Across All Runs)

| Parameter | Value |
|---|---|
| `treesearch.num_draft_nodes` | 2 |
| `treesearch.debug_prob` | 0.3 |
| `treesearch.epsilon` | 0.4 |
| `treesearch.max_iterations` | 8 |
| `treesearch.refinement_iterations` | 2 |
| `exec.timeout` | 5400 s |
| `exec.enable_type_checking` | true |
| `exec.max_type_check_attempts` | 3 |
| `agent.k_fold_validation` | 1 |
| `agent.request_timeout` | 120 s |
| `agent.max_retries` | 3 |
| Total nodes produced per run | 13 |

---

## 4. LLM Model Configurations (The Variable Under Test)

| Run | Mode | Planner | Coder | Reviewer | Summarizer |
|---|---|---|---|---|---|
| **R1** | Mixed (multi-agent) | `gpt-5.6-terra` | `gpt-5.6-luna` | `gpt-5.4-mini` | `gpt-5-mini` |
| **R2** | Mixed (multi-agent, replica of R1) | `gpt-5.6-terra` | `gpt-5.6-luna` | `gpt-5.4-mini` | `gpt-5-mini` |
| **R3** | Single-agent | `gpt-5.6-terra` (all roles) | `gpt-5.6-terra` | `gpt-5.6-terra` | `gpt-5.6-terra` |
| **R4** | Single-agent | `gpt-5.6-luna` (all roles) | `gpt-5.6-luna` | `gpt-5.6-luna` | `gpt-5.6-luna` |

**Key insight:** R1 and R2 share the exact same model configuration — they are mixed-agent replicas of each other. R3 and R4 each use a single model for all agent roles, testing `gpt-5.6-terra` and `gpt-5.6-luna` respectively as end-to-end solo agents.

---

## 5. Cost & Efficiency Summary

| Metric | R1 (mixed) | R2 (mixed) | R3 (gpt-5.6-terra) | R4 (gpt-5.6-luna) |
|---|---|---|---|---|
| **API calls** | 187 | 203 | 156 | 197 |
| **Duration (min)** | 43.95 | 43.93 | 62.33 | 46.18 |
| **Prompt tokens** | 2,032,218 | 1,922,493 | 3,181,869 | 2,895,598 |
| **Completion tokens** | 68,925 | 83,462 | 125,582 | 123,963 |
| **Total tokens** | 2,101,143 | 2,005,955 | 3,307,451 | 3,019,561 |
| **Total cost (USD)** | $2.38 | $2.39 | $9.84 | $3.64 |
| **Cost per minute (USD)** | $0.054 | $0.054 | $0.158 | $0.079 |
| **Avg node score** | 0.414 | 0.429 | 0.620 | 0.475 |
| **Avg exec time per node (s)** | 134.6 | 129.1 | 156.7 | 96.2 |
| **Avg LOC per node** | 147.5 | 179.8 | 248.5 | 148.3 |

### Cost Observations
- **R3 (gpt-5.6-terra solo)** was by far the most expensive at **$9.84** (~4.1× R1/R2) and slowest at 62.3 min, driven by much higher prompt token usage (3.18M vs ~2M).
- **R4 (gpt-5.6-luna solo)** was notably cheaper than R3 at $3.64, with similar total call count but much lower prompt-token pricing ($0.001/token vs $0.0025/token for terra).
- **R1 and R2 (mixed)** were the most cost-efficient at ~$2.38 each, with prompt work distributed across cheaper reviewer models (gpt-5.4-mini, gpt-5-mini) rather than the expensive terra/luna models handling every call.
- The mixed configuration achieved the fastest execution (~44 min) and lowest cost per minute ($0.054).

---

## 6. Role-Based Token Breakdown (Mixed Runs Only)

### R1 (mixed)
| Role | Prompt Tokens | Completion Tokens | Total Tokens | Total USD |
|---|---|---|---|---|
| Planner | 69,264 | 3,504 | 72,768 | $0.23 |
| Coder | 1,270,794 | 49,477 | 1,320,271 | $1.57 |
| Reviewer | 688,060 | 14,081 | 702,141 | $0.58 |
| Summarizer | 4,100 | 1,863 | 5,963 | $0.005 |

### R2 (mixed)
| Role | Prompt Tokens | Completion Tokens | Total Tokens | Total USD |
|---|---|---|---|---|
| Planner | 101,653 | 4,982 | 106,635 | $0.33 |
| Coder | 1,032,477 | 61,853 | 1,094,330 | $1.40 |
| Reviewer | 783,690 | 14,116 | 797,806 | $0.65 |
| Summarizer | 4,673 | 2,511 | 7,184 | $0.006 |

**Observation:** In both mixed runs, the Coder role dominated token usage (~63% in R1, ~55% in R2), followed by Reviewer (~33% / 40%). The Planner and Summarizer used negligible resources (<5% each).

---

## 7. Experiment Results (What Each Agent Produced)

All runs successfully executed the experiment: MovieLens100K preprocessing (ratings > 3 → implicit, 5-core → 938 users, 1,008 items, 54,413 interactions), user-based 80/20 holdout, 5 random seeds, ALS vs. Pop with NDCG@10 and Precision@10 metrics.

### Libraries Used (All Runs)
- **LensKit** (ImplicitMFScorer for ALS, PopScorer for Pop)
- **Preprocessing:** RatingFilter, MakeImplicit, CorePruning
- **Splitting:** UserHoldout (disjoint per-user 80/20)
- **Metrics:** NDCG@10, Precision@10

### Run-by-Run Results

#### R1 (Mixed — gpt-5.4-mini / terra / luna)
**Seeds:** [11, 23, 37, 41, 59]
**ALS hyperparameters:** embedding_size=64, epochs=10, regularization=0.1, weight=1.0, use_ratings=False

| Algorithm | Metric | Mean | Std | Min | Max | Range |
|---|---|---|---|---|---|---|
| ALS | NDCG@10 | 0.184714 | 0.001856 | 0.182073 | 0.186705 | 0.004633 |
| ALS | Precision@10 | 0.164968 | 0.002889 | 0.160235 | 0.167271 | 0.007036 |
| Pop | NDCG@10 | 0.139935 | 0.002240 | 0.137367 | 0.142582 | 0.005214 |
| Pop | Precision@10 | 0.122964 | 0.003204 | 0.118550 | 0.126759 | 0.008209 |

**Analysis quality:** Descriptive statistics only (mean, std, min, max, range). No hypothesis test performed. The summary explicitly states: *"These five split realizations describe seed sensitivity; they do not establish statistically reliable differences between algorithms."*

---

#### R2 (Mixed — replica of R1 config)
**Seeds:** [11, 22, 33, 44, 55]
**ALS hyperparameters:** (not fully specified — "standard hyperparameters"; training seed = 2027)

The agent produced **only paired difference statistics** (ALS − Pop across common splits) rather than absolute per-algorithm values:

| Metric | Mean (ALS − Pop) | Std (diff) | Paired t-like stat |
|---|---|---|---|
| nDCG@10 | 0.022015 | 0.002393 | 20.57 |
| Precision@10 | 0.027740 | 0.003822 | 16.23 |

ALS consistently outperformed Pop across all 5 seeds, with very small variance in the differences.

**Analysis quality:** The agent computed a "paired t-like statistic" but noted that n=5 limits formal inference. Notably, R2's summary is less complete than R1's — it lacks the absolute per-algorithm per-seed values, presumably because the agent only printed the paired comparison JSON.

---

#### R3 (gpt-5.6-terra solo)
**Seeds:** [11, 23, 37, 53, 71]
**ALS hyperparameters:** embedding_size=64, epochs=10, regularization=0.1, weight=40, implicit ratings disabled, user embeddings enabled. Training seed = 2026.

| Algorithm | Metric | Mean | Std | Min | Max | Range | CV |
|---|---|---|---|---|---|---|---|
| ALS | NDCG@10 | 0.162859 | 0.004358 | 0.155384 | 0.166520 | 0.011136 | 2.68% |
| ALS | Precision@10 | 0.148977 | 0.002389 | 0.145096 | 0.150746 | 0.005650 | 1.60% |
| Pop | NDCG@10 | 0.137836 | 0.004264 | 0.131296 | 0.142512 | 0.011216 | 3.09% |
| Pop | Precision@10 | 0.121130 | 0.003272 | 0.116205 | 0.124520 | 0.008316 | 2.70% |

**Analysis quality:** The most comprehensive of all 4 runs. R3 included mean, std, min-max, range, coefficient of variation (CV), per-split results table, and comparative analysis of relative variability. ALS showed lower relative seed sensitivity than Pop (CV of 1.60% vs 2.70% for Precision@10). The gpt-5.6-terra solo agent produced the most thorough statistical breakdown.

---

#### R4 (gpt-5.6-luna solo)
**Seeds:** [11, 22, 33, 44, 55]
**ALS hyperparameters:** empty configuration dictionaries — LensKit defaults used.

| Algorithm | Metric | Mean | Std | Median | Min | Max | 95% CI for mean |
|---|---|---|---|---|---|---|---|
| ALS | NDCG@10 | 0.159656 | 0.002882 | 0.158524 | 0.157247 | 0.163825 | 0.157130–0.162183 |
| ALS | Precision@10 | 0.148380 | 0.002752 | 0.147548 | 0.145736 | 0.152878 | 0.145968–0.150791 |
| Pop | NDCG@10 | 0.136706 | 0.001027 | 0.136601 | 0.135136 | 0.137841 | 0.135807–0.137606 |
| Pop | Precision@10 | 0.119531 | 0.000688 | 0.119510 | 0.118550 | 0.120469 | 0.118927–0.120134 |

ALS NDCG@10 mean: 0.159656 vs Pop: 0.136706. ALS Precision@10 mean: 0.148380 vs Pop: 0.119531.

**Analysis quality:** Included confidence intervals (95% CI for the mean) and exploratory paired comparison, but noted *"no formal hypothesis test; therefore, no significance claim can be made."* ALS variability in absolute terms was larger than Pop, opposite to R3's finding — possibly due to different hyperparameter choices (empty config = LensKit defaults) and different seed sets.

---

## 8. Key Differences Between Runs

### 8.1 Seeds Chosen by Each Agent
The prompt said *"Generate 5 different random seeds"* — the agent chose them autonomously:

| Run | Seeds | Notes |
|---|---|---|
| R1 | 11, 23, 37, 41, 59 | Non-uniform spacing |
| R2 | 11, 22, 33, 44, 55 | Uniform spacing |
| R3 | 11, 23, 37, 53, 71 | Near-uniform with prime-like pattern |
| R4 | 11, 22, 33, 44, 55 | Same as R2 (uniform) |

R2 and R4 chose identical seeds despite different LLM backends. R1 and R3 chose different patterns.

### 8.2 ALS Hyperparameters Chosen
The prompt said *"standard hyperparameters"* — the agent decided what "standard" means:

| Run | Weight | Training Seed | Other Differences |
|---|---|---|---|
| R1 | 1.0 | not fixed | epochs=10, embedding_size=64 |
| R2 | default | 2027 (fixed) | "standard hyperparameters" |
| R3 | 40 | 2026 (fixed) | epochs=10, embedding_size=64 |
| R4 | default | not specified | empty config dict |

**Critical finding:** The ALS `weight` parameter varied from 1.0 (R1) to 40 (R3) to unspecified/default (R2, R4), demonstrating that the agent's LLM backend significantly influences what it considers "standard."

### 8.3 Statistical Analysis Depth
- **R1:** Basic descriptive (mean, std, min, max, range). Explicitly considered incomplete.
- **R2:** Paired differences only, with t-like statistic. Less complete per-algorithm reporting.
- **R3:** Most thorough — mean, std, min/max, range, CV, per-split table, comparative analysis. **Best overall statistical breakdown.**
- **R4:** Mean, std, median, min/max, 95% confidence intervals, exploratory paired comparison.

### 8.4 Result Metric Differences
The same experiment produced different absolute ALS NDCG@10 means: **0.185 (R1) → 0.163 (R3) → 0.160 (R4)**. This is due to the combination of different seeds, different ALS hyperparameters, and possibly different evaluation implementations generated by each agent. Pop results were more stable across runs (0.137–0.140 for NDCG@10).

---

## 9. Summary of Agent Performance

| Criterion | R1 (Mixed) | R2 (Mixed) | R3 (gpt-5.6-terra) | R4 (gpt-5.6-luna) |
|---|---|---|---|---|
| **Completed task?** | Yes | Yes | Yes | Yes |
| **Statistical depth** | Basic | Medium (paired only) | Excellent (best) | Good (CI included) |
| **Reproducibility** | Full per-seed values | Incomplete | Full per-seed values | Full per-seed values |
| **Cost** | $2.38 (cheapest) | $2.39 | $9.84 (most expensive) | $3.64 |
| **Speed** | 43.9 min | 43.9 min | 62.3 min (slowest) | 46.2 min |
| **Avg node score** | 0.414 | 0.429 | 0.620 (highest) | 0.475 |
| **Avg LOC** | 147.5 | 179.8 | 248.5 (most code) | 148.3 |
| **Code efficiency** | Good | Good | Verbose | Good |

---

## 10. Cross-Run Conclusions

1. **All 4 runs successfully executed the experiment** — AutoRecLab is reliable in producing working experiment code from natural-language prompts, regardless of the LLM backend.

2. **Single-agent gpt-5.6-terra (R3) produced the best-quality output** (highest node score 0.620, most comprehensive statistical analysis, highest code volume at 248.5 avg LOC) but at **4× the cost** of the mixed configuration ($9.84 vs $2.38).

3. **The mixed-agent configuration (R1/R2) is the most cost-effective**, achieving the task at ~$2.38 with reasonable output quality. It distributes expensive prompt processing to cheaper models (gpt-5.4-mini for reviewing) while reserving the powerful terra/luna models for planning and coding only.

4. **R1 and R2 (identical configs) produced different results** — different seeds chosen, different ALS hyperparameters, different analysis approach (R1: per-algorithm values; R2: paired differences only). This demonstrates inherent non-determinism in agentic behavior.

5. **gpt-5.6-luna solo (R4) offers a middle ground**: $3.64, good analysis quality with confidence intervals, but lower node scores than both terra-solo and mixed, and the fastest per-node execution time (96.2s avg).

6. **The agent's interpretation of "standard hyperparameters" varies** — the ALS weight parameter ranged from 1.0 to 40 across runs, and the training seed was sometimes fixed and sometimes not. This is expected autonomous behavior but highlights that agent-chosen "defaults" are not consistent.

7. **Seed sensitivity is measurable but small** — across all runs, the range of NDCG@10 across 5 seeds was 0.0046–0.0112 depending on the agent's seed choices, confirming that split randomness has a measurable but modest effect.

---

## 11. Libraries Used (Across All Runs)

| Library | Purpose |
|---|---|
| **LensKit** (`lenskit`) | Recommender algorithms: `ImplicitMFScorer` (ALS), `PopScorer` (popularity) |
| **LensKit preprocessing** | `RatingFilter`, `MakeImplicit`, `CorePruning` — data preprocessing pipeline |
| **LensKit splitting** | `UserHoldout` — user-wise 80/20 holdout split |
| **LensKit metrics** | NDCG@10, Precision@10 — ranking evaluation |
| **NumPy / pandas** | Data manipulation, statistical calculations |
| **CSV / JSON I/O** | Saving per-run results and experiment summaries |

---

## 12. Reproducibility Notes

- Configuring the `TRAINING_SEED` (R2: 2027, R3: 2026) isolates split-seed variability from model-training randomness; R1 and R4 did not fix it.
- The identical prompt and tree-search parameters mean the model configuration is the **only intentional variable** across runs.
- R1 nd R2 are useful as within-config replicates, showing how much variation occurs when the same multi-agent setup is run twice.
