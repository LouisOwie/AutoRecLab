# Curated Run Artifacts

This folder is a compact evidence copy for the Phase-2 run ledger. The
original `out_*` run directories were not modified.

## R1-R4 Mapping

| Ledger | Archived run | Result artifact |
|---|---|---|---:|---|
| R1 | `out_2026-07-12T01-55-07`| `prototype_als_pop_metrics_by_seed.csv` |
| R2 | `out_2026-07-12T02-40-20`| `per_run_results.csv` |
| R3 | `out_2026-07-12T03-28-31`| `als_split_seed_results.csv` |
| R4 | `out_2026-07-12T08-58-59`| `movielens100k_implicit_seed_raw_results.csv` |


Each run folder contains the root configuration, cost log, prompt, run
summary, selected `code.py`, selected `out.log`, selected-node statistics,
and the result/configuration files needed to inspect the reported outputs.

The `audit` folder contains the static verifier, verification reports,
recomputed aggregate statistics, and cross-run comparison report.

For Phase 2, `10/10` means two algorithms by five split seeds, with both
requested metrics present for every algorithm-seed row. It means artifact
completeness, not independent validation of preprocessing, splitting, or
metric computation. R4's selected node is marked `Is Buggy: True` in the
AutoRecLab node statistics, although its captured output contains the
complete result artifact.

Large or redundant files were intentionally omitted from this copy:
generated prediction JSON files, model checkpoints, dataset snapshots,
archives, plots, non-selected nodes, pickles, tree renders, and debug logs.
These files remain in the original `out_*` directories.
