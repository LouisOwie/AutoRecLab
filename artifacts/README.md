# Compact Run Artifacts

This folder contains a compact, table-ID-aligned subset of the preserved
AutoRecLab artifacts. Raw datasets, NumPy arrays, pickle files, plots,
checkpoints, and other large generated files were intentionally omitted.

## Contents

Each available run keeps its README, execution log, code-requirements file,
and generated `runfile.py`. Runs 008 and 010 additionally keep selected CSV
and JSON result summaries because they contain the intermediate 45-row
artifacts referenced in the table.

The 45/45 value means that 45 expected result rows were observed in an
intermediate artifact. It does not establish technical completion or metric
correctness.
