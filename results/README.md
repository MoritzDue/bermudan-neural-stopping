# Results files

The results are separated into a small set of figures and headline statistics, and the detailed observations used to support or audit them.

## Headline results

| File | Purpose |
|---|---|
| [`headline/fig_neural_minus_lsm.png`](headline/fig_neural_minus_lsm.png) | Neural minus LSM across training budgets |
| [`headline/fig_primal_dual_bounds.png`](headline/fig_primal_dual_bounds.png) | Primal–dual estimates for selected policies |
| [`headline/H2.csv`](headline/H2.csv) | Paired neural–LSM means and across-seed intervals |
| [`headline/H3.csv`](headline/H3.csv) | Selected policies' dual estimates, gaps, and stability diagnostics |
| [`headline/summary_main.csv`](headline/summary_main.csv) | Per-budget policy-value and timing summaries |
| [`headline/summary.json`](headline/summary.json) | Machine-readable complete result summary |

`make_figures.py` reads **`headline/H2.csv`** and **`headline/H3.csv`**. Their paths and the output figure paths are unchanged.

## Supporting records

| File | Purpose |
|---|---|
| [`supporting/runs.csv`](supporting/runs.csv) | Individual policy/seed/budget evaluation records, including guard diagnostics and timing |
| [`supporting/paired.csv`](supporting/paired.csv) | Per-seed paired neural–LSM comparisons and guard contrasts |
| [`supporting/dates.csv`](supporting/dates.csv) | Exercise-date validation and checkpoint statistics |
| [`supporting/units.jsonl`](supporting/units.jsonl) | Frozen study-unit records and detailed audit data |
| [`SHA256SUMS`](SHA256SUMS) | SHA-256 hashes for all eight committed data files; paths are relative to this directory |

The supporting records are **preserved byte-for-byte** from the previous `results/headline/` directory. Their location is different, but the values and recorded study provenance have not been rewritten.

This is a curated publication layout, not the original simulation output folder. The research code and archival run are preserved at Git tag [`study-v1`](https://github.com/MoritzDue/bermudan-neural-stopping/tree/study-v1). The historical runner's execution-hash mismatch remains disclosed in [RESEARCH_NOTES.md](../RESEARCH_NOTES.md#reproducibility-and-scope).
