# Payoff-aware neural optimal stopping vs Longstaff–Schwartz

A four-week MSc research project based on Becker, Cheridito & Jentzen (2019), *Deep Optimal Stopping*, JMLR 20(74). It builds on the paper's method; it does not reproduce the paper.

**Main question.** How much training data and computation does direct neural optimal stopping need to become competitive with Longstaff–Schwartz (LSM) on a 5D Bermudan max-call?

**Mechanism.** Do explicit payoff information and a permutation-invariant (sorted) input representation reduce that requirement?

**Validation.** Do both methods produce credible values against a 1D CRR tree and the paper's published 2D/5D references?

A negative result (LSM at least as good at every budget, after adequate neural training) is a valid finding, and the project is set up to report it.

## Files

```
README.md
requirements.txt
stopping.py                         all methods, in one file, read top to bottom
01_method_validation.ipynb          correctness checks + paper-regime implementation check
02_calibration_and_preflight.ipynb  development experiment -> frozen neural settings; runtime preflight
03_main_study.ipynb                 the study and all analysis (tables, figures, intervals)
results/quick/                      outputs of the included quick run (smoke test, NOT results)
```

`results/full/` is created when you run full mode.

## Setup (Windows, VS Code)

Open the `bermudan-optimal-stopping` folder itself in VS Code. Then, in a PowerShell terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open a notebook, click **Select Kernel → Python Environments → .venv**, and click **Run All**.

## How to run

Each notebook's first cell has `MODE = "quick"`. Quick mode is a smoke test that runs all three notebooks in about 1–2 minutes. Its numbers are **not** results.

For the research run, set `MODE = "full"` in each notebook and run them **in order**:

| Step | Notebook | Full-mode time on one CPU thread* |
|---|---|---|
| 1 | `01_method_validation` | ~10–15 min (8 min of this is the paper-regime check) |
| 2 | `02_calibration_and_preflight` | ~25–35 min; writes `frozen_settings.json`. Then set `RUN_PREFLIGHT = True` in its last cell once (a few minutes) to write `preflight.json` |
| 3 | `03_main_study` | ~25–35 min; prints the preflight projection before it starts |

\*Measured on the build machine: one 5D network fit with 1,500 steps per date ≈ 11 s; LSM tuning at 50k paths ≈ 6 s; evaluating 200k test paths ≈ 0.3 s. A slower laptop may take 2–3× longer, which is still well inside 3–4 hours. Notebooks 02 and 03 save after every fit or unit, so if interrupted you just run them again and they resume. Notebook 03 writes each finished unit as one line of `study/units.jsonl` in a single append, so a unit is saved completely or not at all. If a write was interrupted, the incomplete last line is removed before the next append.

**Before running notebook 03**, read notebook 02's output. If it flags a *selection* variant (sorted+payoff or raw+payoff) as not saturated, add a larger limit to `CONFIG["full"]["calibration"]["max_steps"]` and rerun notebook 02, or report the warning. A flagged *diagnostic* variant (plain raw) is a finding, not a blocker. In full mode notebook 03 **enforces** this: it refuses to start while a selection variant is flagged, unless you deliberately set `ACCEPT_RESTRICTED_BUDGET = True`. That choice is frozen with the run, and the headline figures and `summary_main.csv` are then labelled "restricted optimisation budget", meaning the comparison is under a limited training budget, not between adequately converged methods. Decide this **before** looking at any main-study result.

## The frozen specification

All settings live in `CONFIG` at the top of `stopping.py`. Notebook 03 records them, together with the contract parameters and a hash of `stopping.py`, in `results/full/study/run_info.json`, and refuses to resume a full run if any of them change. Any edit to `stopping.py`, even a comment, counts as a change.

| Component | Specification |
|---|---|
| Problem | 5D symmetric max-call, Table 1 parameters: S0 = K = 100, r = 5 %, δ = 10 %, σ = 20 %, ρ = 0, T = 3, 9 exercise dates |
| Reference | 26.159, with the paper's 95 % CI [26.138, 26.174] shown as a band. Gaps within about ±0.02 are not interpretable |
| Methods | Tuned LSM (degree 1–4 × 3 ridge penalties, chosen per seed on validation) vs direct neural stopping with sorted inputs + payoff |
| Training budgets | 1k, 10k, 50k paths per seed (nested within a seed) |
| Validation | 5k paths per seed, shared by all methods |
| Test | 200k paths, fixed and shared by every method and seed |
| Seeds | 10 per budget. A seed redraws the training and validation paths and the network initialisation |
| Ablation | {sorted, raw} × {payoff input, no payoff input} at 1k and 50k |
| Neural network | Paper architecture: 2 hidden layers of width d+40, ReLU, batch norm, Adam. Fixed number of steps per date, best validation checkpoint kept |
| Calibration | Separate development seeds (100–102) and a separate 100k *calibration selection set*. Step limits 750/1,500/3,000 × learning rates 1e-3/3e-3 × budgets 1k/50k. The selection rule uses sorted+payoff and raw+payoff; plain raw is calibrated as a diagnostic only |
| Validation checks | 1D put vs CRR; LSM convergence; paper-regime neural run (fresh paths, 3,000 × 8,192); 2D at 10k vs 13.902 |
| Optional | 10D (`INCLUDE_10D = True` in notebook 03); fresh-path training (`NeuralSettings(fresh_paths=True)`) |

## Methodological rules built into the code

- **Nothing is selected on test paths.** LSM's degree and ridge penalty, neural checkpoints and the guard all use validation paths. Neural settings come from notebook 02's development experiment, using different seeds and a different holdout set. The test set is used only for final evaluation.
- **Same paths for competing methods.** Within a seed, every method gets identical training and validation paths, and all seeds share one test set. Comparisons are paired per path.
- **The data budget includes validation.** "1,000 training paths" means 6,000 simulated paths per seed in total, because every method also uses the 5k validation paths. The study measures training efficiency *conditional on a fixed 5k validation budget*. It does not show performance when the whole simulation budget is 1,000 paths. Figures show both counts.
- **Never-exercise guard (both methods).** At each date, the fitted rule is compared on validation paths with "never exercise here" (both followed by the same later policy). If its gain is negative, the guarded policy does not exercise at that date. Results are reported **guarded and unguarded**, along with how often and at which dates the guard fired. Because earlier dates were trained against the guarded later policy, "unguarded" is an execution-level ablation of the same fitted models, not an independently trained procedure. Frequent guarding of a neural variant signals undertraining.
- **Saturation, not proof of convergence.** Notebook 02 shows whether raising the step limit still improves the policy value, with each improvement's interval across development seeds. The 0.02 tolerance is a pre-specified practical threshold, not a statistical test. This is empirical training saturation. It does not show the policy is optimal.
- **Claim A vs Claim B.** All four ablation variants use the same frozen protocol, so the ablation measures effects *under a common training budget* (does the representation make the network learn faster?). Whether plain raw catches up with more steps is read from notebook 02's diagnostic curves.
- **Compute cost = full training time.** The headline cost is the full fixed-step training time, because the best checkpoint is only known after training past it. The best-checkpoint step is a diagnostic in `dates.csv`.
- **Two kinds of uncertainty, kept apart.** Intervals across seeds (Student-t over 10 seeds) describe training variability, conditional on the shared test set. Each per-seed paired comparison also has its own 95 % Monte Carlo interval (`mc_ci_low`, `mc_ci_high` in `paired.csv`). Evaluations on the same test paths are never pooled as if independent.
- **Ablation interpretation.** The four cell means, the conditional effects and the interaction are all reported. No fixed percentage of the improvement is attributed to either mechanism, and conclusions are specific to this architecture and budget.
- **Fair timing.** NumPy's BLAS (used by LSM) and PyTorch are both limited to one thread via `threadpoolctl`. Costs are reported separately: simulation, final fit, per-seed selection (LSM's grid search), evaluation, and the one-off neural calibration from notebook 02 (both summed fitting time and total elapsed time).
- **Lower bounds.** Every value is the average discounted payoff of a fixed policy on independent paths, so in population it is a lower bound on the true price. No dual upper bound is computed.

## Outputs (in `results/full/`)

| File | Contents |
|---|---|
| `validation_*.csv/png/json` | Notebook 01 checks |
| `calibration.csv`, `calibration_saturation.png`, `frozen_settings.json`, `preflight.json` | Notebook 02 |
| `study/units.jsonl` (source of truth), `study/run_info.json`; `study/runs.csv`, `paired.csv`, `dates.csv` (convenience copies) | Raw per-unit results from notebook 03 |
| `summary_main.csv`, `summary_nn_minus_lsm.csv` | Headline gaps, costs, paired differences |
| `summary_ablation_cells.csv`, `summary_ablation_effects.csv` | 2×2 ablation |
| `summary_guard.csv`, `summary_other_dimensions.csv` | Guard diagnostics; 2D (and 10D) |
| `fig_gap_vs_data_and_compute.png`, `fig_ablation.png` | Headline figures |

## Limitations to state in the write-up

The Black–Scholes testbed was chosen because reference values exist. Values are lower bounds only. The study uses one network architecture and a fixed 5k validation budget. Results are conditional on the shared test set. The frozen settings were selected on the sorted+payoff and raw+payoff variants and applied to all four variants, so ablation effects hold under a common training protocol. Calibration selection-set values were used for choosing settings and are not independent performance estimates.

## References

- Becker, S., Cheridito, P., & Jentzen, A. (2019). Deep Optimal Stopping. *Journal of Machine Learning Research*, 20(74), 1–25.
- Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American Options by Simulation: A Simple Least-Squares Approach. *Review of Financial Studies*, 14(1), 113–147.
