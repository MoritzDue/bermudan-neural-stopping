# Neural optimal stopping vs Longstaff–Schwartz

**Quantitative finance research | Bermudan option pricing · Monte Carlo · optimal stopping · neural networks · primal–dual bounds**

Does a neural exercise policy outperform a well-tuned classical regression method when training data and compute are limited? This project compares **direct neural optimal stopping** with **Longstaff–Schwartz (LSM)** on a five-asset Bermudan max-call, then uses a nested primal–dual estimator to assess how much option value either policy may leave on the table.

The study uses **synthetic risk-neutral GBM paths**, not historical market data or a trading backtest. Its goal is to compare numerical pricing methods under a controlled simulation budget.

## Findings

- **With 50,000 training paths**, neural stopping and LSM have almost identical estimated values: neural − LSM = **+0.004**, with a 95% across-seed interval of **[−0.035, +0.043]**. No difference is detected; equivalence within ±0.05 is *not* established after allowing for shared-test uncertainty.
- **With 1,000 training paths**, LSM performs better in **9 of 10 seeds** (neural − LSM = **−0.123**). The prespecified t-test is borderline (**p = 0.053**), so this is not a conventional 5% rejection.
- **Both selected 50k policies appear close to optimal:** estimated primal–dual gaps are **0.059 for LSM** and **0.046 for neural stopping**—roughly 0.2% of an option price near 26.1. They are estimates with Monte Carlo uncertainty and possible finite-inner bias, **not certified 0.05-optimality guarantees**.

![Neural − LSM value across training budgets, with conditional 95% intervals](results/headline/fig_neural_minus_lsm.png)

### Main comparison

All methods use the same **200,000 held-out test paths**. For each training budget there are **10 independent training seeds** and **5,000 validation paths per seed**.

| Training paths per seed | Neural − LSM | 95% CI across seeds¹ | Interpretation |
|---:|---:|---:|---|
| 1,000 | −0.123 | [−0.248, +0.002] | LSM ahead in 9/10 seeds; t-test p = 0.053 |
| 10,000 | +0.017 | [−0.052, +0.086] | No detectable difference |
| 50,000 | +0.004 | [−0.035, +0.043] | No detectable difference; equivalence inconclusive |

¹ Student-t intervals quantify variation across training seeds **conditional on the shared test set**. Test-path Monte Carlo error does not vanish when the number of training seeds increases. See [Research notes](RESEARCH_NOTES.md#statistical-interpretation).

### Distance to optimality

A **separate** nested primal–dual evaluation uses the **seed-zero, 50k-path policies**, 2,000 outer simulations, and 3,200 independent inner continuations per outer state and exercise date.

| Policy | Estimated policy value | Dual upper estimate | Estimated gap (95% CI) |
|---|---:|---:|---:|
| LSM | 26.097 | 26.156 | 0.059 [0.044, 0.073] |
| Neural | 26.104 | 26.150 | 0.046 [0.038, 0.055] |

![Primal–dual estimates and published reference interval](results/headline/fig_primal_dual_bounds.png)

The reported gaps use a **paired, low-variance policy-value estimate** on the dual experiment's inner simulations, rather than the noisier realised outer-path payoff. Dual upper estimates bound the optimal price **in expectation**, not on every simulated path; the intervals reflect outer-path sampling uncertainty but do not remove finite-inner bias. The published five-asset benchmark is **26.159**, with a 95% interval of **[26.138, 26.174]** (Becker et al., 2019, Table 1).

## What this project demonstrates

- **Stochastic modelling:** exact-step, correlated-capable GBM simulation under a risk-neutral measure; independent random streams and controlled training/test splits.
- **Classical numerical finance:** backward Longstaff–Schwartz regression, polynomial state features, ridge selection and exercise-policy validation.
- **Machine learning for derivatives:** date-specific neural stopping decisions, a differentiable stopping objective, checkpoint selection and out-of-sample evaluation.
- **Pricing validation:** a one-dimensional Bermudan put against CRR and an independent convolution calculation; nested martingale-based upper estimates for the five-asset case.
- **Research discipline:** paired Monte Carlo comparisons, across-seed uncertainty, a frozen experiment plan, reproducible saved summaries and explicit disclosure of untested hypotheses.

## Model and experimental design

| Component | Specification |
|---|---|
| Contract | Bermudan max-call on **5 independent GBM assets**, \(S_0 = K = 100\) |
| Dynamics | Risk-free rate **5%**, dividend yield **10%**, volatility **20%**, maturity **3 years** |
| Exercise | **9 dates**, excluding time zero |
| LSM | Polynomial continuation regression; degree and ridge chosen on validation paths |
| Neural | One network per decision date; sorted log-price inputs plus payoff; **fixed** training settings |
| Evaluation | 1k / 10k / 50k training paths; 10 seeds; 5k validation paths per seed; one 200k-path test set |
| Optimality check | One selected policy per method (50k, seed 0); nested primal–dual estimate |

The **completed full study** covers the 5D comparison and the specified primal–dual pair. The proposed 2D study, representation ablation, neural hyperparameter grid and development-cost analysis were **not completed** under the compute budget. Results are specific to these implementations and model assumptions.

For the objective functions, stopping rules, uncertainty accounting and validity limits, read [**Research notes: methods and limitations**](RESEARCH_NOTES.md).

## Explore the repository

| Start here | What to inspect |
|---|---|
| [`stopping.py`](stopping.py) | GBM simulation, LSM and neural policy fitting, stopping evaluation and seed-level statistics |
| [`dual.py`](dual.py) | Independent inner rollouts and paired primal–dual estimators |
| [`01_method_validation.ipynb`](01_method_validation.ipynb) | Simulator, exercise/discounting checks and 1D Bermudan put validation |
| [`02_calibration_and_preflight.ipynb`](02_calibration_and_preflight.ipynb) | Development and preflight workflow (committed in **quick mode**) |
| [`03_main_study.ipynb`](03_main_study.ipynb) | Study rationale, analysis and documented full-study conclusions |
| [`tests/`](tests/) | Unit tests and numerical control cases |
| [`results/headline/`](results/headline/) | Saved summaries, per-unit records and the figures above |
| [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) | Technical methods, interpretation and reproducibility boundaries |

## Run checks and reproduce figures

The recorded full-study environment used **Python 3.14.4** and the versions pinned in [`requirements.txt`](requirements.txt). The commands below use a Unix-like shell:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
python make_figures.py
```

| Action | What it establishes |
|---|---|
| Unit tests | Selected implementation, numerical and statistical checks |
| `make_figures.py` | Regenerates charts **from committed CSVs**, without simulation |
| Current notebooks | **Quick-mode smoke tests**; their outputs are not the published full-study measurements |
| Historical [`study-v1`](https://github.com/MoritzDue/bermudan-neural-stopping/tree/study-v1) tag | Frozen code, plan and archival tooling for the original study—not a one-command rerun from current `main` |

**Reproducibility qualification.** The historical tag discloses an execution-provenance hash mismatch: its strict runner did not directly regenerate the saved summaries, although a separate archival procedure reportedly reproduced them byte-identically from hash-verified local checkpoints. The checkpoints are not all committed. Accordingly, chart regeneration and the provided CSV/JSON records are **not equivalent to independently rerunning the full study**. Details are in [Research notes](RESEARCH_NOTES.md#reproducibility-and-scope).

## References

- Becker, S., Cheridito, P., & Jentzen, A. (2019). [*Deep Optimal Stopping*](https://jmlr.org/papers/v20/18-232.html). *Journal of Machine Learning Research*, 20(74), 1–25. Five-asset benchmark: Table 1.
- Andersen, L., & Broadie, M. (2004). [*Primal-Dual Simulation Algorithm for Pricing Multidimensional American Options*](https://doi.org/10.1287/mnsc.1040.0258). *Management Science*, 50(9).
- Longstaff, F. A., & Schwartz, E. S. (2001). *Valuing American Options by Simulation: A Simple Least-Squares Approach*. *Review of Financial Studies*, 14(1).

## Implementation provenance

I built an initial simplified version based on *Deep Optimal Stopping*. Claude and Codex assisted with extending the implementation. I defined the research question and scope, made the study-design and compute-budget decisions, and reviewed the results at each step.
