# Neural optimal stopping vs Longstaff–Schwartz for Bermudan options

**Question.** For a 5-asset Bermudan max-call, does a neural stopping policy (Becker, Cheridito & Jentzen, 2019) beat classical Longstaff–Schwartz regression (LSM)? And how close to optimal is either policy?

**Answer.**
- With plenty of data (50k training paths), neural stopping and LSM are **statistically indistinguishable**: the difference is +0.004, with 95% CI [−0.035, +0.043].
- With little data (1k paths), **LSM is ahead in 9 of 10 seeds** (−0.123), but the pre-registered t-test is borderline (p = 0.053).
- Andersen–Broadie upper bounds show that **both policies are within about 0.05 of the optimal value** (≈ 0.2% of a price of 26.1). Both bounds fall inside the published reference interval.

![Neural minus LSM by training budget](results/headline/fig_neural_minus_lsm.png)
![Primal–dual bounds](results/headline/fig_primal_dual_bounds.png)

## Results

All results are on a shared 200k-path test set. Training uses 10 independent seeds per budget, plus 5k validation paths per seed.

| Training paths | Neural − LSM (mean) | 95% CI (across seeds) | Reading |
|---|---|---|---|
| 1k | −0.123 | [−0.248, +0.002] | LSM better in 9/10 seeds; t-test p = 0.053 |
| 10k | +0.017 | [−0.052, +0.086] | no detectable difference |
| 50k | +0.004 | [−0.035, +0.043] | within ±0.05 (conditional on the test set) |

| Policy (50k, seed 0) | Lower bound | Upper bound | Duality gap (95% CI) |
|---|---|---|---|
| LSM | 26.097 | 26.156 | 0.059 ± 0.014 |
| Neural | 26.104 | 26.150 | 0.046 ± 0.008 |

Reference: Becker et al. (2019) report a 95% interval of [26.138, 26.174] for this option.

## Method

- **Problem.** Max-call on 5 independent GBM assets: S₀ = K = 100, r = 5%, dividend yield 10%, σ = 20%, T = 3, nine exercise dates. Paths are simulated exactly; there is no market data.
- **Policies.**
  - LSM: polynomial regression with degree and ridge tuned on validation paths.
  - Neural: a network per exercise date, trained directly on the stopping payoff, with sorted inputs and a payoff feature.

  Both are evaluated as stopping rules on the same test paths, so every price is a **lower bound**.
- **Upper bounds.** The Andersen–Broadie (2004) primal–dual method builds a martingale from nested inner simulations (2,000 outer × 3,200 inner paths). Its expected value bounds the true price from above, and the gap measures how much value a policy leaves on the table.
- **Statistics.** Neural and LSM are compared seed by seed (paired), with Student-t intervals across seeds. Upper and lower bounds are paired on the same outer paths, which makes the gap far more precise than either bound alone. A doubled inner budget (6,400) checked the bounds' stability.
- **Pre-registration.** Hypotheses, thresholds and the full run list were frozen in a manifest before any full result existed, sized from measured timings.
- **Validation.**
  - 1D Bermudan put against a refined binomial (CRR) reference.
  - Exact martingale and discounting checks.
  - Unit tests in `tests/`.

## Limitations

- The 50k equivalence uses the across-seed interval. Error from the shared test set (≈ 0.013) does not shrink with more seeds, and including it makes the ±0.05 equivalence inconclusive.
- The gap was measured for one policy per method (seed 0, 50k).
- The inner-doubling check has low power; inner-simulation bias of about ±0.02 cannot be excluded.
- Not tested, because they did not fit the compute budget:
  - the payoff and sorting ablation (whether these input features help the network)
  - the 2D case
  - a neural tuning grid
  - development cost

  The neural settings were fixed, not optimised.
- Results hold for these implementations, budgets and one machine.

## Repository

```
README.md
01_method_validation.ipynb      # simulator, CRR reference, 1D primal–dual check
02_calibration_and_preflight.ipynb
03_main_study.ipynb             # study design and conclusions (§4)
stopping.py                     # GBM simulation, LSM, neural stopping, evaluation
dual.py                         # Andersen–Broadie nested primal–dual estimator
validation_sections.py, dual_validation.py
make_figures.py                 # figures from results/headline (no simulation)
results/headline/               # full-study summaries (CSV/JSON) and figures
tests/
```

## Reproduce

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover        # tests, about 1 minute
.venv/bin/python make_figures.py             # redraw the figures from saved results
```

The notebooks run in quick mode, a few minutes of smoke tests whose numbers are not results. The full study took about 45 minutes of single-thread compute. Its exact code, frozen manifest and the script that regenerates every summary byte-for-byte are at git tag **`study-v1`**.

## References

- S. Becker, P. Cheridito, A. Jentzen (2019). *Deep Optimal Stopping.* JMLR 20(74).
- L. Andersen, M. Broadie (2004). *Primal-Dual Simulation Algorithm for Pricing Multidimensional American Options.* Management Science 50(9).
- F. Longstaff, E. Schwartz (2001). *Valuing American Options by Simulation.* Review of Financial Studies 14(1).

Implementation was done with AI coding agents (Claude and Codex). I defined the research question and scope, made the design and budget decisions, and reviewed the results at each step.
