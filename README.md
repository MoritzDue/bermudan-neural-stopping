# Neural optimal stopping vs Longstaff–Schwartz for Bermudan options

**Research question.** For a five-asset Bermudan max-call, how does neural optimal stopping (Becker, Cheridito & Jentzen, 2019) compare with Longstaff–Schwartz regression (LSM) as the training budget increases? How close are the resulting exercise policies to the optimal value?

**Key findings.**
- **50k training paths:** Neural − LSM is **+0.004**, with a 95% across-seed interval of [−0.035, +0.043], *conditional on the shared test paths*. No performance difference is detected; equivalence within ±0.05 is inconclusive once shared-test uncertainty is included.
- **1k training paths:** LSM performs better in **9 of 10 training seeds** (neural − LSM = −0.123), although the prespecified t-test does not reject at 5% (p = 0.053).
- **Distance to optimality:** For the selected 50k, seed-zero policies, estimated primal–dual gaps are **0.059 for LSM** and **0.046 for neural stopping**, approximately 0.2% of a price near 26.1. These are estimates, not guarantees of being within 0.05 of optimality.

![Neural minus LSM by training budget](results/headline/fig_neural_minus_lsm.png)
![Primal–dual bounds](results/headline/fig_primal_dual_bounds.png)

## Results

The **policy comparison** evaluates all fitted policies on one shared, independent **200,000-path test set**, with 10 training seeds per budget and 5,000 validation paths per seed. The **primal–dual experiment** is separate: it evaluates one fitted policy per method (50k, seed 0) using 2,000 outer paths and 3,200 nested inner simulations per outer state and exercise date. A 6,400-inner check uses 500 outer paths. No market data are used.

| Training paths | Neural − LSM (mean) | 95% CI across seeds¹ | Interpretation |
|---|---:|---:|---|
| 1k | −0.123 | [−0.248, +0.002] | LSM ahead in 9/10 seeds; prespecified t-test p = 0.053 |
| 10k | +0.017 | [−0.052, +0.086] | No detectable difference |
| 50k | +0.004 | [−0.035, +0.043] | No detectable difference; full ±0.05 equivalence not established |

¹ Training-seed intervals are conditional on the shared test set; they do not include its Monte Carlo uncertainty.

| Policy (50k, seed 0) | Policy-value estimate | Dual upper estimate | Estimated gap (95% CI) |
|---|---:|---:|---:|
| LSM | 26.097 | 26.156 | 0.059 ± 0.014 |
| Neural | 26.104 | 26.150 | 0.046 ± 0.008 |

These gap intervals describe outer-path sampling uncertainty; they do **not** remove finite-inner simulation bias. Neither gap is proven to be below 0.05. Becker et al. (2019) report a reference 95% interval of [26.138, 26.174] for this option; both estimated dual upper means lie within it.

## Method

- **Problem.** Max-call on 5 independent GBM assets: S₀ = K = 100, r = 5%, dividend yield 10%, σ = 20%, T = 3, nine exercise dates. Paths are simulated exactly; there is no market data.
- **Policies.**
  - LSM: polynomial regression with degree and ridge tuned on validation paths.
  - Neural: a network per exercise date, trained directly on the stopping payoff, with sorted inputs and a payoff feature.

  The **expected** discounted payoff under any fixed stopping policy is a lower bound on the optimal price; a finite Monte Carlo estimate can exceed the true price through sampling error.
- **Upper bounds.** The Andersen–Broadie (2004) primal–dual approach constructs a policy-based martingale using independent conditional continuations. The finite-inner dual estimate is an upper bound **in expectation**, not for each simulated sample or confidence endpoint. Inner simulation can introduce upward bias, so the estimated primal–dual gap is not a deterministic optimality guarantee.
- **Statistics.** Neural and LSM payoffs are compared on the same test paths within each seed (paired). Student-t intervals across training seeds are conditional on the shared test set. The primal–dual experiment uses paired estimators to reduce outer-path noise; the doubled inner budget checks sensitivity but cannot certify negligible inner bias.
- **Pre-registration.** Hypotheses, thresholds and the full run list were frozen in a manifest before any full result existed, sized from measured timings.
- **Validation.**
  - 1D Bermudan put against a refined binomial (CRR) reference.
  - Exact martingale and discounting checks.
  - Unit tests in `tests/`.

## Limitations

- At 50k, the across-seed interval lies inside the ±0.05 equivalence margin **conditional on the shared test set**. Shared-test error (estimated at roughly 0.013 for this comparison) does not shrink with more training seeds; accounting for it makes the equivalence conclusion inconclusive.
- The gap was measured for one policy per method (seed 0, 50k).
- The inner-doubling check has low power: its change intervals span roughly ±0.02, so the experiment does not rule out a remaining finite-inner bias of that order. The bias is not estimated precisely enough to claim a certified optimality gap.
- Not tested, because they did not fit the compute budget:
  - the payoff and sorting ablation (whether these input features help the network)
  - the 2D case
  - a neural tuning grid
  - development cost

  The neural settings were fixed, not optimised.
- Results hold for these implementations, budgets and one machine.

## Reading guide

Start with the research question and results **above**, then read the method and limitations. For the implementation, `stopping.py` contains the simulator, stopping policies and out-of-sample evaluation; `dual.py` contains nested primal–dual estimates. The validation checks live in `01_method_validation.ipynb` and `tests/`, while `03_main_study.ipynb` explains the design and interpretation. `02_calibration_and_preflight.ipynb` documents development procedures, not a completed full calibration experiment.

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

## Reproduce and verify

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover        # run the tests
.venv/bin/python make_figures.py             # redraw figures from committed CSVs
```

| Check | What it reproduces |
|---|---|
| Unit tests | Selected numerical and implementation checks under the pinned environment |
| `make_figures.py` | Headline charts from committed summaries; **no simulation or model fitting** |
| Notebooks on the current branch | Quick-mode smoke tests; their outputs are **not** the full-study findings |
| Frozen full study | Historical code and study plan at [`study-v1`](https://github.com/MoritzDue/bermudan-neural-stopping/tree/study-v1); full rerun is **not** a one-command operation from the current branch |

The full study originally took about 45 minutes of single-thread computation. The historical tag includes the frozen manifest and archival tools. Its README records a provenance-hash mismatch: the strict runner could not directly regenerate the saved summaries, although a separate archival procedure reportedly regenerated them byte-identically from hash-verified local checkpoints. The committed summaries and figures can be inspected without those checkpoints, but this does not amount to independently rerunning the original experiment. **Do not claim end-to-end reproducibility from the current branch.** Python source hashes include comments and docstrings, so documentation changes alter the fingerprint without changing the scientific calculations.

## References

- S. Becker, P. Cheridito, A. Jentzen (2019). *Deep Optimal Stopping.* JMLR 20(74).
- L. Andersen, M. Broadie (2004). *Primal-Dual Simulation Algorithm for Pricing Multidimensional American Options.* Management Science 50(9).
- F. Longstaff, E. Schwartz (2001). *Valuing American Options by Simulation.* Review of Financial Studies 14(1).

## Implementation provenance

I developed an initial simplified version based on *Deep Optimal Stopping*. AI coding agents (Claude and Codex) assisted with the extended implementation. I defined the research question and scope, made the study-design and compute-budget decisions, and reviewed the resulting work at each step.
