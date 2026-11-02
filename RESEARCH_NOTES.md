# Research notes — methodological and reproducibility details

These notes supplement the [README](README.md) with the technical definitions and limits most relevant to reviewing the study. They describe the **completed** frozen five-dimensional experiment, not every exploratory option left in the notebooks.

## Contract, simulation and stopping

The main experiment prices a Bermudan call on the maximum of five independent assets under risk-neutral geometric Brownian motion:

```text
dS_i(t) / S_i(t) = (r - q) dt + sigma dW_i(t)
g(S) = max(max_i S_i - K, 0)
t_k = kT/N, k=1,...,N
```

The parameters are S0 = K = 100, r = 0.05, q = 0.10, sigma = 0.20, T = 3, and N = 9. The experiment does not allow exercise at t=0. GBM increments are simulated by the exact discrete transition, not Euler discretisation. All quoted results are **model-based**, not empirical market prices or backtested returns.

A stopping policy exercises at an eligible date based only on its current state. Its true expected discounted payoff is a **lower bound** on the optimal stopping value; a finite Monte Carlo estimate need not numerically lie below the unknown optimal value.

## Policies and selection

**Longstaff–Schwartz (LSM).** For each exercise date, moving backwards, fit ridge regression of realised later-policy continuation payoffs on polynomial features of in-the-money states. Stop when the immediate payoff exceeds the estimated continuation value. Choose polynomial degree and ridge strength on validation paths.

**Direct neural stopping.** Fit one network per exercise date by maximising a differentiable soft stopping reward, then deploy a **hard** exercise decision. The main network receives sorted, scaled log-prices and a payoff feature, exploiting symmetry of the max-call; the payoff is deterministic from the state, not extra market information. The completed study uses **fixed settings** (1,500 Adam steps per exercise date, learning rate 0.001, batch size 512). No completed full neural tuning grid or representation ablation supports claims of optimal hyperparameters or feature superiority.

**Validation guard.** Candidate exercise rules can be disabled on dates where their validation-estimated improvement over continuation is negative. Neural checkpoint selection also uses these validation paths, so validation gains must not be treated as unbiased performance estimates. The training/validation-selected rules are evaluated on held-out simulation paths.

Per budget, ten training/validation seeds are compared using the **same** 200,000-path test set. Within each seed, LSM and neural payoffs are paired path-by-path to reduce variance of their difference. Across-seed Student-t intervals are explicitly **conditional on that shared test sample**; the test-set sampling error is correlated between seeds and does not vanish as the seed count increases.

## Primal–dual estimates

The nested dual method constructs a martingale using independent conditional simulations of the fixed policy's future discounted cash flows. In ideal form it gives an upper bound on the optimal stopping value via the expectation of the maximum discounted payoff after subtracting the martingale. In the finite-inner implementation the **expected** estimated dual maximum remains an upper bound, but an individual realised maximum or confidence endpoint does not have that guarantee.

For a selected seed-zero 50k policy per method, 2,000 outer paths and 3,200 inner rollouts per state/date are used. A coupled 6,400-inner follow-up uses 500 outer paths. The reported gap uses a paired estimator based on a simulated initial policy value and the dual maximum; this is considerably less noisy than using the realised stopping payoff on each outer path. Paired confidence intervals describe outer simulation uncertainty, **not** the finite-inner upward bias. The small observed doubling changes are a stability diagnostic and cannot rule out residual bias around the observed sensitivity scale.

| Policy | Upper estimate | Initial policy-value estimate | Estimated gap (95% CI) |
|---|---:|---:|---:|
| LSM | 26.1559 | 26.0974 | 0.0586 [0.0443, 0.0728] |
| Neural | 26.1502 | 26.1039 | 0.0463 [0.0380, 0.0546] |

The paper's five-asset reference is 26.159 with an interval [26.138, 26.174] (Becker et al., 2019, Table 1). Agreement with that reference is **supporting consistency evidence**, not a mathematical proof of correctness or a claim that the estimated gaps are guaranteed below 0.05.

## Statistical interpretation

- **50k paths:** neural − LSM = 0.0036, conditional 95% across-seed CI [−0.0353, 0.0425]. This does not detect a difference. Although that conditional interval lies inside ±0.05, equivalence **including shared-test uncertainty** is inconclusive.
- **1k paths:** neural − LSM = −0.1233, conditional 95% across-seed CI [−0.2482, 0.0017]. LSM is ahead in nine of ten seeds, but the pre-specified t-test gives p = 0.053, so there is no 5% rejection under that test.
- **10k paths:** neural − LSM = 0.0172, conditional 95% CI [−0.0520, 0.0864]. This comparison is descriptive.
- **Dual gaps:** the separate dual experiment quantifies one trained policy per method, not the optimality of all ten training seeds.

The estimated performance differences are option-price units (with strike 100), not percentage returns. Uncertainty from the published benchmark, training-seed variation, shared test paths and the nested upper estimator are conceptually distinct.

## Independent checks and limitations

The available checks include exact-transition GBM moment/correlation diagnostics, cash-flow discounting and exercise-timing controls, a one-dimensional Bermudan put compared with a refined CRR tree and an independently computed convolution reference, deterministic controls, dual-estimator diagnostics and unit tests. They provide useful consistency checks; they are **not** comprehensive mathematical certification.

The full-study compute budget did not accommodate the planned two-dimensional comparison, four-variant representation ablation, exhaustive neural hyperparameter calibration, ten-dimensional extension or a full development-cost comparison. These experiments should not be represented as completed findings.

Timing figures describe the recorded single-thread CPU environment and these implementations. The neural fixed-setting training cost and LSM per-seed selection cost reflect different fitting/selection procedures; timing alone does not establish a general speed ranking for all architectures or implementations.

## Reproducibility and scope

The **current `main` branch** is intended for reading, quick-mode notebook checks, tests, figures and saved data. The full research run used a separate historical frozen workflow at Git tag [`study-v1`](https://github.com/MoritzDue/bermudan-neural-stopping/tree/study-v1). That recorded environment was **Python 3.14.4**, with the package versions in [`requirements.txt`](requirements.txt).

The committed [results index](results/README.md) describes the publication layout. `results/headline/` keeps the two PNG figures, `H2.csv`, `H3.csv`, `summary_main.csv` and `summary.json`; `results/supporting/` contains `runs.csv`, `paired.csv`, `dates.csv` and `units.jsonl`, preserved byte-for-byte from the original research summaries. `results/SHA256SUMS` records the original checksums for all eight data files at their new paths. `make_figures.py` regenerates the PNGs without model fitting.

The historical `study-v1` README discloses an **execution-provenance hash mismatch**: its strict runner refused the archived execution hash, even though the frozen science-module hashes matched. A separate archival procedure was reported to reproduce the eight saved summaries byte-for-byte from hash-verified local checkpoints. Those raw checkpoints are **not all distributed** on current `main`. This limits independent end-to-end replication; it does not by itself establish a pricing error.

Also, `stopping.py` hashes Python source files byte-for-byte, so source documentation edits alter the fingerprint and may cause archival provenance checks to reject otherwise equivalent code. To inspect the historical computation, refer to the **original tagged revision** rather than substituting current documented modules.

## References

1. Becker, S., Cheridito, P., & Jentzen, A. (2019). [*Deep Optimal Stopping*](https://jmlr.org/papers/v20/18-232.html). *Journal of Machine Learning Research*, 20(74). Table 1 contains the five-dimensional option benchmark.
2. Andersen, L., & Broadie, M. (2004). [*Primal-Dual Simulation Algorithm for Pricing Multidimensional American Options*](https://doi.org/10.1287/mnsc.1040.0258). *Management Science*, 50(9).
3. Longstaff, F. A., & Schwartz, E. S. (2001). *Valuing American Options by Simulation: A Simple Least-Squares Approach*. *Review of Financial Studies*, 14(1).
