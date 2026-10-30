"""
stopping.py - all shared code for the project, in one file. Read it top to bottom.

The three notebooks import this file, so every method (LSM, neural stopping,
evaluation, statistics) exists exactly once and all experiments use identical code.

Sections
  0. CONFIG            - the frozen experimental specification (quick and full modes)
  1. Thread control     - single-threaded NumPy *and* PyTorch, for fair CPU timing
  2. Contracts          - Bermudan put / max-call, exact GBM simulation, references
  3. Policies           - stopping policies, the never-exercise guard, forward evaluation
  4. Longstaff-Schwartz - regression on in-the-money paths, tuned on validation paths
  5. Neural stopping    - Becker, Cheridito & Jentzen (2019) soft-stopping objective
  6. Experiment units   - fit all methods for one (case, budget, seed); paired statistics
  7. Results storage    - incremental CSVs, resumable runs, frozen-config check

Conventions
  * Exercise dates t_k = k*T/N, k = 1..N (no exercise at t=0; at the money this is
    never optimal, see footnote 17 in the paper, so values match Table 1).
  * Policy values are averages of discounted realised payoffs on independent paths,
    so in population every reported value is a LOWER bound for the true price.
  * gap = reference - policy value (positive = shortfall).
"""
from __future__ import annotations

import json
import hashlib
import os
import math
import time
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
from torch import nn

# =============================================================================
# 0. CONFIG - the frozen specification agreed before any full run.
#    Change it only BEFORE running the full study; notebook 3 refuses to resume
#    a full run whose configuration has changed.
# =============================================================================
CONFIG = {
    "quick": {   # smoke test: finishes in a few minutes, numbers are NOT results
        "base_seed": 2026,
        "validation": {          # notebook 1
            "put_train": 2_000, "put_seeds": 1, "val": 1_000, "test": 20_000,
            "lsm_budgets": [500, 2_000, 10_000],
            "nn_settings": dict(max_steps=150, lr=1e-3, batch=256),
            "paper_check": dict(max_steps=60, batch=1_024),
        },
        "calibration": {         # notebook 2
            "budgets": [500, 2_000], "dev_seeds": [100, 101],
            "variants": ["sorted+payoff", "raw+payoff"],   # used by the selection rule
            "diagnostic_variants": ["raw"],                 # saturation evidence only
            "max_steps": [50, 150], "lrs": [1e-3, 3e-3], "batch": 256,
            "val": 1_000, "selection_set": 20_000, "tolerance": 0.05,
        },
        "study": {               # notebook 3
            "budgets": [500, 2_000], "seeds": list(range(2)),
            "ablation_budgets": [500, 2_000],
            "val": 1_000, "test": 20_000,
            "case_2d": {"budget": 2_000, "seeds": list(range(2))},
            "case_10d": {"budget": 2_000, "seeds": list(range(1))},
        },
    },
    "full": {    # the research run
        "base_seed": 2026,
        "validation": {
            "put_train": 10_000, "put_seeds": 3, "val": 5_000, "test": 200_000,
            "lsm_budgets": [1_000, 10_000, 50_000, 100_000],
            "nn_settings": dict(max_steps=1_500, lr=1e-3, batch=512),
            "paper_check": dict(max_steps=3_000, batch=8_192),   # paper's training regime
        },
        "calibration": {
            "budgets": [1_000, 50_000], "dev_seeds": [100, 101, 102],
            "variants": ["sorted+payoff", "raw+payoff"],   # used by the selection rule
            "diagnostic_variants": ["raw"],                 # saturation evidence only
            "max_steps": [750, 1_500, 3_000], "lrs": [1e-3, 3e-3], "batch": 512,
            "val": 5_000, "selection_set": 100_000, "tolerance": 0.02,
        },
        "study": {
            "budgets": [1_000, 10_000, 50_000], "seeds": list(range(10)),
            "ablation_budgets": [1_000, 50_000],
            "val": 5_000, "test": 200_000,
            "case_2d": {"budget": 10_000, "seeds": list(range(10))},
            "case_10d": {"budget": 10_000, "seeds": list(range(5))},   # optional
        },
    },
}

# LSM candidates, chosen per seed on validation paths (cheap, so tuned every time).
LSM_GRID = {"degrees": (1, 2, 3, 4), "ridges": (1e-6, 1e-4, 1e-2)}

# The 2x2 representation ablation: name -> (sorted_inputs, payoff_input)
VARIANTS = {
    "sorted+payoff": (True, True),    # main neural method
    "raw+payoff": (False, True),      # the paper's input representation
    "sorted": (True, False),
    "raw": (False, False),
}
MAIN_VARIANT = "sorted+payoff"


PROJECT_ROOT = Path(__file__).resolve().parent


def results_dir(mode: str) -> Path:
    """Project-anchored outputs; notebook tools may override the results root.

    BERMUDAN_RESULTS_ROOT names the directory containing quick/full, not the mode.
    """
    if mode not in CONFIG:
        raise ValueError("mode must be quick or full")
    path = Path(os.environ.get("BERMUDAN_RESULTS_ROOT", PROJECT_ROOT / "results")) / mode
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# 1. Thread control
# =============================================================================
_THREAD_LIMITER = None


def limit_threads(n: int = 1) -> dict:
    """Limit BOTH the BLAS library behind NumPy (used by LSM's least squares) and
    PyTorch to n threads. torch.set_num_threads alone would leave LSM multi-threaded
    and bias the timing comparison in LSM's favour."""
    global _THREAD_LIMITER
    from threadpoolctl import threadpool_limits
    _THREAD_LIMITER = threadpool_limits(limits=n)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:          # can only be set once per process; harmless
        pass
    return thread_report()


def thread_report() -> dict:
    from threadpoolctl import threadpool_info
    return {"torch_threads": torch.get_num_threads(),
            "blas": [f"{i.get('internal_api')}:{i.get('num_threads')}" for i in threadpool_info()]}


# =============================================================================
# 2. Contracts, simulation, references
# =============================================================================
@dataclass(frozen=True)
class Option:
    kind: str = "put"            # 'put' (d=1) or 'max_call'
    d: int = 1
    s0: float = 100.0
    strike: float = 100.0
    rate: float = 0.05
    dividend: float = 0.0
    sigma: float = 0.20
    rho: float = 0.0
    maturity: float = 1.0
    dates: int = 9               # exercise at t_k = k*T/dates, k = 1..dates

    def __post_init__(self):
        if self.kind not in ("put", "max_call"):
            raise ValueError("kind must be 'put' or 'max_call'")
        if self.kind == "put" and self.d != 1:
            raise ValueError("the put is one-dimensional")
        if self.d > 1 and not (-1 / (self.d - 1) < self.rho < 1):
            raise ValueError("equicorrelation matrix must be positive definite")

    @property
    def name(self) -> str:
        return "put_1d" if self.kind == "put" else f"max_call_{self.d}d"

    @property
    def dt(self) -> float:
        return self.maturity / self.dates

    def payoff(self, x):
        """Undiscounted exercise value g(x); x has shape (..., d)."""
        if self.kind == "put":
            return np.maximum(self.strike - x[..., 0], 0.0)
        return np.maximum(x.max(axis=-1) - self.strike, 0.0)

    def discount(self, k):
        return np.exp(-self.rate * np.asarray(k) * self.dt)


def paper_max_call(d: int) -> Option:
    """Symmetric max-call of Becker et al. (2019), Table 1, with s0 = 100."""
    return Option(kind="max_call", d=d, s0=100, strike=100, rate=0.05, dividend=0.10,
                  sigma=0.20, rho=0.0, maturity=3.0, dates=9)


def validation_put() -> Option:
    return Option(kind="put", d=1, s0=100, strike=100, rate=0.05, dividend=0.0,
                  sigma=0.20, maturity=1.0, dates=9)


ROLES = {"train": 1, "val": 2, "test": 3, "holdout": 4, "init": 5, "fresh": 6, "check": 7, "dual_outer": 8, "dual_inner": 9}


def seed_for(base: int, role: str, d: int, seed: int = 0, extra: int = 0) -> int:
    """Independent, reproducible random streams for every role (train/val/test/...)."""
    return int(np.random.SeedSequence([base, ROLES[role], d, seed, extra]).generate_state(1)[0])


def gbm_paths(opt: Option, n: int, seed=None, rng=None) -> np.ndarray:
    """Exact joint GBM transitions, array of shape (n, dates+1, d).
    With a fixed seed, the first m rows of gbm_paths(n) equal gbm_paths(m) for m < n,
    so training sets of different budgets within one seed are nested."""
    rng = np.random.default_rng(seed) if rng is None else rng
    corr = np.full((opt.d, opt.d), opt.rho)
    np.fill_diagonal(corr, 1.0)
    z = rng.standard_normal((n, opt.dates, opt.d)) @ np.linalg.cholesky(corr).T
    steps = (opt.rate - opt.dividend - 0.5 * opt.sigma ** 2) * opt.dt + opt.sigma * math.sqrt(opt.dt) * z
    paths = np.empty((n, opt.dates + 1, opt.d))
    paths[:, 0, :] = opt.s0
    paths[:, 1:, :] = opt.s0 * np.exp(np.cumsum(steps, axis=1))
    return paths


def crr_bermudan_put(opt: Option, steps_per_date: int = 160) -> float:
    """Cox-Ross-Rubinstein tree; every exercise date lies on the tree grid."""
    steps = steps_per_date * opt.dates
    dt = opt.maturity / steps
    up = math.exp(opt.sigma * math.sqrt(dt))
    down = 1 / up
    p = (math.exp((opt.rate - opt.dividend) * dt) - down) / (up - down)
    disc = math.exp(-opt.rate * dt)
    j = np.arange(steps + 1)
    v = np.maximum(opt.strike - opt.s0 * up ** j * down ** (steps - j), 0.0)
    for k in range(steps - 1, -1, -1):
        v = disc * ((1 - p) * v[:-1] + p * v[1:])
        if k > 0 and k % steps_per_date == 0:
            j = np.arange(k + 1)
            v = np.maximum(v, opt.strike - opt.s0 * up ** j * down ** (k - j))
    return float(v[0])


def convolution_bermudan_put(opt: Option, spacing: float = 0.0005) -> float:
    """Independent log-grid backward induction using the exact Gaussian kernel.

    Trapezoid convolution on log(S/S0) in [-3,3]. Boundary truncation is
    negligible for validation_put; this is a numerical cross-check, not a bound.
    Exercise at dates 1..N only. Grid refinement must be checked separately.
    """
    from scipy.signal import fftconvolve
    if opt.kind != "put" or opt.sigma <= 0:
        raise ValueError("requires a positive-volatility 1D put")
    n = int(math.ceil(3 / spacing))
    x = np.arange(-n, n + 1) * spacing
    sd = opt.sigma * math.sqrt(opt.dt)
    drift = (opt.rate - opt.dividend - opt.sigma ** 2 / 2) * opt.dt
    # Convolution reverses the increment: kernel(t) = density(-t).
    kernel = np.exp(-0.5 * ((x + drift) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)) * spacing
    g = np.maximum(opt.strike - opt.s0 * np.exp(x), 0)
    v = g.copy()
    for k in range(opt.dates - 1, -1, -1):
        v = math.exp(-opt.rate * opt.dt) * fftconvolve(v, kernel, mode="same")
        if k > 0:
            v = np.maximum(v, g)
    return float(v[n])


# BCJ Table 1, transcribed from the Director brief; primary PDF unavailable offline.
# value/low/high remain compatibility aliases for the midpoint and DOS CI.
PAPER_REFERENCES = {
    2: dict(value=13.899, lower=13.895, upper=13.903, midpoint=13.899,
            low=13.880, high=13.910, ci=[13.880, 13.910], lattice=13.902,
            source="BCJ Table 1, d=2; brief transcription, primary source unverified offline"),
    5: dict(value=26.159, lower=26.156, upper=26.162, midpoint=26.159,
            low=26.138, high=26.174, ci=[26.138, 26.174], lattice=None,
            source="BCJ Table 1, d=5; brief transcription, primary source unverified offline"),
    10: dict(value=38.337, low=38.300, high=38.367, verified=False,
             source="Unverified legacy 10D fields; out of scope, not a research benchmark"),
}


def reference_for(opt: Option) -> dict:
    """Reference uncertainty is separate from policy Monte Carlo uncertainty."""
    if opt.kind == "put":
        fine = crr_bermudan_put(opt, 2560)
        convolution = convolution_bermudan_put(opt)
        coarse = convolution_bermudan_put(opt, 0.001)
        err = abs(fine - convolution) + abs(convolution - coarse)
        return dict(value=fine, low=fine - err, high=fine + err,
                    convolution=convolution, grid_band=err,
                    source="CRR 2560 steps/date; heuristic grid band = CRR/convolution discrepancy + convolution refinement; not certified")
    if opt != paper_max_call(opt.d) or opt.d not in PAPER_REFERENCES or opt.d == 10:
        raise ValueError("Verified-scope references require exact Table 1 2D/5D parameters.")
    return dict(PAPER_REFERENCES[opt.d])


# =============================================================================
# 3. Policies, the never-exercise guard, forward evaluation
# =============================================================================
@dataclass
class Policy:
    """A Bermudan stopping policy: one exercise rule per date k = 1..N-1.
    rules[k](x) -> bool array, called only on in-the-money states.
    guard[k] = True means the candidate rule at date k did worse than 'never exercise
    at k' on validation paths, so the GUARDED policy never exercises at k.
    The UNGUARDED policy always uses rules[k]. Training targets at earlier dates were
    built with the guarded later policy, so 'unguarded' is an execution-level ablation
    of the same fitted models, not a separately optimised procedure."""
    method: str
    rules: dict = field(default_factory=dict)
    guard: dict = field(default_factory=dict)
    val_gain: dict = field(default_factory=dict)   # validation gain of rules[k], price units at t=0
    diagnostics: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def stops(self, k: int, x_itm: np.ndarray, guarded: bool = True) -> np.ndarray:
        rule = self.rules.get(k)
        if rule is None or (guarded and self.guard.get(k, False)):
            return np.zeros(len(x_itm), dtype=bool)
        return rule(x_itm)

    def n_guarded(self) -> int:
        return int(sum(bool(v) for v in self.guard.values()))


def exercise_mask(opt: Option, policy: Policy, k: int, x: np.ndarray, guarded: bool = True):
    """Full-length boolean mask: in the money AND the policy says stop."""
    g = opt.payoff(x)
    mask = np.zeros(len(x), dtype=bool)
    itm = np.flatnonzero(g > 0)
    if itm.size:
        mask[itm] = policy.stops(k, x[itm], guarded)
    return mask


def cashflows(opt: Option, paths, policy: Policy, guarded=True, k_from=1):
    """Follow the policy from date k_from; return (payoff at stopping, stopping date)."""
    n = len(paths)
    alive = np.ones(n, dtype=bool)
    pay = np.zeros(n)
    tau = np.full(n, opt.dates)
    for k in range(k_from, opt.dates):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        x = paths[idx, k, :]
        g = opt.payoff(x)
        itm = g > 0
        if not itm.any():
            continue
        stop = policy.stops(k, x[itm], guarded)
        hit = idx[itm][stop]
        pay[hit] = g[itm][stop]
        tau[hit] = k
        alive[hit] = False
    last = np.flatnonzero(alive)
    pay[last] = opt.payoff(paths[last, -1, :])
    return pay, tau


def discounted_payoffs(opt: Option, paths, policy: Policy, guarded=True, chunk=50_000):
    """Per-path discounted payoff of the policy (evaluated in chunks to bound memory)."""
    values = np.empty(len(paths))
    taus = np.empty(len(paths), dtype=int)
    for a in range(0, len(paths), chunk):
        pay, tau = cashflows(opt, paths[a:a + chunk], policy, guarded)
        values[a:a + chunk] = pay * opt.discount(tau)
        taus[a:a + chunk] = tau
    return values, taus


class Backward:
    """Realised cash flows under the already-fitted LATER policy (backward induction).
    future(k) is the cash flow discounted to t_k of continuing at date k."""

    def __init__(self, opt: Option, paths):
        self.opt, self.paths = opt, paths
        self.pay = opt.payoff(paths[:, -1, :]).copy()
        self.tau = np.full(len(paths), opt.dates)

    def future(self, k):
        return self.pay * np.exp(-self.opt.rate * (self.tau - k) * self.opt.dt)

    def update(self, k, stop_mask):
        g = self.opt.payoff(self.paths[:, k, :])
        self.pay[stop_mask] = g[stop_mask]
        self.tau[stop_mask] = k


def validation_gain(opt: Option, k: int, stop_on_itm, g_itm, future_itm, n_total) -> float:
    """Estimated value added at t=0 by the candidate rule at date k versus never
    exercising at k (both followed by the same later policy), on validation paths."""
    return float(np.sum(stop_on_itm * (g_itm - future_itm)) / n_total * opt.discount(k))


def guard_and_update(opt, policy, k, val_bw: Backward, train_bw: Backward | None):
    """Pre-specified guard rule: fall back to 'never exercise at k' iff the candidate's
    validation gain is negative (a rule that exercises on no validation path has gain
    exactly 0 and behaves like the fallback anyway, so it is not counted as guarded).
    Then update the later-policy cash flows using the guarded decision."""
    xv = val_bw.paths[:, k, :]
    gv = opt.payoff(xv)
    itm = gv > 0
    if policy.rules.get(k) is None or not itm.any():
        gain = 0.0
    else:
        stop = policy.rules[k](xv[itm])
        gain = validation_gain(opt, k, stop, gv[itm], val_bw.future(k)[itm], len(xv))
    policy.val_gain[k] = gain
    policy.guard[k] = gain < 0.0
    val_bw.update(k, exercise_mask(opt, policy, k, xv, guarded=True))
    if train_bw is not None:
        train_bw.update(k, exercise_mask(opt, policy, k, train_bw.paths[:, k, :], guarded=True))
    return gain


# =============================================================================
# 4. Longstaff-Schwartz
# =============================================================================
def lsm_features(x: np.ndarray, opt: Option) -> np.ndarray:
    """Permutation-invariant state summary plus the payoff (same information the
    payoff-aware networks receive)."""
    z = np.log(x / opt.strike)
    g = opt.payoff(x) / opt.strike
    if opt.d == 1:
        cols = [z[:, 0]]
    else:
        zs = np.sort(z, axis=1)
        cols = [zs[:, -1], zs[:, 0]] if opt.d == 2 else \
               [zs[:, -1], zs[:, -2], z.mean(axis=1), z.std(axis=1), zs[:, 0]]
    return np.column_stack(cols + [g])


def polynomial(features: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.prod(features[:, c], axis=1)
            for deg in range(1, degree + 1)
            for c in combinations_with_replacement(range(features.shape[1]), deg)]
    return np.column_stack(cols)


class RidgeRegression:
    """Ridge least squares on standardised polynomial features (intercept unpenalised)."""

    def __init__(self, design, y, ridge):
        self.mu = design.mean(axis=0)
        self.sd = np.maximum(design.std(axis=0), 1e-8)
        z = np.column_stack([np.ones(len(design)), (design - self.mu) / self.sd])
        penalty = ridge * len(design) * np.eye(z.shape[1])
        penalty[0, 0] = 0.0
        self.coef = np.linalg.solve(z.T @ z + penalty, z.T @ y)

    def predict(self, design):
        return np.column_stack([np.ones(len(design)), (design - self.mu) / self.sd]) @ self.coef


def fit_lsm(opt: Option, train, val, degree: int, ridge: float) -> Policy:
    """Backward LSM: regress realised continuation cash flows on in-the-money paths;
    exercise iff payoff >= estimated continuation value. Guard per date on validation."""
    policy = Policy(method="lsm", info=dict(degree=degree, ridge=ridge))
    tr, va = Backward(opt, train), Backward(opt, val)
    n_cols = polynomial(lsm_features(train[:2, 1, :], opt), degree).shape[1]
    for k in range(opt.dates - 1, 0, -1):
        x = train[:, k, :]
        g = opt.payoff(x)
        itm = g > 0
        if itm.sum() >= 2 * n_cols + 20:
            reg = RidgeRegression(polynomial(lsm_features(x[itm], opt), degree), tr.future(k)[itm], ridge)
            policy.rules[k] = (lambda z, reg=reg: opt.payoff(z) >= reg.predict(polynomial(lsm_features(z, opt), degree)))
        else:
            policy.rules[k] = None          # too few samples: never exercise
        gain = guard_and_update(opt, policy, k, va, tr)
        policy.diagnostics.append(dict(date=k, n_itm=int(itm.sum()), val_gain=gain, guard=policy.guard[k]))
    return policy


def select_lsm(opt: Option, train, val, grid=LSM_GRID):
    """Tune degree and ridge on VALIDATION paths (guarded policy value).
    Returns the chosen policy, a candidate table, and timing:
    train_s = fit time of the chosen candidate, selection_s = all candidates + validation."""
    start = time.perf_counter()
    rows, best = [], None
    for degree in grid["degrees"]:
        for ridge in grid["ridges"]:
            t0 = time.perf_counter()
            pol = fit_lsm(opt, train, val, degree, ridge)
            fit_s = time.perf_counter() - t0
            value = float(discounted_payoffs(opt, val, pol)[0].mean())
            rows.append(dict(degree=degree, ridge=ridge, val_value=value, fit_s=fit_s))
            if best is None or value > best[0]:
                best = (value, pol, fit_s)
    best[1].info.update(train_s=best[2], selection_s=time.perf_counter() - start,
                        val_value=best[0])
    return best[1], pd.DataFrame(rows)


# =============================================================================
# 5. Direct neural stopping (Becker, Cheridito & Jentzen, 2019)
# =============================================================================
@dataclass(frozen=True)
class NeuralSettings:
    sorted_inputs: bool = True
    payoff_input: bool = True
    max_steps: int = 1_500        # Adam steps per exercise date (fixed, not epochs)
    lr: float = 1e-3
    batch: int = 512
    width_extra: int = 40         # hidden width d + 40, two hidden layers (paper)
    batch_norm: bool = True       # paper uses batch normalisation
    eval_every: int = 0           # 0 -> max(25, max_steps // 20) validation checkpoints
    fresh_paths: bool = False     # paper regime: a new simulated batch at every step

    @property
    def variant(self) -> str:
        return {v: k for k, v in VARIANTS.items()}[(self.sorted_inputs, self.payoff_input)]

    def checkpoint_every(self) -> int:
        return self.eval_every or max(25, self.max_steps // 20)


def with_variant(settings: NeuralSettings, variant: str) -> NeuralSettings:
    sorted_inputs, payoff_input = VARIANTS[variant]
    return replace(settings, sorted_inputs=sorted_inputs, payoff_input=payoff_input)


def nn_features(x: np.ndarray, opt: Option, s: NeuralSettings) -> np.ndarray:
    z = np.log(x / opt.strike)
    if s.sorted_inputs and opt.d > 1:
        z = np.sort(z, axis=1)                  # permutation-invariant representation
    if s.payoff_input:
        z = np.column_stack([z, opt.payoff(x) / opt.strike])   # paper: payoff as extra input
    return z.astype(np.float32)


class StopNet(nn.Module):
    """Outputs a logit; exercise probability = sigmoid(logit)."""

    def __init__(self, n_in: int, width: int, batch_norm: bool):
        super().__init__()
        layers, prev = [], n_in
        for _ in range(2):
            layers += [nn.Linear(prev, width)] + ([nn.BatchNorm1d(width)] if batch_norm else []) + [nn.ReLU()]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NeuralRule:
    """Frozen hard exercise rule: stop iff logit >= 0 (i.e. probability >= 1/2)."""

    def __init__(self, net, mu, sd, opt, settings):
        self.net, self.mu, self.sd, self.opt, self.s = net.eval(), mu, sd, opt, settings

    def __call__(self, x, chunk=100_000):
        out = np.empty(len(x), dtype=bool)
        with torch.no_grad():
            for a in range(0, len(x), chunk):
                f = (nn_features(x[a:a + chunk], self.opt, self.s) - self.mu) / self.sd
                out[a:a + chunk] = self.net(torch.from_numpy(f)).numpy() >= 0.0
        return out


def _train_one_date(opt, k, s, sampler, Xv, Av, n_val, mu, sd, a_scale, seed):
    """Maximise the soft stopping reward mean(p * (g - future)) with Adam for a fixed
    number of steps. Checkpoints are scored by the HARD validation gain; the best
    checkpoint is kept. Returns (net, best_gain, best_step, curve)."""
    torch.manual_seed(seed)
    net = StopNet(Xv.shape[1], opt.d + s.width_extra, s.batch_norm)
    adam = torch.optim.Adam(net.parameters(), lr=s.lr)
    Xv_t = torch.from_numpy((Xv - mu) / sd)
    best_gain, best_step, best_state, curve = -np.inf, 0, None, []
    every = s.checkpoint_every()
    for step in range(1, s.max_steps + 1):
        X, A = sampler(step)
        if len(X) < 2:
            continue
        net.train()
        p = torch.sigmoid(net(torch.from_numpy((X - mu) / sd)))
        loss = -(p * torch.from_numpy(A / a_scale)).mean()      # gradient ASCENT on reward
        adam.zero_grad()
        loss.backward()
        adam.step()
        if step % every == 0 or step == s.max_steps:
            net.eval()
            with torch.no_grad():
                stop = (net(Xv_t) >= 0).numpy() if len(Xv) else np.zeros(0, bool)
            gain = float(np.sum(stop * Av) / n_val * opt.discount(k))
            curve.append((step, gain))
            if gain > best_gain:
                best_gain, best_step = gain, step
                best_state = {key: v.detach().clone() for key, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    return net.eval(), best_gain, best_step, curve


def fit_neural(opt: Option, train, val, settings: NeuralSettings, seed: int) -> Policy:
    """Backward induction over exercise dates. At each date a fresh network is trained
    on in-the-money states (out of the money, stopping pays 0 <= continuing, so these
    states carry no information for the objective). With settings.fresh_paths=True
    the training set is ignored and a new batch of paths is simulated at every step,
    as in the paper."""
    s = settings
    policy = Policy(method=f"nn_{s.variant}", info=asdict(s))
    va = Backward(opt, val)
    tr = None if s.fresh_paths else Backward(opt, train)
    rng = np.random.default_rng(seed_for(seed, "fresh", opt.d)) if s.fresh_paths else None
    gen = torch.Generator().manual_seed(int(seed) % (2 ** 31))
    start = time.perf_counter()

    for k in range(opt.dates - 1, 0, -1):
        t0 = time.perf_counter()
        xv = val[:, k, :]
        gv = opt.payoff(xv)
        iv = gv > 0
        Xv = nn_features(xv[iv], opt, s)
        Av = (gv[iv] - va.future(k)[iv]).astype(np.float32)

        if s.fresh_paths:
            def simulate(n):
                paths = gbm_paths(opt, n, rng=rng)
                g = opt.payoff(paths[:, k, :])
                pay, tau = cashflows(opt, paths, policy, guarded=True, k_from=k + 1)
                fut = pay * np.exp(-opt.rate * (tau - k) * opt.dt)
                itm = g > 0
                return nn_features(paths[itm, k, :], opt, s), (g[itm] - fut[itm]).astype(np.float32)
            X_ref, A_ref = simulate(16_384)                  # pilot batch for scaling only
            sampler = lambda step: simulate(s.batch)
        else:
            x = train[:, k, :]
            g = opt.payoff(x)
            it = g > 0
            X_ref = nn_features(x[it], opt, s)
            A_ref = (g[it] - tr.future(k)[it]).astype(np.float32)
            batch = max(2, min(s.batch, len(X_ref)))

            def sampler(step, X=X_ref, A=A_ref, batch=batch):
                idx = torch.randint(len(X), (batch,), generator=gen).numpy()
                return X[idx], A[idx]

        if len(X_ref) < 50:
            policy.rules[k] = None
            best_gain, best_step, curve = 0.0, 0, []
        else:
            mu = X_ref.mean(axis=0)
            sd = np.maximum(X_ref.std(axis=0), 1e-3).astype(np.float32)
            a_scale = float(max(np.std(A_ref), 1e-6))       # constant rescaling: same optimum
            net, best_gain, best_step, curve = _train_one_date(
                opt, k, s, sampler, Xv, Av, len(xv), mu, sd, a_scale, seed=int(seed) * 100 + k)
            policy.rules[k] = NeuralRule(net, mu, sd, opt, s)
        gain = guard_and_update(opt, policy, k, va, tr)
        policy.diagnostics.append(dict(date=k, n_itm=int(len(X_ref)), val_gain=gain,
                                       guard=policy.guard[k], best_step=best_step,
                                       fit_s=time.perf_counter() - t0, curve=curve))
    elapsed = time.perf_counter() - start
    policy.info.update(train_s=elapsed, selection_s=elapsed)   # settings are frozen: no per-seed search
    return policy


# =============================================================================
# 6. Experimental units and statistics
# =============================================================================
def mean_ci(x, level=0.95):
    """Mean and Student-t interval ACROSS SEEDS (training variability, conditional on
    the shared test set). Never used to pool evaluations of the same test paths."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    m = float(x.mean()) if len(x) else np.nan
    if len(x) < 2:
        return m, np.nan, np.nan, np.nan
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    h = float(student_t.ppf(0.5 + level / 2, len(x) - 1) * se)
    return m, se, m - h, m + h


def paired(a, b):
    """Paired Monte Carlo statistics for a - b on the SAME test paths (one seed)."""
    diff = np.asarray(a) - np.asarray(b)
    return float(diff.mean()), float(diff.std(ddof=1) / math.sqrt(len(diff)))


def paired_row(key, guarded, contrast, diff, se):
    """One paired comparison for one seed, with its 95% Monte Carlo (test-path) interval.
    This interval is NOT the across-seed interval; the two are reported separately."""
    return dict(**key, guarded=guarded, contrast=contrast, diff=diff, paired_se=se,
                mc_ci_low=diff - 1.96 * se, mc_ci_high=diff + 1.96 * se)


ABLATION_CONTRASTS = {   # each is a linear combination of per-path payoffs
    "sorting effect | payoff": {"sorted+payoff": 1, "raw+payoff": -1},
    "sorting effect | no payoff": {"sorted": 1, "raw": -1},
    "payoff effect | sorted": {"sorted+payoff": 1, "sorted": -1},
    "payoff effect | raw": {"raw+payoff": 1, "raw": -1},
    "interaction": {"sorted+payoff": 1, "raw+payoff": -1, "sorted": -1, "raw": 1},
}


def run_unit(opt: Option, n_train: int, seed: int, variants, settings: NeuralSettings,
             n_val: int, test_paths, reference: dict, base_seed: int, budget_label=None):
    """Fit LSM and every requested neural variant on the SAME training and validation
    paths of this seed, evaluate all of them (guarded and unguarded) on the SAME test
    paths, and return tidy rows. Nothing here selects anything using test paths."""
    case = opt.name
    t0 = time.perf_counter()
    train = gbm_paths(opt, n_train, seed_for(base_seed, "train", opt.d, seed))
    val = gbm_paths(opt, n_val, seed_for(base_seed, "val", opt.d, seed))
    sim_s = time.perf_counter() - t0

    policies = {}
    lsm, _ = select_lsm(opt, train, val)
    policies["lsm"] = lsm
    for v in variants:
        policies[f"nn_{v}"] = fit_neural(opt, train, val, with_variant(settings, v),
                                         seed=seed_for(base_seed, "init", opt.d, seed))

    key = dict(case=case, d=opt.d, budget=n_train, seed=seed)
    runs, pairs, dates, payoffs = [], [], [], {}
    for name, pol in policies.items():
        for guarded in (True, False):
            t1 = time.perf_counter()
            values, tau = discounted_payoffs(opt, test_paths, pol, guarded)
            eval_s = time.perf_counter() - t1
            payoffs[(name, guarded)] = values
            v = float(values.mean())
            runs.append(dict(**key, method=name, guarded=guarded, value=v,
                             test_se=float(values.std(ddof=1) / math.sqrt(len(values))),
                             gap=reference["value"] - v,
                             early_exercise=float(np.mean(tau < opt.dates)),
                             n_guarded_dates=pol.n_guarded(),
                             train_s=pol.info["train_s"], selection_s=pol.info["selection_s"],
                             sim_s=sim_s, eval_s=eval_s, n_val=n_val, n_test=len(test_paths),
                             lsm_degree=pol.info.get("degree"), lsm_ridge=pol.info.get("ridge")))
        for row in pol.diagnostics:
            dates.append(dict(**key, method=name, **{k: v for k, v in row.items() if k != "curve"}))

    for guarded in (True, False):
        for name in policies:
            if name != "lsm":
                m, se = paired(payoffs[(name, guarded)], payoffs[("lsm", guarded)])
                pairs.append(paired_row(key, guarded, f"{name} - lsm", m, se))
        if all(f"nn_{v}" in policies for v in VARIANTS):
            for cname, weights in ABLATION_CONTRASTS.items():
                comb = sum(w * payoffs[(f"nn_{v}", guarded)] for v, w in weights.items())
                pairs.append(paired_row(key, guarded, cname, float(comb.mean()),
                                        float(comb.std(ddof=1) / math.sqrt(len(comb)))))
    for name in policies:   # what the guard changes: guarded - unguarded
        m, se = paired(payoffs[(name, True)], payoffs[(name, False)])
        pairs.append(paired_row(key, True, f"{name}: guarded - unguarded", m, se))
    return dict(runs=runs, paired=pairs, dates=dates)


# =============================================================================
# 7. Results storage: one line per unit, resumable, frozen configuration
# =============================================================================
def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def code_hash() -> str:
    """Hash all project Python modules and notebook code cells, not prose/outputs.

    Includes filenames; excludes tests, caches and temporary notebook copies.
    """
    digest = hashlib.sha256()
    for path in sorted(PROJECT_ROOT.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    for path in sorted(PROJECT_ROOT.glob("[0-9][0-9]_*.ipynb")):
        notebook = json.loads(path.read_text())
        digest.update(path.name.encode())
        digest.update(json.dumps([c["source"] for c in notebook["cells"] if c["cell_type"] == "code"], sort_keys=True).encode())
    return digest.hexdigest()[:16]


def calibration_provenance(mode: str) -> dict:
    from importlib.metadata import version
    import platform
    return dict(code_hash=code_hash(), config_hash=stable_hash(CONFIG), mode=mode,
                environment=dict(python=platform.python_version(), packages={p: version(p) for p in
                    ("numpy", "scipy", "pandas", "matplotlib", "torch", "threadpoolctl", "ipykernel")}))


def calibration_path(out: Path, mode: str) -> Path:
    """Resume only matching provenance; keep unversioned/old evidence untouched."""
    provenance = calibration_provenance(mode)
    path = Path(out) / f"calibration-{stable_hash(provenance)}.csv"
    meta = path.with_suffix(".json")
    if meta.exists():
        if json.loads(meta.read_text()) != provenance:
            raise RuntimeError("Calibration provenance mismatch")
    elif path.exists():
        raise RuntimeError("Calibration CSV has no provenance; refusing resume")
    else:
        meta.write_text(json.dumps(provenance, indent=2))
    return path


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


class ResultStore:
    """Each finished unit is ONE line of units.jsonl, written with a single append and
    flushed to disk. A unit therefore either exists completely or not at all: an
    interrupted write leaves at most a malformed last line, which is ignored, so that
    unit simply runs again. The runs/paired/dates tables are built from this file.
    A full run refuses to resume if the frozen configuration (study design, neural
    settings, contract parameters, code hash) differs from the one it started with."""
    TABLES = ("runs", "paired", "dates")

    def __init__(self, folder: Path, run_info: dict, mode: str):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "units.jsonl"
        info_path = self.folder / "run_info.json"
        frozen = json.dumps(run_info["frozen"], sort_keys=True, default=str)
        if info_path.exists():
            old = json.loads(info_path.read_text())
            if json.dumps(old.get("frozen"), sort_keys=True, default=str) != frozen:
                if mode == "full":
                    raise RuntimeError(
                        f"{info_path} was written with a different frozen configuration or code version. "
                        "Restore the original configuration/code, or move the folder away to start a new study.")
                # Preserve old quick evidence; deterministic sibling for this config.
                namespace = self.folder.with_name(self.folder.name + "-" + stable_hash(run_info["frozen"]))
                self.__init__(namespace, run_info, mode)
                return
        info_path.write_text(json.dumps(run_info, indent=2, default=str))

    def units(self) -> list:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                unit = json.loads(line)
            except json.JSONDecodeError:
                continue                               # interrupted write: ignored, unit reruns
            if all(t in unit for t in self.TABLES):
                out.append(unit)
        return out

    def done(self) -> set:
        return {tuple(u["key"]) for u in self.units()}

    def _repair_tail(self):
        """Make sure the file ends with a newline before the next append.
        If the last line has no newline (an interrupted write):
          * it is a complete, valid unit record -> keep it and just add the missing newline;
          * it is incomplete or malformed      -> cut it off (that unit was never counted
            as finished, so it simply runs again).
        Either way the next record starts on a clean line instead of being glued onto it."""
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        if not data or data.endswith(b"\n"):
            return
        start = data.rfind(b"\n") + 1                  # start of the unterminated last line
        try:
            record = json.loads(data[start:].decode("utf-8"))
            complete = isinstance(record, dict) and all(t in record for t in self.TABLES)
        except (UnicodeDecodeError, json.JSONDecodeError):
            complete = False
        if complete:
            with open(self.path, "ab") as f:
                f.write(b"\n")
        else:
            with open(self.path, "r+b") as f:
                f.truncate(start)

    def append(self, unit: dict, key: tuple):
        self._repair_tail()
        line = json.dumps(dict(key=list(key), **{t: unit[t] for t in self.TABLES}), default=_jsonable)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            import os
            os.fsync(f.fileno())

    def load(self, name: str) -> pd.DataFrame:
        latest = {tuple(u["key"]): u for u in self.units()}   # if a unit appears twice, keep the last
        rows = [r for u in latest.values() for r in u[name]]
        return pd.DataFrame(rows)
