"""Shared notebook-01 sections; full timing uses these exact functions and assertions."""
import json, time
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
from stopping import *

def validation_checks():
    put = validation_put()
    x = gbm_paths(put, 50_000, seed=1)
    expected = put.s0 * np.exp((put.rate - put.dividend) * put.maturity)
    assert abs(x[:, -1, 0].mean() - expected) < 4 * x[:, -1, 0].std() / np.sqrt(len(x)), "risk-neutral mean"
    
    corr_opt = Option(kind="max_call", d=3, rho=0.3, dividend=0.1, maturity=3.0)
    r = np.diff(np.log(gbm_paths(corr_opt, 50_000, seed=2)), axis=1)[:, 0, :]
    assert abs(np.corrcoef(r.T)[0, 1] - 0.3) < 0.02, "correlation"
    
    opt5 = paper_max_call(5)
    assert np.array_equal(gbm_paths(opt5, 1_000, seed=3), gbm_paths(opt5, 5_000, seed=3)[:1_000]), "nested training sets"
    
    # Discounting and exercise mechanics of the forward evaluator
    paths = gbm_paths(put, 1_000, seed=4)
    never = Policy("never")                                         # no rules: hold to maturity
    v, tau = discounted_payoffs(put, paths, never)
    assert np.allclose(v, put.payoff(paths[:, -1, :]) * np.exp(-put.rate * put.maturity)) and (tau == put.dates).all(), "terminal discounting"
    first = Policy("first", rules={1: lambda z: np.ones(len(z), bool)})   # exercise at t_1 whenever in the money
    v, tau = discounted_payoffs(put, paths, first)
    g1 = put.payoff(paths[:, 1, :])
    assert np.allclose(v[g1 > 0], g1[g1 > 0] * np.exp(-put.rate * put.dt)) and (tau[g1 > 0] == 1).all(), "early-exercise discounting"
    assert (tau[g1 == 0] > 1).all(), "never exercise out of the money"
    print("Discounting and exercise checks passed.")
    
    crr = {n: crr_bermudan_put(put, n) for n in (40, 80, 160, 320, 1280, 2560)}
    assert all(abs(crr[n] - crr[320]) < 0.01 for n in (80, 160)), "CRR convergence"
    assert crr[320] >= 5.5735 - 1e-3, "Bermudan put >= European put (5.5735)"
    print("CRR Bermudan put by steps per exercise interval:", {k: round(v, 5) for k, v in crr.items()})
    put_ref = reference_for(put)
    print("1D reference:", put_ref)
    print("Published max-call references:", {d: PAPER_REFERENCES[d]["value"] for d in (2, 5)})
    assert abs(crr[2560] - put_ref["convolution"]) <= put_ref["grid_band"]
    print("Grid band (heuristic, not MC error):", put_ref["grid_band"])
    return locals()

def validation_put_fit(MODE, cfg, BASE, OUT, CHECK_SEED, put, put_ref):
    test_put = gbm_paths(put, cfg["test"], seed_for(BASE, "test", 1))
    put_rows = []
    for seed in range(CHECK_SEED, CHECK_SEED + cfg["put_seeds"]):
        unit = run_unit(put, cfg["put_train"], seed, ["sorted+payoff"], NeuralSettings(**cfg["nn_settings"]),
                        cfg["val"], test_put, put_ref, BASE)
        put_rows += unit["runs"]
    put_df = pd.DataFrame(put_rows)
    put_df["ci95"] = 1.96 * put_df.test_se
    put_df.to_csv(OUT / "validation_put.csv", index=False)
    print(f"CRR reference: {put_ref['value']:.4f}")
    put_df[["seed", "method", "guarded", "value", "ci95", "gap", "n_guarded_dates", "train_s"]].round(4)
    # A pre-specified 4-SE implementation diagnostic, not a 95% coverage claim.
    # Low-training smoke policies can be suboptimal; only the full fit gates agreement.
    put_df["agreement_4se_plus_grid"] = (
        abs(put_df.value - put_ref["value"]) <= 4*put_df.test_se + put_ref["grid_band"])
    if MODE == "full":
        assert put_df.loc[put_df.guarded, "agreement_4se_plus_grid"].all(), "1D fit disagrees with refined reference"
    print(put_df[["method", "seed", "guarded", "agreement_4se_plus_grid"]])
    return locals()

def validation_convergence(MODE, cfg, BASE, OUT, CHECK_SEED):
    lsm_rows = []
    dimensions = (2, 5) if MODE == "quick" else (5,)
    for d in dimensions:
        opt = paper_max_call(d)
        ref = reference_for(opt)
        test = gbm_paths(opt, cfg["test"], seed_for(BASE, "check", d, extra=1))   # NOT the main-study test set
        for check_seed in range(CHECK_SEED, CHECK_SEED + 3):
            val = gbm_paths(opt, cfg["val"], seed_for(BASE, "val", d, check_seed))
            for n in (cfg["lsm_budgets"] if MODE == "quick" else (1000, 10000, 50000)):
                train = gbm_paths(opt, n, seed_for(BASE, "train", d, check_seed))
                pol, table = select_lsm(opt, train, val)
                v = discounted_payoffs(opt, test, pol)[0]
                lsm_rows.append(dict(d=d, budget=n, seed=check_seed, value=v.mean(), ci95=1.96 * v.std(ddof=1) / np.sqrt(len(v)),
                                     gap=ref["value"] - v.mean(), degree=pol.info["degree"], ridge=pol.info["ridge"],
                                     selection_s=pol.info["selection_s"]))
    lsm_df = pd.DataFrame(lsm_rows)
    lsm_df.to_csv(OUT / "validation_lsm_convergence.csv", index=False)
    
    fig, axes = plt.subplots(1, len(dimensions), figsize=(5.5*len(dimensions), 3.6), squeeze=False)
    for ax, d in zip(axes.ravel(), dimensions):
        sub = lsm_df[lsm_df.d == d].groupby("budget", as_index=False)[["gap", "ci95"]].mean()
        ref = PAPER_REFERENCES[d]
        ax.axhspan(ref["value"] - ref["high"], ref["value"] - ref["low"], color="grey", alpha=0.25, label="reference uncertainty")
        ax.errorbar(sub.budget, sub.gap, yerr=sub.ci95, fmt="o-", capsize=3, label="tuned LSM")
        ax.axhline(0, color="black", lw=0.8)
        ax.set(xscale="log", title=f"{d}D max-call", xlabel="Training paths", ylabel="Gap = reference - value")
        ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "validation_lsm_convergence.png", dpi=160); plt.show()
    lsm_df.round(4)
    return locals()

def validation_paper(cfg, BASE, OUT, CHECK_SEED):
    opt = paper_max_call(5)
    ref = reference_for(opt)
    test5 = gbm_paths(opt, cfg["test"], seed_for(BASE, "check", 5, extra=1))  # NOT the main-study test set
    val5 = gbm_paths(opt, cfg["val"], seed_for(BASE, "val", 5, CHECK_SEED))
    paper_settings = NeuralSettings(sorted_inputs=False, payoff_input=True, lr=1e-3, fresh_paths=True, **cfg["paper_check"])
    t0 = time.perf_counter()
    paper_policy = fit_neural(opt, None, val5, paper_settings, seed=seed_for(BASE, "check", 5))
    elapsed = time.perf_counter() - t0
    v = discounted_payoffs(opt, test5, paper_policy)[0]
    check = dict(value=v.mean(), ci95=1.96 * v.std(ddof=1) / np.sqrt(len(v)), reference=ref["value"],
                 reference_ci=[ref["low"], ref["high"]], gap=ref["value"] - v.mean(),
                 bcj_lower=ref["lower"], bcj_upper=ref["upper"], bcj_midpoint=ref["midpoint"],
                 gap_to_lower=ref["lower"]-v.mean(), gap_to_midpoint=ref["midpoint"]-v.mean(),
                 guarded_dates=paper_policy.n_guarded(), minutes=elapsed / 60, settings=asdict(paper_settings))
    (OUT / "validation_paper_check.json").write_text(json.dumps(check, indent=2, default=float))
    print(json.dumps(check, indent=2, default=float))
    return locals()

def validation_dual(MODE, cfg, BASE, OUT, CHECK_SEED, put, put_ref):
    from dual import policy_dual
    
    dual_train = gbm_paths(put, cfg["put_train"], seed_for(BASE, "train", 1, CHECK_SEED, extra=81))
    dual_val = gbm_paths(put, cfg["val"], seed_for(BASE, "val", 1, CHECK_SEED, extra=81))
    dual_policy = select_lsm(put, dual_train, dual_val)[0]
    dual_rows = []
    for inner in ([32, 128] if MODE == "quick" else [1600, 3200]):
        result = policy_dual(put, dual_policy, outer=128 if MODE == "quick" else 2000,
                             inner=inner, base_seed=BASE, seed=85101,
                             outer_chunk=16, memory_mb=256)
        row = dict(inner=inner, metadata=result["metadata"], reference=put_ref,
                   **{key: result[key] for key in ("lower", "upper", "gap")})
        row["brackets_4se_plus_grid"] = bool(
            result["lower"]["mean"] - 4*result["lower"]["se"] <= put_ref["high"] and
            result["upper"]["mean"] + 4*result["upper"]["se"] >= put_ref["low"])
        dual_rows.append(row)
    (OUT / "validation_dual.json").write_text(json.dumps(dual_rows, indent=2, default=float))
    display(pd.DataFrame([dict(inner=r["inner"], lower=r["lower"]["mean"],
                              lower_se=r["lower"]["se"], upper=r["upper"]["mean"],
                              upper_se=r["upper"]["se"], gap=r["gap"]["mean"],
                              paired_gap_se=r["gap"]["se"],
                              brackets=r["brackets_4se_plus_grid"]) for r in dual_rows]))
    if MODE == "full":
        assert all(r["brackets_4se_plus_grid"] for r in dual_rows), "Primal/dual disagrees with CRR"
    return locals()

