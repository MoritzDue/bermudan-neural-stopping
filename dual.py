"""Nested policy-value dual, following Andersen & Broadie (2004),
Management Science 50(9), 1222–1234, doi:10.1287/mnsc.1040.0258.

For the outer GBM filtration F_k let Z_k=discount(k)*g(S_k),
C_k=E[Z_tau(k+1)|F_k], L_k=Z_k on a policy stop, else C_k,
and L_N=Z_N. Then E[L_(k+1)|F_k]=C_k. Thus M_0=0 and
M_(k+1)-M_k=L_(k+1)-C_k define a martingale. For any stopping
time tau, E[Z_tau-M_tau]=E[Z_tau], hence E[max_{1..N}(Z-M)]
is an upper bound. On continuing dates L_k=C_k, giving telescoping.

Estimate every C_k (including k=0) with fresh independent inner rollouts
of the fixed policy, discounted to zero. Conditional on the WHOLE outer
path, each estimate is unbiased; linearity gives E[Mhat|outer]=M.
Conditional Jensen for the maximum therefore preserves the upper bound
in expectation and adds nonnegative inner bias. Mhat need NOT be a
martingale in a filtration revealing its inner samples at each date:
-C_hat_k is then already known. Mean-zero unconditional increments alone
are not a proof. Neither an individual maximum nor its sample mean is
a guaranteed upper bound. Outer CIs describe sampling error around the
finite-inner expectation; they do not remove inner bias.

Policies must be fixed deterministic state rules, trained independently
of these paths. No inner estimate is used to decide exercise. Primary
paper verification is pending offline source access.
"""
import numpy as np
from stopping import gbm_paths, exercise_mask, discounted_payoffs, seed_for


def continuation(opt, states, k, policy, inner, rng, guarded=True):
    """Independent continuations from t_k; stream states, not path tensors."""
    x = np.repeat(states, inner, axis=0)
    pay = np.zeros(len(x))
    alive = np.ones(len(x), dtype=bool)
    corr = np.full((opt.d, opt.d), opt.rho)
    np.fill_diagonal(corr, 1.)
    chol = np.linalg.cholesky(corr)
    drift = (opt.rate-opt.dividend-opt.sigma**2/2)*opt.dt
    sd = opt.sigma*np.sqrt(opt.dt)
    for j in range(k+1, opt.dates+1):
        # Draw for all paths to keep random draws independent of policy decisions.
        x *= np.exp(drift + sd*(rng.standard_normal(x.shape) @ chol.T))
        stop = np.ones(len(x), bool) if j == opt.dates else exercise_mask(opt, policy, j, x, guarded)
        hit = alive & stop
        pay[hit] = opt.payoff(x[hit])*opt.discount(j)
        alive[hit] = False
    return pay.reshape(len(states), inner).mean(axis=1)


def _summary(x):
    mean = np.mean(x, axis=0)
    se = np.std(x, axis=0, ddof=1)/np.sqrt(len(x))
    return dict(mean=mean, se=se, ci_low=mean-1.96*se, ci_high=mean+1.96*se)


def policy_dual(opt, policy, *, outer=1000, inner=200, base_seed=2026,
                seed=0, outer_chunk=32, memory_mb=256, guarded=True):
    """Return paired primal/dual samples, 95% normal CIs and M_k statistics.

    memory_mb caps a conservative working-array estimate, not process RSS or
    arbitrary policy inference allocations. The caller must measure process RSS.
    Outer paths are generated internally with a separate seed role. Chunk-specific
    SeedSequences separate dates and chunks without a 32-bit derived-seed collision.
    Changing chunk size changes random streams; record it as configuration.
    """
    for name, value in [('outer',outer),('inner',inner),('outer_chunk',outer_chunk)]:
        if not isinstance(value, int) or value < (2 if name == 'outer' else 1):
            raise ValueError(name+' must be a positive integer (outer >= 2)')
    if not np.isfinite(memory_mb) or memory_mb <= 0:
        raise ValueError('memory_mb must be positive')
    bytes_per_outer = 8*(16*inner*(opt.d+1)+8*(opt.dates+1)*opt.d)
    chunk = min(outer_chunk, int(memory_mb*1024**2//bytes_per_outer))
    if chunk < 1:
        raise ValueError('memory cap cannot hold one outer path and its inner batch')
    osd = seed_for(base_seed, 'dual_outer', opt.d, seed)
    isd = seed_for(base_seed, 'dual_inner', opt.d, seed)
    if osd == isd:
        raise ValueError('seed collision')
    outer_rng = np.random.default_rng(osd)
    upper, lower = np.empty(outer), np.empty(outer)
    mart = np.zeros((outer, opt.dates+1))
    for block, a in enumerate(range(0, outer, chunk)):
        b = min(a+chunk, outer)
        paths = gbm_paths(opt, b-a, rng=outer_rng)
        lower[a:b] = discounted_payoffs(opt, paths, policy, guarded)[0]
        z = opt.payoff(paths)*opt.discount(np.arange(opt.dates+1))
        c = np.empty((b-a, opt.dates))
        for k in range(opt.dates):
            rng = np.random.default_rng(np.random.SeedSequence([isd, block, k]))
            c[:, k] = continuation(opt, paths[:, k], k, policy, inner, rng, guarded)
        for k in range(1, opt.dates+1):
            lk = z[:, k] if k == opt.dates else np.where(
                exercise_mask(opt, policy, k, paths[:, k], guarded), z[:, k], c[:, k])
            mart[a:b, k] = mart[a:b, k-1]+lk-c[:, k-1]
        upper[a:b] = np.max(z[:, 1:]-mart[a:b, 1:], axis=1)
    return dict(upper=_summary(upper), lower=_summary(lower), gap=_summary(upper-lower),
                martingale=_summary(mart), increments=_summary(np.diff(mart, axis=1)),
                upper_samples=upper, lower_samples=lower, martingale_samples=mart,
                metadata=dict(outer=outer, inner=inner, outer_chunk=chunk, memory_mb=memory_mb,
                              estimated_chunk_bytes=chunk*bytes_per_outer, outer_seed=osd,
                              inner_seed=isd, base_seed=base_seed, seed=seed, guarded=guarded,
                              ci='95% asymptotic normal; conditional on fixed policy; inner bias retained'))
