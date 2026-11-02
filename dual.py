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
import time
import numpy as np
from stopping import gbm_paths, exercise_mask, discounted_payoffs, seed_for


def continuation(opt, states, k, policy, inner, rng, guarded=True):
    """Estimate discounted continuation via independent conditional rollouts.

    Each outer state is replicated across inner paths. Only simulated
    post-k shocks enter future stopping decisions, and cash flows are
    discounted to time zero. This targets the fixed policy's continuation,
    not optimal continuation.
    """
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
    """Return paired policy payoffs, finite-inner dual estimates and diagnostics.

    The expected dual maximum is an upper bound for an independently fixed
    policy; an individual sample or confidence endpoint is not guaranteed
    to bound the true price. A paired difference reduces outer sampling
    noise but does not remove inner-simulation bias.

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


def nested_policy_dual(opt, policy, *, budgets, outer=2000, seed=912,
                       group=200, memory_mb=256, guarded=True, max_seconds=None,
                       outer_indices=None):
    """Evaluate nested inner budgets on coupled simulation streams.

    Coupling improves precision of comparisons between budgets, but does not
    eliminate the finite-inner upward bias of the upper estimator.

    Each (outer index, date, group) has a distinct SeedSequence. Outer randomness
    uses a separate role. No future outer states enter the inner simulation.
    Processing one outer state bounds memory independently of the largest budget.
    Increasing budgets leaves every earlier sample bitwise unchanged.
    outer_indices selects global indices from the fixed outer population; both
    inner seeds and outer states match an unsplit run, including doubling subsets.
    Seed roles use the registered base 2026. value_gap estimates U-C_hat_0:
    its expectation is the same policy gap, often with much less outer noise
    than U-Z_tau. It retains inner bias and is not a pathwise bound.
    """
    started = time.perf_counter()
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive")
    budgets = tuple(budgets)
    if (not budgets or outer < 2 or group < 1 or
        any(not isinstance(n, int) or n < group or n % group for n in budgets) or
        tuple(sorted(set(budgets))) != budgets):
        raise ValueError('sorted distinct budgets must be positive multiples of group')
    estimated_bytes = (8*16*group*(opt.d+1) +
                       8*outer*(opt.dates+1)*(opt.d+len(budgets)+5))
    if not np.isfinite(memory_mb) or memory_mb <= 0 or estimated_bytes > memory_mb*1024**2:
        raise ValueError('working arrays exceed memory cap')
    osd = seed_for(2026, 'dual_outer', opt.d, seed)
    isd = seed_for(2026, 'dual_inner', opt.d, seed)
    if osd == isd:
        raise ValueError('seed collision')
    # Global outer indices also key inner streams: chunking and doubling never
    # change the experiment. Generate the same outer population for each chunk.
    indices = np.arange(outer) if outer_indices is None else np.asarray(outer_indices)
    if (indices.ndim != 1 or len(indices) < 2 or
        not np.issubdtype(indices.dtype, np.integer) or
        np.any(indices < 0) or np.any(indices >= outer) or
        np.any(np.diff(indices) <= 0)):
        raise ValueError('outer_indices must be sorted distinct integers within outer')
    paths = gbm_paths(opt, outer, seed=osd)[indices]
    population = outer
    outer = len(indices)
    z = opt.payoff(paths)*opt.discount(np.arange(opt.dates+1))
    lower = discounted_payoffs(opt, paths, policy, guarded)[0]
    c = np.empty((len(budgets), outer, opt.dates))
    for a in range(outer):
        if max_seconds is not None and time.perf_counter()-started >= max_seconds:
            raise TimeoutError("nested dual exceeded wall-time allowance; incomplete run")
        for k in range(opt.dates):
            total = 0.
            for g in range(budgets[-1]//group):
                rng = np.random.default_rng(np.random.SeedSequence([isd,int(indices[a]),k,g]))
                total += continuation(opt, paths[a:a+1,k], k, policy, group, rng, guarded)[0]
                n = (g+1)*group
                if n in budgets:
                    c[budgets.index(n),a,k] = total/(g+1)
    results = []
    for i,n in enumerate(budgets):
        mart = np.zeros((outer,opt.dates+1))
        for k in range(1,opt.dates+1):
            lk = z[:,k] if k == opt.dates else np.where(
                exercise_mask(opt,policy,k,paths[:,k],guarded),z[:,k],c[i,:,k])
            mart[:,k] = mart[:,k-1]+lk-c[i,:,k-1]
        u = np.max(z[:,1:]-mart[:,1:],axis=1)
        results.append(dict(upper=_summary(u), upper_samples=u,
            lower=_summary(lower), lower_samples=lower, gap=_summary(u-lower),
            value_gap=_summary(u-c[i,:,0]), value_gap_samples=u-c[i,:,0],
            initial_value_samples=c[i,:,0].copy(), martingale_samples=mart,
            metadata=dict(outer=outer,inner=n,outer_seed=osd,inner_seed=isd,
                          seed=seed,group=group,outer_population=population,
                          outer_indices=indices.tolist(),estimated_working_bytes=estimated_bytes,
                          coupling='IID group prefix')))
    return results
