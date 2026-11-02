"""Diagnostic experiments for the nested primal-dual estimator.

The size sweeps, conditional-drift checks and numerical oracles assess whether
the simulated estimator behaves consistently with its theoretical properties.
They are not themselves formal convergence certificates. Outputs are treated
as smoke or development evidence, separate from the frozen research results.

Run from the repository root (e.g. python -c "from tools.dual_validation import ..."). Output is JSON on stdout; no cached evidence
is overwritten. Main-study sizes remain unset unless all budget gates pass.
"""
import json
import math
import resource
import time
from dataclasses import replace
import numpy as np
from scipy.stats import beta
from stopping import (Policy, validation_put, paper_max_call, gbm_paths,
                      select_lsm, discounted_payoffs, reference_for, limit_threads,
                      seed_for, code_hash, calibration_provenance)
from dual import policy_dual, _summary


def crr_boundary_policy(opt, steps_per_date=2560):
    """Extract exercise boundaries from the refined CRR lattice.

    Between lattice nodes use a price threshold. This is an approximate state
    policy, not an exact off-lattice CRR stopping rule; grid error remains.
    """
    steps = steps_per_date * opt.dates
    dt = opt.maturity / steps
    up = math.exp(opt.sigma * math.sqrt(dt))
    p = (math.exp((opt.rate-opt.dividend)*dt)-1/up)/(up-1/up)
    j = np.arange(steps+1)
    v = np.maximum(opt.strike-opt.s0*up**(2*j-steps), 0.)
    boundaries = {}
    for k in range(steps-1, -1, -1):
        v = math.exp(-opt.rate*dt)*((1-p)*v[:-1]+p*v[1:])
        if k > 0 and k % steps_per_date == 0:
            s = opt.s0*up**(2*np.arange(k+1)-k)
            g = np.maximum(opt.strike-s, 0.)
            ex = (g > v) & (g > 0)
            boundaries[k//steps_per_date] = float(s[ex].max()) if ex.any() else 0.
            v = np.maximum(v,g)
    policy = Policy('crr_boundary', rules={k:(lambda x,b=b:x[:,0]<=b)
                    for k,b in boundaries.items()})
    return policy, boundaries, float(v[0])


def independent_gap(upper, primal):
    """Independent sample errors combine in quadrature, without inner-bias removal."""
    u, l = _summary(upper), _summary(primal)
    m = u['mean']-l['mean']
    se = np.hypot(u['se'],l['se'])
    return dict(mean=m,se=se,ci_low=m-1.96*se,ci_high=m+1.96*se)


def conditional_drift(opt, result, bins=5):
    """Bins use only S_k; do not condition on future states or inner samples.

    Individual 3-SE checks are not simultaneous guarantees (41 tests in 1D).
    """
    paths = gbm_paths(opt,result['metadata']['outer'],seed=result['metadata']['outer_seed'])
    increments = np.diff(result['martingale_samples'],axis=1)
    rows=[]
    for k in range(opt.dates):
        groups = [np.arange(len(paths))] if k==0 else np.array_split(np.argsort(paths[:,k,0]),bins)
        for b, ix in enumerate(groups):
            s = _summary(increments[ix,k])
            rows.append(dict(date=k,bin=b,n=len(ix),**s,
                             within_3se=bool(abs(s['mean']) <= 3*s['se'])))
    return rows


def run_validation():
    limit_threads(1)
    start=time.perf_counter()
    o=validation_put(); ref=reference_for(o)
    boundary,bounds,tree=crr_boundary_policy(o)
    lsm,_=select_lsm(o,gbm_paths(o,10000,seed=71001),gbm_paths(o,5000,seed=71002))
    immediate=Policy('immediate',rules={k:lambda x:np.ones(len(x),bool) for k in range(1,o.dates)})
    report=dict(kind='bounded dual validation, not full research results',source_hash=code_hash(),
                environment=calibration_provenance('quick')['environment'],reference=ref,
                crr_value=tree,boundaries=bounds,policies={},sweep=[],coverage={})
    primal_paths=gbm_paths(o,500000,seed=71003)
    results={}
    for name,pol in [('crr',boundary),('lsm',lsm),('never',Policy('never')),('immediate',immediate)]:
        r=policy_dual(o,pol,outer=3000,inner=800,seed=71004)
        primal=discounted_payoffs(o,primal_paths,pol)[0]
        results[name]=r
        report['policies'][name]=dict(upper=r['upper'],paired_gap=r['gap'],
            independent_primal=_summary(primal),independent_gap=independent_gap(r['upper_samples'],primal),
            metadata=r['metadata'],lower_ci_below_reference=bool(r['upper']['ci_low']<=ref['high']),
            upper_within_3se=bool(r['upper']['mean']+3*r['upper']['se']>=ref['low']))
    previous=None
    for inner in (50,200,800,1600,3200):
        r=results['crr'] if inner==800 else policy_dual(o,boundary,outer=3000,inner=inner,seed=71004)
        row=dict(inner=inner,upper=r['upper'],paired_gap=r['gap'])
        if previous is not None:
            row['previous_minus_current']=_summary(previous['upper_samples']-r['upper_samples'])
        report['sweep'].append(row); previous=r
    report['conditional_drift']=conditional_drift(o,results['crr'])
    violations=0
    coverage_rows=[]
    for rep in range(50):
        r=policy_dual(o,boundary,outer=300,inner=200,seed=72000+rep)
        violation=bool(r['upper']['ci_high']<ref['low'])
        violations+=violation
        coverage_rows.append(dict(seed=72000+rep,upper=r['upper'],violation=violation))
    report['coverage']=dict(repetitions=50,violations=violations,rate=violations/50,
        binomial_ci=[0. if violations==0 else beta.ppf(.025,violations,51-violations),
                     1. if violations==50 else beta.ppf(.975,violations+1,50-violations)],
        rows=coverage_rows,note='Coverage of finite-inner U against CRR; upward bias makes this conservative.')
    o5=replace(paper_max_call(5),dividend=0.)
    p5,_=select_lsm(o5,gbm_paths(o5,10000,seed=73001),gbm_paths(o5,5000,seed=73002))
    report['no_dividend']=[]
    for name,pol in [('never',Policy('never')),('lsm',p5)]:
        for inner in (200,800,3200):
            r=policy_dual(o5,pol,outer=500,inner=inner,seed=73003)
            report['no_dividend'].append(dict(policy=name,inner=inner,upper=r['upper'],
                exact=58.80711789683454,passes_3se=bool(r['upper']['mean']+3*r['upper']['se']>=58.80711789683454)))
    last,penultimate=report['sweep'][-1],report['sweep'][-2]
    stable=abs(last['upper']['mean']-penultimate['upper']['mean'])<last['upper']['se']
    report['budget_selection']=dict(selected=None,inner_1600_to_3200_stable=bool(stable),
        reason='No 5D production sizes frozen from small smoke cases. Requires independent review and a measured timing preflight; no acceptance gate relaxed.')
    report['peak_rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    report['elapsed_seconds']=time.perf_counter()-start
    return report


def budget_diagnostic(previous, current, previous_inner, current_inner):
    """Paired fixed-outer comparison; c/n is a diagnostic, not bias correction.

    Different inner sizes need not share inner random draws. Pairing is by outer
    path only. The two-point model cannot certify residual bias or convergence.
    """
    if current_inner != 2 * previous_inner:
        raise ValueError('budget comparison requires doubling')
    drop = _summary(previous['upper_samples'] - current['upper_samples'])
    stable = abs(drop['mean']) < min(current['upper']['se'], .02)
    return dict(previous_inner=previous_inner, inner=current_inner,
                paired_drop=drop, stable=bool(stable),
                residual_bias_c_over_n=drop['mean'],
                residual_bias_note='Signed two-point c/n extrapolation at larger n; unvalidated model, not a bound.')


def terminal_drift_decomposition(opt, result, bins=5):
    """Exact Black--Scholes decomposition at the last pre-maturity date.

    Expected positive-payoff counts use exact conditional probabilities, never
    realised counts. Cells below 30 are non-testable by the normal total-drift
    diagnostic. Earlier dates still require the independent grid-based check.
    This is a proposed replacement gate that requires independent review.
    """
    from scipy.stats import norm
    if opt.d != 1 or opt.kind != 'put' or opt.sigma <= 0:
        raise ValueError('requires a nondegenerate 1D put')
    paths = gbm_paths(opt, result['metadata']['outer'], seed=result['metadata']['outer_seed'])
    k = opt.dates - 1
    s = paths[:, k, 0]
    vol = opt.sigma * np.sqrt(opt.dt)
    d1 = (np.log(s / opt.strike) + (opt.rate-opt.dividend+opt.sigma**2/2)*opt.dt)/vol
    d2 = d1-vol
    exact = opt.discount(k)*(opt.strike*np.exp(-opt.rate*opt.dt)*norm.cdf(-d2)
                             - s*np.exp(-opt.dividend*opt.dt)*norm.cdf(-d1))
    terminal = opt.discount(opt.dates)*opt.payoff(paths[:, -1])
    inc = np.diff(result['martingale_samples'], axis=1)[:, k]
    inner_error = inc - (terminal-exact)
    rows = []
    for b, ix in enumerate(np.array_split(np.argsort(s), bins)):
        expected = float(norm.cdf(-d2[ix]).sum())
        e = _summary(inner_error[ix])
        rows.append(dict(date=k, bin=b, expected_positive_payoffs=expected,
                         observed_positive_payoffs=int((terminal[ix]>0).sum()),
                         total_drift_normal_testable=expected >= 30,
                         exact_martingale_difference=_summary((terminal-exact)[ix]),
                         inner_error=e, inner_error_within_3se=bool(abs(e['mean'])<=3*e['se'])))
    return rows


def run_correction(outer=2000):
    """First-pass 5D dual sizing diagnostic; every run is smoke evidence only.

    Run separately from the old evidence-producing function. Emit rows as they
    finish to retain partial evidence on interruption. No study CONFIG is changed.
    Coverage and tightness checks must follow at a reviewed chosen budget; an
    unstable sweep deliberately leaves sizes unset instead of guessing a budget.
    """
    from stopping import fit_neural, NeuralSettings
    import hashlib
    from pathlib import Path
    limit_threads(1)
    start = time.perf_counter()
    seed = lambda role, d, extra: seed_for(2026, role, d, extra)
    opt = paper_max_call(5)
    train = gbm_paths(opt, 10000, seed=seed('train',5,910))
    val = gbm_paths(opt, 5000, seed=seed('val',5,910))
    lsm, _ = select_lsm(opt, train, val)
    settings = NeuralSettings(max_steps=1500, lr=.001, batch=256)
    neural = fit_neural(opt, train, val, settings, seed=seed('init',5,910))
    report = dict(kind='dual sizing smoke evidence; not full-study findings', outer=outer,
                  source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in ('stopping.py','dual.py','tools/dual_validation.py')},
                  seed_protocol='base 2026; train/val/init extra 910; dual seed 911',
                  neural_settings=vars(settings), policies={},
                  criterion_changes_pending_review=[
                      'Bad-policy check is non-violation, not tightness.',
                      'Last-date drift decomposed with exact BS; expected positives <30 non-testable for normal total drift.'])
    for name, policy in [('lsm',lsm),('neural',neural)]:
        rows = []; previous = None
        for inner in (800,1600,3200,6400):
            t = time.perf_counter()
            r = policy_dual(opt, policy, outer=outer, inner=inner, seed=911)
            row = dict(inner=inner, upper=r['upper'], seconds=time.perf_counter()-t,
                       metadata=r['metadata'])
            if previous is not None:
                row['comparison'] = budget_diagnostic(previous, r, inner//2, inner)
            rows.append(row); previous = r
            print(json.dumps(dict(policy=name, **row)), flush=True)
        report['policies'][name] = rows
    eligible = all(rows[-1]['comparison']['stable'] for rows in report['policies'].values())
    # A small pilot cannot meet the requested 2000-outer sizing experiment.
    selected = 6400 if eligible and outer >= 2000 else None
    required = max(math.ceil((1.96*rows[-1]['upper']['se']*np.sqrt(outer)/.03)**2)
                   for rows in report['policies'].values())
    report['budget_selection'] = dict(selected_inner=selected, projected_outer=max(2000,required),
        frozen=False, reason='Independent review, 1D tightness, chosen-budget coverage and preflight still required.')
    o = validation_put(); boundary, _, _ = crr_boundary_policy(o)
    r = policy_dual(o, boundary, outer=3000, inner=3200, seed=71004)
    report['terminal_decomposition'] = terminal_drift_decomposition(o,r)
    report['pending'] = ['Earlier-date grid decomposition', '1D LSM and CRR tightness at chosen budget',
                         '50-repetition coverage at chosen budget', 'Independent review and criterion-change recording']
    report['elapsed_seconds'] = time.perf_counter()-start
    report['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print(json.dumps(dict(correction_report=report)), flush=True)
    return report


def grid_inner_error(opt, policy, result, h):
    """Independent zero-discounted policy-value convolution oracle in 1D.

    Return C_k - C_hat_k, not the noisy total martingale increment. Refine h
    separately: this numerical oracle is not an analytical exact expectation.
    """
    from scipy.signal import fftconvolve
    from stopping import gbm_paths, exercise_mask
    if opt.d != 1 or opt.kind != 'put' or opt.sigma <= 0:
        raise ValueError('grid oracle requires a nondegenerate 1D put')
    n = int(round(3/h))
    x = np.arange(-n, n+1)*h
    spot = opt.s0*np.exp(x)
    sd = opt.sigma*np.sqrt(opt.dt)
    drift = (opt.rate-opt.dividend-opt.sigma**2/2)*opt.dt
    kernel = np.exp(-.5*((x+drift)/sd)**2)*h/(sd*np.sqrt(2*np.pi))
    paths = gbm_paths(opt, result['metadata']['outer'], seed=result['metadata']['outer_seed'])
    if np.max(abs(np.log(paths[:, :, 0]/opt.s0))) >= 3:
        raise ValueError('outer states outside grid; extend and refine domain')
    increments = np.diff(result['martingale_samples'], axis=1)
    value = opt.payoff(spot[:, None])*opt.discount(opt.dates)
    next_estimate = opt.payoff(paths[:, -1])*opt.discount(opt.dates)
    errors = np.empty_like(increments)
    for k in range(opt.dates-1, -1, -1):
        cont = fftconvolve(value, kernel, mode='same')
        estimate = next_estimate-increments[:, k]
        errors[:, k] = np.interp(np.log(paths[:, k, 0]/opt.s0), x, cont)-estimate
        if k > 0:
            stop = exercise_mask(opt, policy, k, spot[:, None], True)
            value = np.where(stop, opt.payoff(spot[:, None])*opt.discount(k), cont)
            stop_outer = exercise_mask(opt, policy, k, paths[:, k], True)
            next_estimate = np.where(stop_outer, opt.payoff(paths[:, k])*opt.discount(k), estimate)
    return paths, errors



def run_revision_confirmation():
    """Run the registered dual-sizing revision; exclusive output creation preserves evidence."""
    import json, time, math
    from pathlib import Path
    import numpy as np
    from scipy.stats import norm
    from stopping import (validation_put, paper_max_call, gbm_paths, select_lsm,
                          Policy, exercise_mask, fit_neural, NeuralSettings, limit_threads, seed_for)
    from dual import policy_dual, nested_policy_dual, _summary
    from tools.dual_validation import crr_boundary_policy, grid_inner_error, conditional_drift
    limit_threads(1)
    out=Path('results/quick/m3-validation')
    manifest=json.loads((out/'revision-manifest.json').read_text())
    def save(name,value):
        with (out/name).open('x') as f: json.dump(value,f,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x)
    o=validation_put(); boundary,_,_=crr_boundary_policy(o)
    lsm,_=select_lsm(o,gbm_paths(o,10000,seed=71001),gbm_paths(o,5000,seed=71002))
    rows=[]; threshold=norm.isf(.01/(2*246))
    for seed in manifest['confirmation_seeds']:
        for name,p in [('crr_boundary',boundary),('lsm',lsm),('never',Policy('never'))]:
            r=policy_dual(o,p,outer=3000,inner=800,seed=seed)
            paths,e=grid_inner_error(o,p,r,.0005)
            _,ef=grid_inner_error(o,p,r,.00025)
            # e = C - C_hat. Delta M error = e_k - continue(k+1)e_(k+1).
            dif=e.copy(); diff=ef.copy()
            for k in range(o.dates-1):
                cont=~exercise_mask(o,p,k+1,paths[:,k+1],True)
                dif[:,k]-=cont*e[:,k+1]; diff[:,k]-=cont*ef[:,k+1]
            cells=[]
            for k in range(o.dates):
                groups=[np.arange(3000)] if k==0 else np.array_split(np.argsort(paths[:,k,0]),5)
                for b,ix in enumerate(groups):
                    s=_summary(diff[ix,k]); grid=float(np.max(abs(diff[ix,k]-dif[ix,k])))
                    # Independent outer units, increment error range at most 4K.
                    radius=4*o.strike*math.sqrt(math.log(2*246/.01)/(2*len(ix)))
                    horizon = (o.dates-k)*o.dt
                    d2 = (np.log(paths[ix,k,0]/o.strike)+(o.rate-o.dividend-o.sigma**2/2)*horizon)/(o.sigma*np.sqrt(horizon))
                    expected = float(norm.cdf(-d2).sum()*800)
                    cells.append(dict(date=k,bin=b,n=len(ix),error=s,grid_sensitivity=grid,
                        expected_positive_inner_payoffs=expected,normal_applicable=expected>=30,
                        normal_threshold=threshold,normal_failure=bool(abs(s['mean'])>threshold*s['se']+grid),
                        bounded_radius=radius,bounded_failure=bool(abs(s['mean'])>radius+grid)))
            item=dict(seed=seed,policy=name,cells=cells,raw_drift=conditional_drift(o,r),upper=r['upper'])
            rows.append(item);save(f'revision-confirmation-{seed}-{name}.json',item)
            print('confirmation',seed,name,'normal failures',sum(c['normal_failure'] for c in cells),'bounded failures',sum(c['bounded_failure'] for c in cells),flush=True)
    save('revision-confirmation-summary.json',dict(family_size=246,threshold=threshold,
        bounded_failures=sum(c['bounded_failure'] for r in rows for c in r['cells']),
        normal_failures=sum(c['normal_failure'] for r in rows for c in r['cells']),
        limitation='Bounded test is rigorous for exact continuation but extremely low power; grid sensitivity is not a certified error bound. Normal diagnostics are not treated as proof.'))


def run_revision_sizing(preflight_only=False):
    """Run the registered dual-sizing revision; exclusive output creation preserves evidence."""
    import json,time,math
    from pathlib import Path
    import numpy as np
    from stopping import (validation_put, paper_max_call, gbm_paths, select_lsm,
                          Policy, exercise_mask, fit_neural, NeuralSettings, limit_threads, seed_for)
    from dual import nested_policy_dual,_summary
    limit_threads(1)
    out=Path('results/quick/m3-validation')
    def save(name,r):
        with (out/name).open('x') as f:json.dump(r,f,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x)
    o=paper_max_call(5)
    train=gbm_paths(o,10000,seed=seed_for(2026,'train',5,910)); val=gbm_paths(o,5000,seed=seed_for(2026,'val',5,910))
    lsm,_=select_lsm(o,train,val)
    nn=fit_neural(o,train,val,NeuralSettings(max_steps=1500,lr=.001,batch=256),seed=seed_for(2026,'init',5,910))
    # Each inference method needs its own timing: neural overhead cannot be
    # inferred from a regression-policy pilot. Use a new evidence namespace.
    timing = {}
    for name,p in [('lsm',lsm),('neural',nn)]:
        t=time.perf_counter()
        try:
            nested_policy_dual(o,p,budgets=(800,1600),outer=20,group=800,
                               seed=95200,max_seconds=120)
        except TimeoutError:
            save('revision-sizing-preflight-v2.json',dict(passed=False,
                timed_out_policy=name,completed_policy_seconds=timing,
                reason='120-second pilot allowance exceeded; no large run'))
            raise
        timing[name]=time.perf_counter()-t
    projection=sum(timing.values())*100*8
    save('revision-sizing-preflight-v2.json',dict(policy_seconds=timing,
        projected_two_policy_12800_seconds=projection,ceiling=7200,
        passed=projection<=7200,outer=2000,seed=95201))
    if projection>7200:
        raise RuntimeError('policy-specific preflight exceeds registered 2h sizing allowance')
    if preflight_only:
        return
    report={}
    remaining_seconds=7200.
    for name,p in [('lsm',lsm),('neural',nn)]:
        saved = out / ('revision-sizing-'+name+'.json')
        if saved.exists():
            cached = json.loads(saved.read_text())
            if any(x['metadata']['seed'] != 95201 or x['metadata']['outer'] != 2000
                   for x in cached['rows']):
                raise ValueError('incompatible completed sizing evidence')
            report[name] = cached['rows']
            remaining_seconds -= cached['seconds']
            continue
        if remaining_seconds <= 0:
            raise RuntimeError('combined sizing allowance exhausted')
        t=time.perf_counter(); r=nested_policy_dual(o,p,budgets=(800,1600,3200,6400,12800),outer=2000,group=800,seed=95201,max_seconds=remaining_seconds)
        rows=[]
        for i,v in enumerate(r):
            row=dict(inner=v['metadata']['inner'],upper=v['upper'],value_gap=v['value_gap'],metadata=v['metadata'])
            if i:
                drop=_summary(r[i-1]['upper_samples']-v['upper_samples'])
                row.update(paired_drop=drop,stable=bool(abs(drop['mean'])<v['upper']['se']))
            rows.append(row)
        remaining_seconds -= time.perf_counter()-t
        report[name]=rows
        save('revision-sizing-'+name+'.json',dict(rows=rows,seconds=time.perf_counter()-t))
        print(name,rows,flush=True)
    eligible=[row['inner'] for row in report['lsm'][1:] if row['stable'] and next(x for x in report['neural'] if x['inner']==row['inner'])['stable']]
    selected=min(eligible) if eligible else None
    save('revision-sizing-selection.json',dict(selected_inner=selected,selected_outer=2000 if selected else None,
        gate='abs paired doubling change < current outer SE for both policies',
        status='candidate requiring independent review and a timing preflight' if selected else 'unresolved',
        no_bias_correction=True))


def run_revision_put_sizing():
    """Registered boundary-policy sizing and independent primal comparison."""
    import json,time
    from pathlib import Path
    import numpy as np
    from stopping import validation_put, limit_threads, gbm_paths, discounted_payoffs, reference_for
    from dual import nested_policy_dual,_summary
    from tools.dual_validation import crr_boundary_policy,independent_gap
    limit_threads(1)
    out=Path('results/quick/m3-validation')
    def save(name,r):
        with (out/name).open('x') as f:json.dump(r,f,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x)
    o=validation_put();p,_,_=crr_boundary_policy(o)
    t=time.perf_counter();nested_policy_dual(o,p,budgets=(800,1600),outer=20,group=800,seed=95203)
    projection=(time.perf_counter()-t)*150*8
    save('revision-put-preflight.json',dict(projected_seconds=projection,ceiling_seconds=1800,passed=projection<=1800))
    if projection>1800:raise RuntimeError('1D sizing preflight exceeds allowance')
    r=nested_policy_dual(o,p,budgets=(800,1600,3200,6400,12800),outer=3000,group=800,seed=95202)
    primal=discounted_payoffs(o,gbm_paths(o,500000,seed=71003),p)[0]
    rows=[]
    for i,v in enumerate(r):
        row=dict(inner=v['metadata']['inner'],upper=v['upper'],independent_gap=independent_gap(v['upper_samples'],primal),value_gap=v['value_gap'])
        if i:
            drop=_summary(r[i-1]['upper_samples']-v['upper_samples'])
            row.update(paired_drop=drop,stable=bool(abs(drop['mean'])<v['upper']['se']))
        rows.append(row)
    save('revision-put-sizing.json',dict(rows=rows,independent_primal=_summary(primal),reference=reference_for(o),
        selected_inner=next((x['inner'] for x in rows[1:] if x['stable']),None)))


if __name__ == '__main__':
    print(json.dumps(run_validation(), indent=2, default=lambda x:x.item() if isinstance(x,np.generic) else x))


def cumulative_sizing_rows(results):
    """Compare each candidate with its doubling and the largest evaluated prefix.

    Selection uses observed mean drops, not paired significance: the registered
    tolerance is the marginal outer SE. Paired CIs expose comparison uncertainty.
    No claim about bias beyond the largest evaluated prefix follows.
    """
    if len(results) < 2:
        raise ValueError('at least two evaluated budgets are required')
    for previous, current in zip(results, results[1:]):
        if current['metadata']['inner'] != 2*previous['metadata']['inner']:
            raise ValueError('budgets must be ordered evaluated doublings')
        if np.shape(previous['upper_samples']) != np.shape(current['upper_samples']):
            raise ValueError('paired outer sample shapes differ')
    rows = []
    largest = results[-1]
    for i, result in enumerate(results):
        row = dict(inner=result['metadata']['inner'], upper=result['upper'],
                   value_gap=result.get('value_gap'), metadata=result['metadata'])
        cumulative = _summary(result['upper_samples']-largest['upper_samples'])
        row['cumulative_drop'] = cumulative
        row['cumulative_gate'] = bool(abs(cumulative['mean']) < largest['upper']['se'])
        row['eligible'] = False
        if i+1 < len(results):
            following = results[i+1]
            if following['metadata']['inner'] != 2*row['inner']:
                raise ValueError('an actually evaluated doubling is required')
            drop = _summary(result['upper_samples']-following['upper_samples'])
            row['adjacent_drop'] = drop
            row['adjacent_gate'] = bool(abs(drop['mean']) < following['upper']['se'])
            row['eligible'] = row['adjacent_gate'] and row['cumulative_gate']
        rows.append(row)
    return rows


def run_cumulative_followup(preflight_only=False, output_dir="results/quick/m3-validation/sizing-96301"):
    """Fresh sizing after the cumulative-rule amendment; no old evidence replaced."""
    from pathlib import Path
    from dataclasses import asdict
    from stopping import fit_neural, NeuralSettings
    from dual import nested_policy_dual
    import hashlib
    out = Path(output_dir)
    if not out.resolve().is_relative_to((Path(__file__).resolve().parents[1]/'results/quick').resolve()):
        raise ValueError('sizing evidence must stay under results/quick')
    out.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        with (out/('followup-'+name+'.json')).open('x') as f:
            json.dump(value, f, indent=2)
    manifest = dict(budgets=[800,1600,3200,6400,12800], outer=2000,
        outer_blocks=4, block_outer=500, block_seed_rule='SeedSequence([96301, policy_index, block]); same seed for every inner prefix',
        group=800, pilot_outer=20, pilot_seed=96300, evaluation_seed=96301,
        policy_seed=910, target_outer_ci_halfwidth=.03,
        rule='smallest common candidate: absolute adjacent drop < doubling outer SE AND absolute cumulative drop < largest outer SE; candidate must have evaluated doubling; both policies must meet precision',
        precision_rule='1.96 times marginal upper SE <= .03 for both policies',
        ceiling_seconds=7200, preflight_margin=1.5,
        neural=asdict(NeuralSettings(max_steps=1500,lr=.001,batch=256)),
        limitations='Largest prefix is not an oracle. No extrapolation. Fresh evaluation seed after adaptation. Quick validation only.')
    manifest['source_sha256'] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                 for name in ('stopping.py','dual.py','tools/dual_validation.py')}
    manifest_path = out/'followup-manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('saved manifest differs; preserve this attempt and use a new namespace')
    else:
        save('manifest',manifest)
    if (out/'followup-outcome.json').exists():
        return json.loads((out/'followup-outcome.json').read_text())
    limit_threads(1)
    start=time.perf_counter()
    o=paper_max_call(5)
    train=gbm_paths(o,10000,seed=seed_for(2026,'train',5,910))
    val=gbm_paths(o,5000,seed=seed_for(2026,'val',5,910))
    lsm,_=select_lsm(o,train,val)
    neural=fit_neural(o,train,val,NeuralSettings(**manifest['neural']),seed=seed_for(2026,'init',5,910))
    preflight_path = out/'followup-preflight.json'
    if not preflight_path.exists():
        timings={}
        for name,p in [('lsm',lsm),('neural',neural)]:
            t=time.perf_counter()
            try:
                nested_policy_dual(o,p,budgets=(800,1600),outer=manifest['pilot_outer'],group=manifest['group'],seed=manifest['pilot_seed'],max_seconds=120)
            except (TimeoutError,KeyboardInterrupt) as exc:
                save('preflight-failure-'+str(time.time_ns()),dict(policy=name,error=type(exc).__name__,completed_timings=timings,seconds=time.perf_counter()-t))
                raise
            timings[name]=time.perf_counter()-t
            print('uncontended pilot',name,timings[name],flush=True)
        projection=sum(timings.values())*(manifest['outer']/manifest['pilot_outer'])*(max(manifest['budgets'])/1600)*manifest['preflight_margin']
        remaining=manifest['ceiling_seconds']-(time.perf_counter()-start)
        passed=projection<=remaining
        save('preflight',dict(policy_seconds=timings,projected_sweep_seconds=projection,
             remaining_seconds=remaining,passed=passed,concurrent_experiments=False,
             setup_and_preflight_seconds=time.perf_counter()-start))
        if not passed:
            save('outcome',dict(status='unresolved: preflight exceeds allowance',main_study_budget=dict(inner=None,outer=None)))
            return
    if preflight_only:
        return json.loads(preflight_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    if not preflight['passed']:
        raise ValueError('preflight did not pass')
    return checkpoint_sizing(o, [('lsm',lsm),('neural',neural)], manifest, out,
                             started=start)


def checkpoint_sizing(option, policies, manifest, out, *, started=None, work_seconds=600):
    """Checkpoint four independent outer blocks, each with all nested prefixes.

    The block seed deliberately excludes inner size: changing it per prefix
    would destroy the paired comparison. Each finished block contains every
    (policy, inner size) cell for that outer block, saved in one atomic file.
    """
    import signal
    import os
    from pathlib import Path
    from dual import nested_policy_dual
    out = Path(out)
    started = time.perf_counter() if started is None else started
    prior_seconds = sum(json.loads(p.read_text())['seconds'] for p in out.glob('invocation-*.json'))
    preflight_record = out/'followup-preflight.json'
    if preflight_record.exists():
        prior_seconds += json.loads(preflight_record.read_text()).get('setup_and_preflight_seconds',0)
    interrupted = []
    def on_term(signum, frame):
        interrupted.append(signum)
        raise KeyboardInterrupt('SIGTERM')
    old_handler = signal.signal(signal.SIGTERM, on_term)
    def atomic_json(path, value):
        temp = path.with_suffix('.pending')
        with temp.open('w') as f:
            json.dump(value, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    def remaining():
        return sum(not (out/f'{name}-block-{b}.npz').exists()
                   for name,_ in policies for b in range(4))
    try:
        for policy_index,(name,policy) in enumerate(policies):
            for block in range(4):
                path = out/f'{name}-block-{block}.npz'
                if path.exists():
                    continue
                if time.perf_counter()-started >= min(work_seconds, manifest.get('ceiling_seconds',7200)-prior_seconds):
                    return dict(status='incomplete', blocks_remaining=remaining())
                seed = int(np.random.SeedSequence(
                    [manifest['evaluation_seed'], policy_index, block]).generate_state(1)[0])
                t = time.perf_counter()
                try:
                    results = nested_policy_dual(option,policy,
                        budgets=tuple(manifest['budgets']),outer=manifest['outer']//4,
                        group=manifest['group'],seed=seed,max_seconds=max(.001,min(580, work_seconds-(time.perf_counter()-started), manifest.get('ceiling_seconds',7200)-prior_seconds-(time.perf_counter()-started))))
                    temp = path.with_suffix('.pending')
                    with temp.open('wb') as f:
                        np.savez_compressed(f,upper=np.array([r['upper_samples'] for r in results]),
                            seed=seed,seconds=time.perf_counter()-t)
                        f.flush(); os.fsync(f.fileno())
                    os.replace(temp,path)
                    print(name, 'block', block, 'saved; remaining', remaining(), flush=True)
                except (KeyboardInterrupt,TimeoutError) as exc:
                    atomic_json(out/f'interruption-{time.time_ns()}.json',
                        dict(policy=name,block=block,seed=seed,error=type(exc).__name__,
                             signal=interrupted,seconds=time.perf_counter()-t,
                             blocks_remaining=remaining(),status='incomplete'))
                    return dict(status='incomplete',blocks_remaining=remaining())
        reports = {}
        for name,_ in policies:
            blocks=[]
            for block in range(4):
                with np.load(out/f'{name}-block-{block}.npz') as data:
                    blocks.append(data['upper'])
            samples=np.concatenate(blocks,axis=1)
            results=[dict(metadata=dict(inner=n,outer=manifest['outer'],blocks=4),
                          upper_samples=x,upper=_summary(x))
                     for n,x in zip(manifest['budgets'],samples)]
            reports[name]=cumulative_sizing_rows(results)
            atomic_json(out/f'followup-{name}.json',dict(rows=reports[name]))
        eligible=[r['inner'] for r in reports['lsm'] if r['eligible'] and
                  1.96*r['upper']['se']<=.03 and any(q['inner']==r['inner'] and
                  q['eligible'] and 1.96*q['upper']['se']<=.03 for q in reports['neural'])]
        selected=min(eligible) if eligible else None
        outcome=dict(status='candidate awaiting independent review' if selected else
                     'unresolved: convergence or precision gate',blocks_remaining=0,
                     main_study_budget=dict(inner=selected,outer=manifest['outer'] if selected else None),
                     residual_bias='Unmeasured beyond largest evaluated prefix; no zero-bias claim.')
        atomic_json(out/'followup-outcome.json',outcome)
        return outcome
    finally:
        atomic_json(out/f'invocation-{time.time_ns()}.json',
                    dict(seconds=time.perf_counter()-started,blocks_remaining=remaining()))
        signal.signal(signal.SIGTERM,old_handler)
