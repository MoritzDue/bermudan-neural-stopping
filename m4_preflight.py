"""Controller-owned, quick-only M4 timing and design-freezing tools.

Run once in a new quick namespace. Timings are smoke measurements, not values
for scientific interpretation. The controller supplies notebook runtimes and
the confirmed budget for the combined design-freezing step.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from stopping import (CONFIG, NeuralSettings, calibration_provenance,
                      discounted_payoffs, fit_neural, gbm_paths,
                      guard_diagnostic, limit_threads, paper_max_call,
                      select_lsm, seed_for)
from dual import nested_policy_dual


def budget_ledger(*, notebook_seconds, confirmed_remaining_seconds,
                  snapshot_timestamp, remaining_m4_seconds, cumulative_costs):
    """Evaluate the approved L0–L5 ladder from a controller balance snapshot.

    Costs must already include measured timing × 1.2 and all unit overhead.
    Completed M4 expenditure is reflected in the balance, never charged twice.
    This calculation does not write a freeze or claim budget confirmation.
    """
    values = (notebook_seconds, confirmed_remaining_seconds,
              snapshot_timestamp, remaining_m4_seconds, *cumulative_costs)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('all durations and timestamp must be finite and nonnegative')
    costs = list(cumulative_costs)
    if len(costs) != 6 or any(b < a for a, b in zip(costs, costs[1:])):
        raise ValueError('supply six nondecreasing cumulative L0–L5 projections')
    reserve = notebook_seconds + 120
    levels = []
    selected = None
    lower_fit = True
    for level, cost in enumerate(costs):
        correction = max(300., 0.1 * cost)
        allowance = min(28800., 0.8 * (confirmed_remaining_seconds -
                        remaining_m4_seconds - reserve - correction))
        fits = lower_fit and cost <= allowance
        levels.append(dict(level=level, D=cost, A=allowance,
                           R_corr=correction, fits=fits))
        if fits:
            selected = level
        lower_fit = fits
    return dict(B=confirmed_remaining_seconds, snapshot_timestamp=snapshot_timestamp,
                E_M4_remaining=remaining_m4_seconds, Q=notebook_seconds,
                R_M6=reserve, levels=levels, selected_level=selected,
                escalate=selected is None)


def quick_destination(value):
    path = Path(value).resolve()
    root = Path(__file__).resolve().parent / 'results' / 'quick'
    if not path.is_relative_to(root) or path == root:
        raise ValueError('use a new child namespace under results/quick')
    return path


def dual_projection(seconds, outer):
    """Linear sizing only; includes a full coupled doubling subset per policy."""
    if seconds <= 0 or outer < 2:
        raise ValueError('positive timing and at least two outer samples required')
    return dict(base_2000_s=seconds * 2000 / outer,
                doubling_500_s=seconds * 500 / outer * 2,
                assumption='linear outer/inner scaling; group=200; needs review')


def production_projection(seconds, outer, safety=1.2, *, base_seconds=None):
    """Project a coupled (3200,6400) probe, using resumable 100-path blocks.

    The 3200 base costs half the measured coupled run. The doubling subset
    must recompute both prefixes, so its cost uses the whole measurement.
    Fitting, setup and serialization are additional measured components.
    """
    if (not math.isfinite(seconds) or seconds <= 0 or outer < 100 or
            not math.isfinite(safety) or safety < 1.2):
        raise ValueError('production sizing needs >=100 paths and safety >=1.2')
    if base_seconds is not None and (not math.isfinite(base_seconds) or base_seconds <= 0):
        raise ValueError('base timing must be finite and positive')
    per_outer = seconds / outer
    base_per_outer = per_outer/2 if base_seconds is None else base_seconds/outer
    return dict(base_2000_s=safety*base_per_outer*2000,
                doubling_500_s=safety*per_outer*500,
                base_block_100_s=safety*base_per_outer*100,
                doubling_block_100_s=safety*per_outer*100,
                safety=safety, base_measured=base_seconds is not None,
                assumption=('linear outer scaling; separately measured base and coupled runs; overhead separate'
                            if base_seconds is not None else
                            'linear inner scaling from coupled probe; overhead separate'))


def production_probe(destination):
    """Controller-owned quick sizing; no main test paths or design selection."""
    out = quick_destination(destination)
    out.mkdir(parents=True, exist_ok=False)
    limit_threads(1)
    report = dict(kind='production-size quick timing; NOT research outcomes',
                  development_seed=170, dual_seed=97170, outer=100,
                  budgets=[3200, 6400], group=200,
                  provenance=calibration_provenance('quick'), components=[])
    start = time.perf_counter()

    def timed(name, operation, **keys):
        t = time.perf_counter()
        value = operation()
        report['components'].append(dict(component=name, seconds=time.perf_counter()-t, **keys))
        report['elapsed_s'] = time.perf_counter()-start
        (out/'production_timings.json').write_text(json.dumps(report, indent=2))
        return value

    opt = paper_max_call(5)
    base = CONFIG['full']['base_seed']
    train, val = timed('setup', lambda: (
        gbm_paths(opt,50000,seed_for(base,'train',5,170)),
        gbm_paths(opt,5000,seed_for(base,'val',5,170))))
    for method in ('lsm','nn'):
        policy = timed('fit', lambda: select_lsm(opt,train,val)[0] if method=='lsm'
            else fit_neural(opt,train,val,NeuralSettings(max_steps=1500,lr=1e-3,batch=512),
                            seed=seed_for(base,'init',5,170)), method=method)
        base_result = timed('base_nested', lambda: nested_policy_dual(
            opt,policy,budgets=(3200,),outer=100,seed=97170,group=200,
            max_seconds=120), method=method)
        base_seconds = report['components'][-1]['seconds']
        result = timed('coupled_nested', lambda: nested_policy_dual(
            opt,policy,budgets=(3200,6400),outer=100,seed=97170,group=200,
            max_seconds=240), method=method)
        seconds = report['components'][-1]['seconds']
        # Coupled prefix must reproduce the standalone 3200 calculation.
        import numpy as np
        np.testing.assert_array_equal(base_result[0]['upper_samples'], result[0]['upper_samples'])
        np.testing.assert_array_equal(base_result[0]['martingale_samples'], result[0]['martingale_samples'])
        report.setdefault('prefix_identical',{})[method] = True
        report.setdefault('projections',{})[method] = production_projection(
            seconds,100,base_seconds=base_seconds)
        def serialize():
            data=json.dumps(result,default=lambda x:x.tolist()).encode()
            (out/(method+'-coupled.json')).write_bytes(data)
            return hashlib.sha256(data).hexdigest()
        digest=timed('serialization',serialize,method=method)
        report.setdefault('artifact_sha256',{})[method]=digest
    report['elapsed_s']=time.perf_counter()-start
    (out/'production_timings.json').write_text(json.dumps(report,indent=2))


def checkpoint_dual_chunk(path, opt, policy, *, contract):
    """Persist one deterministic chunk; reject incompatible resume evidence.

    The controller must supply frozen code/config/policy hashes and indices.
    This helper does not certify these hashes or authorize a full execution.
    Samples, rather than chunk means, are the aggregation source of truth.
    """
    import numpy as np
    path = Path(path)
    required = {'code_hash', 'config_hash', 'policy_hash', 'outer',
                'outer_indices', 'seed', 'budgets', 'group'}
    if set(contract) != required or any(not contract[k] for k in
                                      ('code_hash', 'config_hash', 'policy_hash')):
        raise ValueError('complete frozen chunk contract required')
    # Schema 2 makes old raw-payoff-only chunks incompatible, without deleting them.
    encoded = json.dumps(dict(schema=2, **contract), sort_keys=True, separators=(',', ':'), allow_nan=False)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if saved['contract'].item() != encoded:
                raise ValueError('chunk provenance differs; preserve existing evidence')
            expected = (len(contract['budgets']), len(contract['outer_indices']))
            arrays = tuple(saved[key].copy() for key in ('upper', 'initial_value', 'realised_payoff'))
            if any(a.shape != expected or not np.isfinite(a).all() for a in arrays):
                raise ValueError('invalid saved sample shape or nonfinite samples')
            return arrays
    results = nested_policy_dual(opt, policy, outer=contract['outer'],
        outer_indices=contract['outer_indices'], seed=contract['seed'],
        budgets=tuple(contract['budgets']), group=contract['group'])
    upper = np.stack([r['upper_samples'] for r in results])
    initial = np.stack([r['initial_value_samples'] for r in results])
    realised = np.stack([r['lower_samples'] for r in results])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('wb') as handle:
        np.savez(handle, contract=encoded, upper=upper, initial_value=initial, realised_payoff=realised)
    temporary.replace(path)
    return upper, initial, realised


def run(destination):
    out = quick_destination(destination)
    out.mkdir(parents=True, exist_ok=False)
    limit_threads(1)
    started = time.perf_counter()
    report = dict(kind='M4 smoke timing only; design NOT frozen',
                  provenance=calibration_provenance('quick'),
                  settings=asdict(NeuralSettings(max_steps=1500, lr=1e-3, batch=512)),
                  components=[], guard=[], inventory=[])
    for path in sorted(Path('results').rglob('*')):
        if path.is_file() and not path.resolve().is_relative_to(out):
            report['inventory'].append(dict(path=str(path), bytes=path.stat().st_size))

    def save():
        report['elapsed_s'] = time.perf_counter() - started
        temporary = out / 'timings.tmp'
        temporary.write_text(json.dumps(report, indent=2, default=float))
        temporary.replace(out / 'raw_timings.json')

    def timed(name, operation, **key):
        start = time.perf_counter()
        result = operation()
        report['components'].append(dict(component=name, seconds=time.perf_counter()-start, **key))
        save()
        return result

    opt = paper_max_call(5)
    base = CONFIG['full']['base_seed']
    test = timed('test_setup', lambda: gbm_paths(opt, 200000, seed=seed_for(base, 'check', 5, 170)))
    fresh = gbm_paths(opt, 20000, seed=seed_for(base, 'check', 5, 171))
    for budget in (1000, 50000):
        train, val = timed('train_validation_setup', lambda: (
            gbm_paths(opt, budget, seed=seed_for(base, 'train', 5, 170)),
            gbm_paths(opt, 5000, seed=seed_for(base, 'val', 5, 170))), budget=budget)
        for method in ('lsm', 'nn'):
            policy = timed('fit', lambda: select_lsm(opt, train, val)[0] if method == 'lsm'
                else fit_neural(opt, train, val, NeuralSettings(**report['settings']),
                                seed=seed_for(base, 'init', 5, 170)), budget=budget, method=method)
            timed('test_evaluation', lambda: [discounted_payoffs(opt, test, policy, g)
                                           for g in (True, False)], budget=budget, method=method)
            if budget == 1000:
                report['guard'].append(dict(method=method, development_seed=170,
                    fresh_seed=seed_for(base, 'check', 5, 171),
                    rows=timed('guard_diagnostic', lambda: guard_diagnostic(opt, fresh, policy), method=method)))
            if budget == 50000:
                # Same group-prefix implementation intended for the doubling study.
                result = timed('dual_nested', lambda: nested_policy_dual(
                    opt, policy, budgets=(3200,), outer=2, seed=97170, group=200,
                    max_seconds=90), method=method)
                seconds = report['components'][-1]['seconds']
                report.setdefault('dual_projection', {})[method] = dual_projection(seconds, 2)
                def serialize():
                    data = json.dumps(result, default=lambda x: x.tolist()).encode()
                    (out / (method + '-dual-smoke.json')).write_bytes(data)
                    return hashlib.sha256(data).hexdigest()
                timed('dual_serialization', serialize, method=method)
    save()


def resume_pilot(destination):
    """Production-sized development pilot, with a controlled checkpoint interruption.

    This tests restart at the saved-chunk boundary, not SIGKILL during a write.
    Only independent development paths are used; numerical outputs are smoke
    evidence and must not be used to choose the scientific design.
    """
    import numpy as np
    from stopping import (ResultStore, run_unit, reference_for, code_hash,
                          stable_hash)
    out = quick_destination(destination)
    out.mkdir(parents=True, exist_ok=False)
    limit_threads(1)
    opt = paper_max_call(5)
    base = CONFIG['full']['base_seed']
    settings = NeuralSettings(max_steps=1500, lr=1e-3, batch=512)
    provenance = calibration_provenance('quick')
    report = dict(kind='development restart pilot; NOT research outcomes',
                  provenance=provenance, interruption='controlled exception after durable checkpoint',
                  limitation='does not test process termination during serialization')

    def save():
        (out/'resume_evidence.json').write_text(json.dumps(report, indent=2))

    def digest(a):
        a = np.ascontiguousarray(a, dtype=np.float64)
        return hashlib.sha256(a.tobytes()).hexdigest()

    train = gbm_paths(opt, 50000, seed_for(base, 'train', 5, 170))
    val = gbm_paths(opt, 5000, seed_for(base, 'val', 5, 170))
    policy = select_lsm(opt, train, val)[0]
    # Hash actual fitted rules, not merely the policy name.
    fingerprint = dict(contract=asdict(opt), guard=policy.guard,
                       degree=policy.info['degree'], ridge=policy.info['ridge'], rules={})
    for k, rule in sorted(policy.rules.items()):
        reg = rule.__defaults__[0] if rule is not None else None
        fingerprint['rules'][k] = None if reg is None else {
            name: getattr(reg, name).tolist() for name in ('mu', 'sd', 'coef')}
    policy_hash = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    contract = dict(code_hash=code_hash(), config_hash=stable_hash(CONFIG),
                    policy_hash=policy_hash, outer=2000, outer_indices=list(range(100)),
                    seed=97170, budgets=[3200, 6400], group=200)
    chunk = out/'resumed-chunk.npz'
    class CheckpointInterruption(Exception):
        pass
    try:
        checkpoint_dual_chunk(chunk, opt, policy, contract=contract)
        raise CheckpointInterruption('stop after first saved 100-path chunk')
    except CheckpointInterruption:
        report['interruption_observed'] = True
        save()
    # A restart must load the saved chunk without calling the estimator.
    from unittest.mock import patch
    with patch('m4_preflight.nested_policy_dual', side_effect=AssertionError('recomputed saved chunk')):
        resumed_u, resumed_l, resumed_z = checkpoint_dual_chunk(chunk, opt, policy, contract=contract)
    uninterrupted = nested_policy_dual(opt, policy, outer=2000,
        outer_indices=contract['outer_indices'], seed=97170, budgets=(3200,6400), group=200)
    full_u = np.stack([r['upper_samples'] for r in uninterrupted])
    full_l = np.stack([r['initial_value_samples'] for r in uninterrupted])
    full_z = np.stack([r['lower_samples'] for r in uninterrupted])
    np.savez(out/'uninterrupted-chunk.npz', upper=full_u, initial_value=full_l, realised_payoff=full_z)
    report['dual'] = {}
    for name, a, b in [('U', resumed_u, full_u), ('L', resumed_l, full_l),
                       ('realised_payoff', resumed_z, full_z),
                       ('raw_gap', resumed_u-resumed_z, full_u-full_z),
                       ('delta', resumed_u-resumed_l, full_u-full_l)]:
        np.testing.assert_array_equal(a, b)
        report['dual'][name] = dict(resumed_sha256=digest(a), uninterrupted_sha256=digest(b),
                                   identical=True, shape=list(a.shape))
    save()

    test = gbm_paths(opt, 200000, seed_for(base, 'check', 5, 172))
    info = dict(frozen=dict(provenance=provenance, pilot=True, settings=asdict(settings)))
    key = (opt.name, 1000, 170)
    store = ResultStore(out/'study', info, 'quick')
    try:
        unit = run_unit(opt, 1000, 170, ['sorted+payoff'], settings, 5000,
                        test, reference_for(opt), base)
        store.append(unit, key=key)
        raise CheckpointInterruption('stop after saved study unit')
    except CheckpointInterruption:
        pass
    restarted = ResultStore(out/'study', info, 'quick')
    assert restarted.done() == {key}
    fresh = run_unit(opt, 1000, 170, ['sorted+payoff'], settings, 5000,
                     test, reference_for(opt), base)
    # Wall times are inherently different; retain both raw results, compare
    # every other field including policy diagnostics and paired contrasts.
    def numerical(value):
        if isinstance(value, dict):
            return {k: numerical(v) for k,v in value.items() if not k.endswith('_s')}
        if isinstance(value, list):
            return [numerical(v) for v in value]
        return value
    saved = restarted.units()[0]
    a = numerical({k:saved[k] for k in ResultStore.TABLES})
    b = numerical(fresh)
    encoded_a = json.dumps(a, sort_keys=True, allow_nan=True)
    encoded_b = json.dumps(b, sort_keys=True, allow_nan=True)
    assert encoded_a == encoded_b, 'study numerical outputs differ'
    (out/'uninterrupted-study.json').write_text(json.dumps(fresh, indent=2))
    report['study'] = dict(identical=True, excluded_fields='keys ending _s (wall-clock times)',
        resumed_sha256=hashlib.sha256(encoded_a.encode()).hexdigest(),
        uninterrupted_sha256=hashlib.sha256(encoded_b.encode()).hexdigest(),
        budget=1000, validation=5000, evaluation=200000, development_seed=170)
    save()


def validation_probe(destination):
    """Measure every retained nb01 full section, with exact sizes/assertions.

    Shared plain functions are called by notebook 01 too. This is a timing
    job in a new quick namespace, not a notebook or full research result.
    Each section is a separate resumable P1 unit; put and dual units rebuild
    their reference rather than relying on an earlier process's memory.
    """
    import matplotlib
    matplotlib.use('Agg')
    out = quick_destination(destination)
    out.mkdir(parents=True, exist_ok=False)
    limit_threads(1)
    report = dict(kind='exact retained P1 path timing; NOT research evidence',
                  components=[], provenance=calibration_provenance('quick'),
                  scope='5D convergence at 1k/10k/50k, 3 seeds; paper-regime implementation check not tested')
    def timed(name, operation):
        t = time.perf_counter()
        value = operation()
        report['components'].append(dict(component=name, seconds=time.perf_counter()-t))
        (out/'validation_timings.json').write_text(json.dumps(report, indent=2))
        return value
    # Fresh-process import measurement includes stopping, dual and all notebook
    # plotting/data imports, even though this parent already imported stopping.
    import subprocess, sys
    result = subprocess.run([sys.executable, '-c',
        'import time; t=time.perf_counter(); import validation_sections; import dual; '
        'print(time.perf_counter()-t)'], check=True, capture_output=True, text=True)
    import_s = float(result.stdout.strip().splitlines()[-1])
    from validation_sections import (validation_checks, validation_put_fit,
                                    validation_convergence, validation_dual)
    from stopping import validation_put, reference_for
    cfg = CONFIG['full']['validation']
    base = CONFIG['full']['base_seed']
    timed('checks', validation_checks)
    def put_section():
        opt = validation_put()
        return validation_put_fit('full', cfg, base, out, 500, opt, reference_for(opt))
    timed('put', put_section)
    timed('convergence', lambda: validation_convergence('full', cfg, base, out, 500))
    def dual_section():
        opt = validation_put()
        return validation_dual('full', cfg, base, out, 500, opt, reference_for(opt))
    timed('dual', dual_section)
    report['import_seconds'] = import_s
    report['units'] = [dict(id='P1-'+r['component'], section=r['component'],
        projected_s=1.2*(r['seconds']+import_s+10.)) for r in report['components']]
    report['projection'] = dict(safety=1.2,
        seconds=sum(u['projected_s'] for u in report['units']),
        limitation='exact full section measurements; fresh-process imports and 10 s restart overhead per section; runtime variability unmeasured')
    (out/'validation_timings.json').write_text(json.dumps(report, indent=2))



LADDER_VARIANTS = ('sorted+payoff', 'raw+payoff', 'sorted', 'raw')

def ladder_manifest(unit_cost, dual_cost, validation_seconds):
    """Explicit cumulative manifest; costs include safety and restart overhead."""
    units = ([dict(id='P1', level=0, kind='validation', projected_s=validation_seconds)]
             if isinstance(validation_seconds, (int, float)) else
             [dict(u, level=0, kind='validation') for u in validation_seconds])
    def study(d, b, seed, variant, level):
        units.append(dict(id=f'study-{d}-{b}-{seed}-{variant}', kind='study',
            level=level, d=d, budget=b, seed=seed, variant=variant,
            produces=['lsm', variant],
            projected_s=unit_cost[(d,b,variant)]))
    for seed in range(10):
        for b in (1000,50000):
            study(5,b,seed,'sorted+payoff',0 if seed<5 else 1)
        study(5,10000,seed,'sorted+payoff',2)
        for b in (1000,50000):
            for v in ('raw','raw+payoff','sorted'):
                study(5,b,seed,v,3)
        study(2,10000,seed,'sorted+payoff',4)
    policies=[]
    for b in (1000,10000,50000):
        for seed in range(3):
            for method in ('lsm','sorted+payoff'):
                primary=b==50000 and seed==0
                policies.append((5,b,seed,method,0 if primary else 5,primary))
    for b in (1000,50000):
        for v in ('raw','raw+payoff','sorted'):
            policies.append((5,b,0,v,5,False))
    for method in ('lsm','sorted+payoff'):
        policies.append((2,10000,0,method,5,False))
    for d,b,seed,method,level,primary in policies:
        for coupled,count in ((False,20),(True,5 if primary else 0)):
            for chunk in range(count):
                units.append(dict(id=f'dual-{d}-{b}-{seed}-{method}-{coupled}-{chunk}',
                    kind='dual',level=level,d=d,budget=b,seed=seed,method=method,
                    outer=2000,outer_indices=list(range(chunk*100,(chunk+1)*100)),
                    inner=[3200,6400] if coupled else [3200],group=200,
                    rng_seed=seed_for(97170,'check',d,seed),
                    rng_contract='path-indexed nested_policy_dual per (d, seed); common random numbers across LSM and NN deliberately reduce comparison variance; same seed for base and coupled',
                    projected_s=dual_cost[(d,b,method,coupled)]))
    costs=[sum(u['projected_s'] for u in units if u['level']<=l) for l in range(6)]
    return units,costs


def combined_freeze(destination, *, balance, timestamp, prior_seconds, resume_evidence=None, notebook_seconds=198.18):
    """One controller job: development timings, guard diagnostic, immutable freeze.

    No full outcomes are generated. prior_seconds is the controller's measured
    expenditure after the supplied snapshot, excluding this job itself.
    """
    import numpy as np
    from datetime import datetime, timezone
    from stopping import VARIANTS, with_variant, stable_hash
    assert all(v in VARIANTS for v in LADDER_VARIANTS), 'invalid ladder variant'
    for value in (balance,timestamp,prior_seconds):
        if not math.isfinite(value) or value < 0:
            raise ValueError('finite nonnegative budget inputs required')
    started=time.perf_counter()
    out=quick_destination(destination)
    out.mkdir(parents=True,exist_ok=False)
    limit_threads(1)
    provenance=calibration_provenance('quick')
    report=dict(provenance=provenance,components=[],guard=[],
        snapshot_utc=datetime.fromtimestamp(timestamp, timezone.utc).isoformat(), prior_post_snapshot_seconds=prior_seconds)
    preserved=['validation_lsm_convergence.csv','validation_lsm_convergence.png',
               'validation_paper_check.json','validation_put.csv']
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    before={n:sha(Path('results/quick')/n) for n in preserved}
    inventory=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
               for p in sorted(Path('results').rglob('*')) if p.is_file()]
    resume_pilot(out/'resume-pilot')
    resume_path=out/'resume-pilot/resume_evidence.json'
    resume=json.loads(resume_path.read_text())
    if resume['provenance'] != provenance:
        raise ValueError('resume pilot provenance differs from freeze')
    for entry in [resume['study'],*resume['dual'].values()]:
        if not entry['identical'] or entry['resumed_sha256']!=entry['uninterrupted_sha256']:
            raise ValueError('resume evidence fails identity')
    report['resume_evidence']=dict(path=str(resume_path),sha256=sha(resume_path),
        original_provenance=resume['provenance'],
        matches_frozen_code=resume['provenance']==provenance,
        limitation='pilot rerun in this job at frozen code hash; no process termination during serialization')
    def save():
        (out/'raw_timings.json').write_text(json.dumps(report,indent=2,default=float))
    def timed(name,operation,**keys):
        t=time.perf_counter(); value=operation()
        seconds=time.perf_counter()-t
        report['components'].append(dict(component=name,seconds=seconds,**keys));save()
        return value,seconds
    settings=NeuralSettings(max_steps=1500,lr=1e-3,batch=512)
    base=CONFIG['full']['base_seed']
    unit_cost={};fit_cost={};nested={};serial={};max_setup=0.
    for d,b in ((5,1000),(5,10000),(5,50000),(2,10000)):
        opt=paper_max_call(d)
        (train,val,test),setup=timed('setup',lambda:(
            gbm_paths(opt,b,seed_for(base,'train',d,170)),
            gbm_paths(opt,5000,seed_for(base,'val',d,170)),
            gbm_paths(opt,200000,seed_for(base,'check',d,170))),d=d,budget=b)
        max_setup=max(max_setup,setup)
        lsm,ls=timed('fit',lambda:select_lsm(opt,train,val)[0],d=d,budget=b,method='lsm')
        _,le=timed('evaluation',lambda:[discounted_payoffs(opt,test,lsm,g) for g in (True,False)],d=d,budget=b,method='lsm')
        fit_cost[d,b,'lsm']=ls
        variants=('sorted+payoff','raw','raw+payoff','sorted') if d==5 and b!=10000 else ('sorted+payoff',)
        for v in variants:
            pol,ns=timed('fit',lambda:fit_neural(opt,train,val,with_variant(settings,v),seed=seed_for(base,'init',d,170)),d=d,budget=b,method=v)
            _,ne=timed('evaluation',lambda:[discounted_payoffs(opt,test,pol,g) for g in (True,False)],d=d,budget=b,method=v)
            fit_cost[d,b,v]=ns
            # Repeating setup and LSM for every ablation unit is conservative.
            unit_cost[d,b,v]=1.2*(setup+ls+le+ns+ne+10.)
            if d==5 and b==1000 and v=='sorted+payoff':
                fresh=gbm_paths(opt,20000,seed_for(base,'check',d,171))
                for name,p in [('lsm',lsm),('nn',pol)]:
                    rows,_=timed('guard',lambda:guard_diagnostic(opt,fresh,p),method=name)
                    report['guard'].append(dict(method=name,development_seed=170,fresh_seed=seed_for(base,'check',d,171),rows=rows))
            if d==5 and b==50000 and v=='sorted+payoff':
                for name,p in [('lsm',lsm),('sorted+payoff',pol)]:
                    results=[]
                    for coupled in (False,True):
                        result,sec=timed('nested',lambda:nested_policy_dual(opt,p,
                            budgets=(3200,6400) if coupled else (3200,),outer=2000,
                            outer_indices=list(range(100)),seed=97170,group=200),method=name,coupled=coupled)
                        nested[name,coupled]=sec;results.append(result)
                        def serialize():
                            arrays={f'{i}_{k}':r[k] for i,r in enumerate(result) for k in
                                ('upper_samples','initial_value_samples','lower_samples','value_gap_samples')}
                            np.savez(out/f'{name}-{coupled}.npz',**arrays)
                        _,serial[name,coupled]=timed('serialization',serialize,method=name,coupled=coupled)
                    for k in ('upper_samples','initial_value_samples','lower_samples','value_gap_samples'):
                        np.testing.assert_array_equal(results[0][0][k],results[1][0][k])
                report['coupled_prefix_identical']=True
    # Re-measure P1 components at this exact code hash, rather than mixing runs.
    validation_probe(out/'validation')
    validation=json.loads((out/'validation/validation_timings.json').read_text())
    p1=validation['units']
    # Every chunk refits its policy. Unmeasured 2D/ablation nesting is bounded
    # by the slower measured 5D policy, a sizing assumption, not a benchmark.
    dual_cost={}
    for d,b,v in fit_cost:
        for coupled in (False,True):
            name='lsm' if v=='lsm' else 'sorted+payoff'
            ns=max(nested[m,coupled] for m in ('lsm','sorted+payoff'))
            ss=max(serial[m,coupled] for m in ('lsm','sorted+payoff'))
            dual_cost[d,b,v,coupled]=1.2*(max_setup+fit_cost[d,b,v]+ns+ss+10.)
    units,costs=ladder_manifest(unit_cost,dual_cost,p1)
    elapsed=time.perf_counter()-started
    ledger=budget_ledger(notebook_seconds=notebook_seconds,confirmed_remaining_seconds=balance,
        snapshot_timestamp=timestamp,remaining_m4_seconds=prior_seconds+elapsed+300.,cumulative_costs=costs)
    level=ledger['selected_level']
    selected=[] if level is None else [u for u in units if u['level']<=level]
    oversized=[u['id'] for u in selected if u['projected_s']>720]
    if provenance!=calibration_provenance('quick'):
        raise ValueError('source/config/environment changed during probe')
    if before!={n:sha(Path('results/quick')/n) for n in preserved}:
        raise ValueError('preserved quick artifacts changed')
    report.update(ledger=ledger,elapsed_s=elapsed,oversized_units=oversized,
        supersedes=dict(namespace='m4-combined-freeze-rev4-final2', reason='P1 omitted full convergence and paper-check costs; exact shared section sizing and authorized full scope reduction before any full outcome; resume pilot rerun at current hash'),
        conservative_extra_m4_seconds=300.,
        budget_accounting='E_M4,rem includes prior expenditure + current elapsed + an extra conservative 300 s; this addition is separate from, and not charged again as, R_corr in the ledger',
        projection_assumptions=['1.2 safety on all measured components; 10 seconds extra restart/serialization overhead per unit',
        'dual chunks repeat fitting and use maximum measured 5D nested and serialization time for every policy',
        'P1 measures exact shared full section functions and assertions; each section includes measured fresh-process import plus 10 s restart overhead',
        'timings use development seed 170; runtime variability remains'],
        validation_timings=validation,preserved_sha256=before)
    (out/'preflight.json').write_text(json.dumps(report,indent=2,default=float))
    if level is None or oversized:
        (out/'freeze_blocked.json').write_text(json.dumps(dict(reason='L0 infeasible or unit exceeds 720 seconds',ledger=ledger,oversized=oversized),indent=2))
        return
    freeze=dict(created_utc=datetime.now(timezone.utc).isoformat(),level=level,
        namespace=out.name,full_namespace='results/full/'+out.name,
        provenance=provenance,config=CONFIG,settings=asdict(settings),
        source_sha256={p.name:sha(p) for p in sorted(Path('.').glob('*.py'))},
        environment_hash=stable_hash(provenance['environment']),ledger=ledger,
        manifest=selected,all_level_projections=units,artifact_inventory=inventory,
        validation_contract=dict(module='validation_sections.py', mode='full',
            sections=['checks','put','convergence','dual'],
            RUN_PAPER_CHECK=False, paper_regime_implementation_check='not tested',
            convergence=dict(dimensions=[5], budgets=[1000,10000,50000], seeds=[500,501,502], test=200000),
            resume='separate section outputs; put and dual rebuild their reference'),
        study_contract='Each study unit produces its NN variant and a paired LSM result, including every sorted+payoff unit',
        conservative_extra_m4_seconds=report['conservative_extra_m4_seconds'],
        budget_accounting=report['budget_accounting'],
        preserved_sha256=before,resume_evidence=report['resume_evidence'],
        limitations=report['projection_assumptions']+['resume pilot does not test process termination during serialization'],
        dual_sample_contract=dict(U='upper_samples',L='initial_value_samples = C0',
            realised='lower_samples = discounted Z_tau',delta='U - L',raw_gap='U - realised',
            resume_requires=['code_hash','config_hash','policy_hash','outer_indices','seed','budgets','group'],
            policy_hash='SHA256 of actual fitted policy arrays, checked on every resume',
            doubling_indices=list(range(500)),aggregate='aligned per-path samples, never chunk means'))
    with (out/'frozen_plan.json').open('x') as f:
        json.dump(freeze,f,indent=2,default=float)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--balance', type=float)
    parser.add_argument('--snapshot-timestamp', type=float)
    parser.add_argument('--prior-seconds', type=float, default=0.)
    parser.add_argument('--resume-evidence')
    parser.add_argument('--notebook-seconds', type=float, default=198.18)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--production', action='store_true')
    group.add_argument('--combined-freeze', action='store_true')
    group.add_argument('--resume-pilot', action='store_true')
    group.add_argument('--validation', action='store_true')
    args = parser.parse_args()
    if args.combined_freeze:
        if args.balance is None or args.snapshot_timestamp is None:
            parser.error('combined freeze requires balance and timestamp')
        combined_freeze(args.out, balance=args.balance, timestamp=args.snapshot_timestamp,
                        prior_seconds=args.prior_seconds, resume_evidence=args.resume_evidence,
                        notebook_seconds=args.notebook_seconds)
    else:
        (validation_probe if args.validation else resume_pilot if args.resume_pilot else production_probe if args.production else run)(args.out)
