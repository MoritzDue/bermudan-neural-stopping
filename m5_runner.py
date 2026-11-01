"""Frozen-study execution helpers for controller-owned temporary notebooks.

This module does not authorize jobs or execute work on import. Full execution is
called only from controller-authorized notebook copies.
"""
import hashlib
import json
from pathlib import Path
import time
import math
import numpy as np

import stopping as st

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_plan(path):
    path = Path(path)
    plan = json.loads(path.read_text())
    for name, expected in plan['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'frozen source mismatch: {name}')
    if plan['config'] != st.CONFIG:
        raise ValueError('frozen CONFIG mismatch')
    current = st.calibration_provenance('full')['environment']
    if current != plan['provenance']['environment']:
        raise ValueError('frozen environment mismatch')
    ids = [u['id'] for u in plan['manifest']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate manifest unit ids')
    return plan, sha256(path)


def unit_contract(plan, plan_hash, unit_id):
    matches = [u for u in plan['manifest'] if u['id'] == unit_id]
    if len(matches) != 1:
        raise ValueError('unit must occur exactly once in frozen manifest')
    return dict(schema=1, plan_sha256=plan_hash, unit=matches[0],
                frozen_source_sha256=plan['source_sha256'],
                runner_sha256=sha256(__file__),
                frozen_code_hash=plan['provenance']['code_hash'],
                execution_code_hash=st.code_hash())


CHECKPOINT_NAMESPACE = 'checkpoints-v2'


def sanitise_result(value):
    """Canonical JSON result: non-finite diagnostics become missing values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): sanitise_result(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitise_result(v) for v in value]
    return value


def checkpoint(path, contract, execute):
    """A complete atomic record is the only evidence accepted for resume."""
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get('contract') != contract or saved.get('complete') is not True:
            raise ValueError('incompatible checkpoint; existing evidence preserved')
        if saved.get('result_sha256') != st.stable_hash(saved['result']):
            raise ValueError('checkpoint result hash mismatch')
        return saved['result']
    started = time.perf_counter()
    result = sanitise_result(execute())
    record = dict(contract=contract, complete=True, result=result,
                  result_sha256=st.stable_hash(result), seconds=time.perf_counter()-started)
    encoded = json.dumps(record, default=st._jsonable, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(encoded)
    temporary.replace(path)
    return result


def run_study_unit(plan_path, unit_id, output_root):
    """Run one manifest study unit; caller owns notebook isolation and timeout."""
    plan, digest = load_plan(plan_path)
    contract = unit_contract(plan, digest, unit_id)
    unit = contract['unit']
    if unit['kind'] != 'study':
        raise ValueError('study unit required')
    output_root = Path(output_root).resolve()
    if output_root != (ROOT / plan['full_namespace']).resolve():
        raise ValueError('output must equal frozen full namespace')

    def execute():
        st.limit_threads(1)
        cfg = plan['config']['full']
        opt = st.paper_max_call(unit['d'])
        test = st.gbm_paths(opt, cfg['study']['test'],
                            st.seed_for(cfg['base_seed'], 'test', opt.d))
        return study_with_policies(opt, unit['budget'], unit['seed'], [unit['variant']],
                           st.NeuralSettings(**plan['settings']), cfg['study']['val'],
                           test, st.reference_for(opt), cfg['base_seed'])

    return checkpoint(output_root / CHECKPOINT_NAMESPACE / 'study_units' / (unit_id + '.json'), contract, execute)


def validation_section(section, plan, destination):
    """Execute exactly the retained notebook-01 section used by preflight."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import validation_sections as vs

    cfg = plan['config']['full']['validation']
    base = plan['config']['full']['base_seed']
    st.limit_threads(1)
    try:
        if section == 'checks':
            result = vs.validation_checks()
            return dict(reference=result['put_ref'],
                        crr={str(k): v for k, v in result['crr'].items()})
        if section == 'convergence':
            result = vs.validation_convergence('full', cfg, base, destination, 500)
            return dict(rows=result['lsm_df'].to_dict(orient='records'))
        put = st.validation_put()
        reference = st.reference_for(put)
        if section == 'put':
            result = vs.validation_put_fit('full', cfg, base, destination, 500, put, reference)
            return dict(reference=reference, rows=result['put_df'].to_dict(orient='records'))
        if section == 'dual':
            result = vs.validation_dual('full', cfg, base, destination, 500, put, reference)
            return dict(rows=result['dual_rows'])
        raise ValueError('unknown frozen validation section')
    finally:
        plt.close('all')


def run_validation_item(plan_path, unit_id, output_root):
    """Checkpoint P1 assertions and verify section artifacts on every resume.

    Partial section outputs remain evidence but never count as completion. A
    failed assertion leaves no complete checkpoint; the next invocation reruns
    the entire section, as with interrupted study units.
    """
    plan, digest = load_plan(plan_path)
    contract = unit_contract(plan, digest, unit_id)
    unit = contract['unit']
    if unit['kind'] != 'validation':
        raise ValueError('validation unit required')
    output_root = Path(output_root).resolve()
    if output_root != (ROOT / plan['full_namespace']).resolve():
        raise ValueError('output must equal frozen full namespace')
    destination = output_root / CHECKPOINT_NAMESPACE / 'validation' / unit_id

    def execute():
        destination.mkdir(parents=True, exist_ok=True)
        evidence = validation_section(unit['section'], plan, destination)
        artifacts = {str(p.relative_to(destination)): sha256(p)
                     for p in sorted(destination.iterdir()) if p.is_file()}
        return dict(evidence=evidence, artifacts=artifacts)

    result = checkpoint(output_root / CHECKPOINT_NAMESPACE / 'validation_units' / (unit_id + '.json'),
                        contract, execute)
    for relative, expected in result['artifacts'].items():
        artifact = destination / relative
        if not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError('validation artifact missing or changed; evidence preserved')
    return result


def encode_policy(policy):
    """Store fitted numerical rules as JSON, without executable pickle payloads."""
    from dataclasses import asdict
    rules = {}
    for k, rule in policy.rules.items():
        if rule is None:
            rules[str(k)] = None
        elif policy.method == 'lsm':
            # fit_lsm binds its fitted regression as the rule's first default.
            # Round-trip tests cover this coupling to the frozen implementation.
            reg = rule.__defaults__[0]
            rules[str(k)] = dict(mu=reg.mu.tolist(), sd=reg.sd.tolist(),
                                 coef=reg.coef.tolist())
        else:
            rules[str(k)] = dict(mu=rule.mu.tolist(), sd=rule.sd.tolist(),
                settings=asdict(rule.s),
                state={n: dict(values=t.tolist(), dtype=str(t.numpy().dtype))
                       for n, t in rule.net.state_dict().items()})
    return dict(method=policy.method, guard=policy.guard, rules=rules,
                degree=policy.info.get('degree'), ridge=policy.info.get('ridge'))


def decode_policy(data, opt):
    import numpy as np
    import torch
    policy = st.Policy(data['method'])
    policy.guard = {int(k): v for k, v in data['guard'].items()}
    for key, row in data['rules'].items():
        k = int(key)
        if row is None:
            policy.rules[k] = None
        elif data['method'] == 'lsm':
            reg = st.RidgeRegression.__new__(st.RidgeRegression)
            reg.mu, reg.sd, reg.coef = (np.asarray(row[n]) for n in ('mu', 'sd', 'coef'))
            degree = data['degree']
            policy.rules[k] = lambda x, reg=reg: opt.payoff(x) >= reg.predict(
                st.polynomial(st.lsm_features(x, opt), degree))
        else:
            settings = st.NeuralSettings(**row['settings'])
            net = st.StopNet(len(row['mu']), opt.d + settings.width_extra, settings.batch_norm)
            net.load_state_dict({n: torch.from_numpy(np.asarray(t['values'], dtype=t['dtype']))
                                 for n, t in row['state'].items()})
            policy.rules[k] = st.NeuralRule(net, np.asarray(row['mu'], dtype=np.float32),
                np.asarray(row['sd'], dtype=np.float32), opt, settings)
    return policy


def policy_key(method):
    return 'lsm' if method == 'lsm' else 'nn_' + method


def persisted_dual_policy(plan, digest, unit, output_root):
    """Reuse the exact study policy; never refit against a different execution."""
    key = policy_key(unit['method'])
    matches = [u for u in plan['manifest'] if u['kind'] == 'study'
               and all(u[k] == unit[k] for k in ('d', 'budget', 'seed'))
               and (key == 'lsm' or 'nn_' + u['variant'] == key)]
    if len(matches) != 1:
        raise ValueError('dual requires one matching frozen study unit')
    study = matches[0]
    contract = unit_contract(plan, digest, study['id'])
    path = Path(output_root) / CHECKPOINT_NAMESPACE / 'study_units' / (study['id'] + '.json')
    if not path.is_file():
        raise ValueError('matching study checkpoint must complete before dual')
    def no_refit():
        raise ValueError('study policy cannot be refitted')
    result = checkpoint(path, contract, no_refit)
    data = result['policies'][key]
    opt = st.paper_max_call(unit['d'])
    return opt, decode_policy(data, opt), st.stable_hash(data)


def run_dual_chunk(plan_path, unit_id, output_root):
    """Controller notebook entry point; retain all three per-path paired samples."""
    from dual import nested_policy_dual
    import numpy as np
    plan, digest = load_plan(plan_path)
    contract = unit_contract(plan, digest, unit_id)
    unit = contract['unit']
    if unit['kind'] != 'dual':
        raise ValueError('dual unit required')
    output_root = Path(output_root).resolve()
    if output_root != (ROOT / plan['full_namespace']).resolve():
        raise ValueError('output must equal frozen full namespace')
    st.limit_threads(1)
    opt, policy, policy_hash = persisted_dual_policy(plan, digest, unit, output_root)
    contract['policy_sha256'] = policy_hash

    def execute():
        results = nested_policy_dual(opt, policy, budgets=tuple(unit['inner']),
            outer=unit['outer'], outer_indices=unit['outer_indices'],
            seed=unit['rng_seed'], group=unit['group'])
        rows = []
        for result in results:
            arrays = {name: np.asarray(result[key]) for name, key in
                      [('U', 'upper_samples'), ('C0', 'initial_value_samples'),
                       ('Ztau', 'lower_samples')]}
            if any(a.shape != (len(unit['outer_indices']),) or not np.isfinite(a).all()
                   for a in arrays.values()):
                raise ValueError('invalid dual samples')
            rows.append(dict(inner=result['metadata']['inner'],
                             **{k: a.tolist() for k, a in arrays.items()}))
        return dict(policy_sha256=policy_hash, outer_indices=unit['outer_indices'], samples=rows)

    return checkpoint(output_root / CHECKPOINT_NAMESPACE / 'dual_units' / (unit_id + '.json'), contract, execute)


# Frozen run_unit evaluation, with fitted parameters retained for dual reuse.
def study_with_policies(opt: st.Option, n_train: int, seed: int, variants, settings: st.NeuralSettings, n_val: int, test_paths, reference: dict, base_seed: int, budget_label=None):
    """Fit LSM and every requested neural variant on the SAME training and validation
    paths of this seed, evaluate all of them (guarded and unguarded) on the SAME test
    paths, and return tidy rows. Nothing here selects anything using test paths."""
    case = opt.name
    t0 = time.perf_counter()
    train = st.gbm_paths(opt, n_train, st.seed_for(base_seed, 'train', opt.d, seed))
    val = st.gbm_paths(opt, n_val, st.seed_for(base_seed, 'val', opt.d, seed))
    sim_s = time.perf_counter() - t0
    policies = {}
    lsm, _ = st.select_lsm(opt, train, val)
    policies['lsm'] = lsm
    for v in variants:
        policies[f'nn_{v}'] = st.fit_neural(opt, train, val, st.with_variant(settings, v), seed=st.seed_for(base_seed, 'init', opt.d, seed))
    key = dict(case=case, d=opt.d, budget=n_train, seed=seed)
    runs, pairs, dates, payoffs = ([], [], [], {})
    for name, pol in policies.items():
        for guarded in (True, False):
            t1 = time.perf_counter()
            values, tau = st.discounted_payoffs(opt, test_paths, pol, guarded)
            eval_s = time.perf_counter() - t1
            payoffs[name, guarded] = values
            v = float(values.mean())
            runs.append(dict(**key, method=name, guarded=guarded, value=v, test_se=float(values.std(ddof=1) / math.sqrt(len(values))), gap=reference['value'] - v, early_exercise=float(np.mean(tau < opt.dates)), n_guarded_dates=pol.n_guarded(), train_s=pol.info['train_s'], selection_s=pol.info['selection_s'], sim_s=sim_s, eval_s=eval_s, n_val=n_val, n_test=len(test_paths), lsm_degree=pol.info.get('degree'), lsm_ridge=pol.info.get('ridge')))
        for row in pol.diagnostics:
            dates.append(dict(**key, method=name, **{k: v for k, v in row.items() if k != 'curve'}))
    for guarded in (True, False):
        for name in policies:
            if name != 'lsm':
                m, se = st.paired(payoffs[name, guarded], payoffs['lsm', guarded])
                pairs.append(st.paired_row(key, guarded, f'{name} - lsm', m, se))
        if all((f'nn_{v}' in policies for v in st.VARIANTS)):
            for cname, weights in st.ABLATION_CONTRASTS.items():
                comb = sum((w * payoffs[f'nn_{v}', guarded] for v, w in weights.items()))
                pairs.append(st.paired_row(key, guarded, cname, float(comb.mean()), float(comb.std(ddof=1) / math.sqrt(len(comb)))))
    for name in policies:
        m, se = st.paired(payoffs[name, True], payoffs[name, False])
        pairs.append(st.paired_row(key, True, f'{name}: guarded - unguarded', m, se))
    return dict(runs=runs, paired=pairs, dates=dates, policies={name: encode_policy(pol) for name, pol in policies.items()})


def sample_ci(values):
    """Student-t interval over independent outer paths (or training seeds)."""
    from scipy.stats import t
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('CI requires at least two finite samples')
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    half = float(t.ppf(.975, len(x)-1) * se)
    return dict(mean=mean, se=se, ci_low=mean-half, ci_high=mean+half,
                half_width=half, n=len(x))


def read_completed(path, contract):
    if not Path(path).is_file():
        raise ValueError(f'missing required checkpoint: {path}')
    def missing():
        raise ValueError('aggregation never executes unfinished work')
    return checkpoint(path, contract, missing)


def dual_statistics(entries):
    """Aggregate indexed samples, never chunk means; check coupled prefixes."""
    base, doubled = {}, {}
    for unit, result in entries:
        indices = unit['outer_indices']
        if result['outer_indices'] != indices:
            raise ValueError('outer index mismatch')
        if [r['inner'] for r in result['samples']] != unit['inner']:
            raise ValueError('inner budget mismatch')
        target = base if unit['inner'] == [3200] else doubled
        if unit['inner'] not in ([3200], [3200, 6400]):
            raise ValueError('unexpected inner budgets')
        for j, i in enumerate(indices):
            if i in target:
                raise ValueError('duplicate outer index')
            target[i] = {}
            for row in result['samples']:
                arrays = [np.asarray(row[k], dtype=float) for k in ('U', 'C0', 'Ztau')]
                if any(a.shape != (len(indices),) or not np.isfinite(a).all() for a in arrays):
                    raise ValueError('invalid dual samples')
                target[i][row['inner']] = tuple(float(a[j]) for a in arrays)
    outer = entries[0][0]['outer']
    if sorted(base) != list(range(outer)):
        raise ValueError('incomplete base outer coverage')
    u, c0, ztau = np.asarray([base[i][3200] for i in range(outer)]).T
    summary = dict(upper=sample_ci(u), initial_value=sample_ci(c0),
                   gap=sample_ci(u-c0), raw_gap=sample_ci(u-ztau),
                   raw_lower=sample_ci(ztau))
    if doubled:
        if sorted(doubled) != list(range(500)):
            raise ValueError('incomplete doubling subset')
        # Exact identity is a frozen scientific requirement, not a tolerance check.
        for i in range(500):
            if doubled[i][3200] != base[i][3200]:
                raise ValueError(f'coupled 3200 prefix mismatch at outer index {i}')
        changes = [doubled[i][6400][0] - base[i][3200][0] for i in range(500)]
        summary['doubling'] = sample_ci(changes)
        summary['doubling']['threshold_full_upper_se'] = summary['upper']['se']
        summary['doubling']['stable'] = abs(summary['doubling']['mean']) < summary['upper']['se']
        summary['h3'] = dict(gap=summary['gap']['mean'] <= .10,
                             precision=summary['gap']['half_width'] <= .03,
                             stability=summary['doubling']['stable'])
    return summary


def aggregate(plan_path, output_root):
    """Regenerate L2 tables only from complete, hash-checked frozen checkpoints."""
    import pandas as pd
    plan, digest = load_plan(plan_path)
    root = Path(output_root).resolve()
    if root != (ROOT / plan['full_namespace']).resolve():
        raise ValueError('output must equal frozen full namespace')
    if plan['level'] != 2:
        raise ValueError('this analysis implements the frozen L2 design only')
    studies, groups, validation = [], {}, []
    for unit in plan['manifest']:
        kind = unit['kind']
        contract = unit_contract(plan, digest, unit['id'])
        if kind == 'dual':
            _, _, policy_hash = persisted_dual_policy(plan, digest, unit, root)
            contract['policy_sha256'] = policy_hash
        result = read_completed(root / CHECKPOINT_NAMESPACE / (kind + '_units') / (unit['id']+'.json'), contract)
        if kind == 'study':
            studies.append(dict(id=unit['id'], contract=contract,
                                **{k: result[k] for k in ('runs', 'paired', 'dates')}))
        elif kind == 'dual':
            if result['policy_sha256'] != contract['policy_sha256']:
                raise ValueError('dual policy mismatch')
            key = tuple(unit[k] for k in ('d', 'budget', 'seed', 'method'))
            groups.setdefault(key, []).append((unit, result))
        elif kind == 'validation':
            for relative, expected in result['artifacts'].items():
                if sha256(root / CHECKPOINT_NAMESPACE / 'validation' / unit['id'] / relative) != expected:
                    raise ValueError('validation artifact mismatch')
            validation.append(dict(id=unit['id'], **result))
        else:
            raise ValueError('unknown manifest kind')
    # All checks precede replacing any derived table.
    runs = pd.DataFrame([r for s in studies for r in s['runs']])
    pairs = pd.DataFrame([r for s in studies for r in s['paired']])
    dates = pd.DataFrame([r for s in studies for r in s['dates']])
    main = []
    for (d, budget, method), sub in runs[runs.guarded].groupby(['d', 'budget', 'method']):
        if sub.seed.duplicated().any():
            raise ValueError('duplicate study seed')
        main.append(dict(d=int(d), budget=int(budget), method=method,
                         value=float(sub.value.mean()), gap=float(sub.gap.mean()),
                         train_s=float(sub.train_s.mean()),
                         selection_s=float(sub.selection_s.mean()),
                         **st.absolute_gap_uncertainty(sub.gap, sub.test_se)))
    h2 = []
    contrast = pairs[pairs.guarded & (pairs.contrast == 'nn_sorted+payoff - lsm')]
    for (d, budget), sub in contrast.groupby(['d', 'budget']):
        if sub.seed.duplicated().any():
            raise ValueError('duplicate contrast seed')
        row = dict(d=int(d), budget=int(budget), **sample_ci(sub['diff']),
                   shared_test_se_bound=float(sub.paired_se.mean()),
                   shared_test_se_description='mean of per-seed paired SEs; not a combined SE',
                   uncertainty='training-seed variation conditional on shared test set',
                   role='primary' if budget in (1000, 50000) else 'descriptive')
        h2.append(row)
    h3 = [dict(d=k[0], budget=k[1], seed=k[2], method=k[3],
               **dual_statistics(entries)) for k, entries in sorted(groups.items())]
    summary = dict(plan_sha256=digest, level=plan['level'], H2=h2, H3=h3,
                   H1='not tested: L3 not frozen', two_dimensional='not tested: L4 not frozen',
                   development_cost='not measured', validation=validation,
                   limitations=['Finite-inner U bounds in expectation, not pathwise.',
                                'Seed intervals are conditional on the common test set.',
                                'No calibration grid; fixed neural settings.'])
    outputs = {'units.jsonl': ''.join(json.dumps(s, default=st._jsonable, allow_nan=False)+'\n' for s in studies),
               'summary.json': json.dumps(summary, indent=2, default=st._jsonable, allow_nan=False),
               'runs.csv': runs.to_csv(index=False), 'paired.csv': pairs.to_csv(index=False),
               'dates.csv': dates.to_csv(index=False), 'summary_main.csv': pd.DataFrame(main).to_csv(index=False),
               'H2.csv': pd.DataFrame(h2).to_csv(index=False)}
    flat = []
    for row in h3:
        item = {k: row[k] for k in ('d', 'budget', 'seed', 'method')}
        for name, values in row.items():
            if isinstance(values, dict):
                item.update({name+'_'+k: v for k, v in values.items()})
        flat.append(item)
    outputs['H3.csv'] = pd.DataFrame(flat).to_csv(index=False)
    for name, text in outputs.items():
        path = root / name
        pending = path.with_suffix(path.suffix+'.partial')
        pending.write_text(text)
        pending.replace(path)
    return summary


def inventory_superseded(root):
    """Preserve and inventory failed-run files before versioned execution."""
    # Inventory earlier failed-run artifacts without touching or resuming them.
    inventory_path = root / CHECKPOINT_NAMESPACE / 'superseded-inventory.json'
    if not inventory_path.exists():
        inventory = [dict(path=str(p.relative_to(root)), sha256=sha256(p),
                          status='superseded mechanical-repair evidence')
                     for p in sorted(root.rglob('*')) if p.is_file()
                     and CHECKPOINT_NAMESPACE not in p.relative_to(root).parts]
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text(json.dumps(dict(artifacts=inventory,
            reason='NaN serialization repair; strict runner contract retained'), indent=2))


def run_manifest_stage(stage):
    """Entry point used only by the three controller-authorized full notebooks."""
    plan_path = ROOT / 'results/quick/m4-combined-freeze-rev5/frozen_plan.json'
    plan, _ = load_plan(plan_path)
    root = ROOT / plan['full_namespace']
    inventory_superseded(root)
    if stage == 'aggregate':
        return aggregate(plan_path, root)
    kinds = {'study': ('validation', 'study'), 'dual': ('dual',)}[stage]
    runners = dict(validation=run_validation_item, study=run_study_unit, dual=run_dual_chunk)
    for unit in plan['manifest']:
        if unit['kind'] in kinds:
            print('Starting/resuming', unit['id'], flush=True)
            runners[unit['kind']](plan_path, unit['id'], root)
            print('Complete', unit['id'], flush=True)
