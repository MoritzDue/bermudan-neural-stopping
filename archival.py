"""Read-only verification of historical evidence; never authorizes resume or fits."""
import json
from pathlib import Path
import numpy as np
import m5_runner as runner
import stopping as st

OUTPUTS = ('units.jsonl', 'summary.json', 'runs.csv', 'paired.csv',
           'dates.csv', 'summary_main.csv', 'H2.csv', 'H3.csv')


def verify_record(record, expected):
    """Only the historical umbrella execution hash may differ."""
    contract = record.get('contract', {})
    historical = contract.get('execution_code_hash')
    if not isinstance(historical, str) or len(historical) != 16:
        raise ValueError('invalid historical execution hash')
    expected = dict(expected, execution_code_hash=historical)
    if contract != expected or record.get('complete') is not True:
        raise ValueError('unexpected archival contract')
    if record.get('result_sha256') != st.stable_hash(record['result']):
        raise ValueError('archival result hash mismatch')
    return historical


def verify_artifacts(destination, artifacts):
    destination = Path(destination).resolve()
    for relative, digest in artifacts.items():
        path = (destination / relative).resolve()
        if not path.is_relative_to(destination) or runner.sha256(path) != digest:
            raise ValueError('validation artifact mismatch')


def verify_policy(result, expected):
    if result.get('policy_sha256') != expected:
        raise ValueError('persisted policy integrity mismatch')


def regenerate(plan_path, destination):
    """Write eight derived files to new scratch storage, preserving old contracts.

    Source artifacts are read only. This is deliberately separate from the
    runner's strict resume entry points. No execution hash attribution is inferred.
    """
    import pandas as pd
    plan, digest = runner.load_plan(plan_path)
    root = (runner.ROOT / plan['full_namespace']).resolve()
    destination = Path(destination).resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError('archival output must be outside full evidence')
    if destination.exists():
        raise ValueError('archival destination must be new')
    if plan['level'] != 2:
        raise ValueError('only frozen L2 analysis supported')
    records = {}
    for unit in plan['manifest']:
        path = root / runner.CHECKPOINT_NAMESPACE / (unit['kind']+'_units') / (unit['id']+'.json')
        records[unit['id']] = json.loads(path.read_text())
    studies, groups, validation, hashes = [], {}, [], set()
    for unit in plan['manifest']:
        record = records[unit['id']]
        result = record['result']
        contract = runner.unit_contract(plan, digest, unit['id'])
        if unit['kind'] == 'dual':
            key = runner.policy_key(unit['method'])
            matches = [u for u in plan['manifest'] if u['kind'] == 'study'
                       and all(u[k] == unit[k] for k in ('d', 'budget', 'seed'))
                       and (key == 'lsm' or 'nn_'+u['variant'] == key)]
            if len(matches) != 1:
                raise ValueError('ambiguous persisted policy')
            data = records[matches[0]['id']]['result']['policies'][key]
            policy_hash = st.stable_hash(data)
            contract['policy_sha256'] = policy_hash
            verify_policy(result, policy_hash)
        hashes.add(verify_record(record, contract))
        if unit['kind'] == 'study':
            studies.append(dict(id=unit['id'], contract=record['contract'],
                                **{k: result[k] for k in ('runs', 'paired', 'dates')}))
        elif unit['kind'] == 'dual':
            key = tuple(unit[k] for k in ('d', 'budget', 'seed', 'method'))
            groups.setdefault(key, []).append((unit, result))
        elif unit['kind'] == 'validation':
            verify_artifacts(root / runner.CHECKPOINT_NAMESPACE / 'validation' / unit['id'],
                             result['artifacts'])
            validation.append(dict(id=unit['id'], **result))
        else:
            raise ValueError('unknown manifest kind')
    if len(hashes) != 1:
        raise ValueError('inconsistent historical execution hashes')
    destination.mkdir(parents=True)
    _render(plan, digest, destination, studies, groups, validation)
    comparisons = {name: (destination/name).read_bytes() == (root/name).read_bytes()
                   for name in OUTPUTS}
    report = dict(historical_execution_code_hash=next(iter(hashes)),
                  current_execution_code_hash=st.code_hash(),
                  exception='Umbrella execution hash only; no attribution inferred.',
                  byte_identical=comparisons,
                  runner_sha256=runner.sha256(runner.__file__),
                  plan_sha256=digest)
    (destination/'archival_verification.json').write_text(json.dumps(report, indent=2))
    if not all(comparisons.values()):
        raise ValueError('archival byte comparison failed; scratch evidence retained')
    return report


# Rendering mirrors the hash-verified frozen runner; no simulation is invoked.
def _render(plan, digest, root, studies, groups, validation):
    import pandas as pd
    sample_ci = runner.sample_ci
    dual_statistics = runner.dual_statistics
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
