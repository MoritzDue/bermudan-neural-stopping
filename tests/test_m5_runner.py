import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from m5_runner import ROOT, checkpoint, load_plan, unit_contract


class RunnerTests(unittest.TestCase):
    def test_frozen_sources_and_manifest_mapping(self):
        path = ROOT / 'results/quick/m4-combined-freeze-rev5/frozen_plan.json'
        plan, digest = load_plan(path)
        unit = next(u for u in plan['manifest'] if u['kind'] == 'study')
        contract = unit_contract(plan, digest, unit['id'])
        self.assertEqual(contract['unit'], unit)
        self.assertEqual(contract['frozen_source_sha256'], plan['source_sha256'])
        self.assertEqual(len(contract['runner_sha256']), 64)
        with self.assertRaises(ValueError):
            unit_contract(plan, digest, 'not-a-unit')

    def test_resume_rejects_changed_contract_and_corruption(self):
        # Honour the controller's writable TMPDIR when source is read-only.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'unit.json'
            execute = Mock(return_value={'value': 3.5})
            self.assertEqual(checkpoint(path, {'seed': 1}, execute), {'value': 3.5})
            checkpoint(path, {'seed': 1}, execute)
            self.assertEqual(execute.call_count, 1)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                checkpoint(path, {'seed': 2}, execute)
            self.assertEqual(before, path.read_bytes())
            saved = json.loads(path.read_text())
            saved['result']['value'] = 0
            path.write_text(json.dumps(saved))
            with self.assertRaises(ValueError):
                checkpoint(path, {'seed': 1}, execute)

    def test_put_shape_nonfinite_checkpoint_roundtrip(self):
        import numpy as np
        import pandas as pd
        import stopping as st
        import m5_runner as runner
        from unittest.mock import patch
        # Exercise the actual section adapter; avoid rerunning the full fit.
        frame = pd.DataFrame([
            dict(method='lsm', lsm_degree=3, value=6.0),
            dict(method='nn_sorted+payoff', lsm_degree=None, value=6.01)])
        self.assertTrue(np.isnan(frame.iloc[1].lsm_degree))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'put.json'
            plan = dict(config=dict(full=dict(validation={}, base_seed=1)))
            with patch('validation_sections.validation_put_fit', return_value={'put_df': frame}):
                result = runner.validation_section('put', plan, Path(tmp))
            result['nested'] = [np.float64(np.inf), {'minus': -np.inf, 'nan': float('nan')}]
            first = checkpoint(path, {'frozen': 'binding'}, lambda: result)
            saved = json.loads(path.read_text())
            self.assertIsNone(first['rows'][1]['lsm_degree'])
            self.assertEqual(first['nested'], [None, {'minus': None, 'nan': None}])
            self.assertEqual(saved['result'], first)
            self.assertEqual(saved['result_sha256'], st.stable_hash(first))
            self.assertEqual(first, checkpoint(path, {'frozen': 'binding'},
                Mock(side_effect=AssertionError('must resume'))))

    def test_versioned_inventory_preserves_old_evidence(self):
        import m5_runner as runner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / 'validation_units/P1-checks.json'
            partial = root / 'validation/P1-put/validation_put.csv'
            for path, content in [(old, 'old checkpoint'), (partial, 'partial csv')]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            before = {str(p.relative_to(root)): p.read_bytes() for p in (old, partial)}
            runner.inventory_superseded(root)
            inventory = root / runner.CHECKPOINT_NAMESPACE / 'superseded-inventory.json'
            record = json.loads(inventory.read_text())
            self.assertEqual({a['path'] for a in record['artifacts']}, set(before))
            saved = inventory.read_bytes()
            runner.inventory_superseded(root)
            self.assertEqual(saved, inventory.read_bytes())
            for relative, content in before.items():
                self.assertEqual((root / relative).read_bytes(), content)

    def test_interrupted_work_is_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'unit.json'
            with self.assertRaises(RuntimeError):
                checkpoint(path, {}, Mock(side_effect=RuntimeError('interrupted')))
            self.assertFalse(path.exists())
            self.assertEqual(checkpoint(path, {}, lambda: {'done': True}), {'done': True})


class ValidationRunnerTests(unittest.TestCase):
    def test_validation_resume_checks_artifacts_without_reexecution(self):
        from unittest.mock import patch
        import m5_runner as runner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unit = dict(id='P1-put', kind='validation', section='put')
            plan = dict(full_namespace=str(root), manifest=[unit])

            def section(name, frozen, destination):
                self.assertEqual(name, 'put')
                (destination / 'result.csv').write_text('value\n6.0\n')
                return {'passed': True}

            with patch.object(runner, 'load_plan', return_value=(plan, 'digest')), \
                 patch.object(runner, 'unit_contract', return_value={'unit': unit}), \
                 patch.object(runner, 'validation_section', side_effect=section) as execute:
                first = runner.run_validation_item('unused', 'P1-put', root)
                self.assertEqual(first, runner.run_validation_item('unused', 'P1-put', root))
                self.assertEqual(execute.call_count, 1)
                artifact = root / 'checkpoints-v2/validation/P1-put/result.csv'
                artifact.write_text('corrupt')
                with self.assertRaisesRegex(ValueError, 'artifact'):
                    runner.run_validation_item('unused', 'P1-put', root)
                self.assertEqual(execute.call_count, 1)
                self.assertEqual(artifact.read_text(), 'corrupt')

    def test_validation_failure_is_not_checkpointed(self):
        from unittest.mock import patch
        import m5_runner as runner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unit = dict(id='P1-dual', kind='validation', section='dual')
            with patch.object(runner, 'load_plan', return_value=({'full_namespace': str(root)}, 'digest')), \
                 patch.object(runner, 'unit_contract', return_value={'unit': unit}), \
                 patch.object(runner, 'validation_section', side_effect=AssertionError('CRR')):
                with self.assertRaises(AssertionError):
                    runner.run_validation_item('unused', 'P1-dual', root)
                self.assertFalse((root / 'checkpoints-v2/validation_units/P1-dual.json').exists())


class PolicyPersistenceTests(unittest.TestCase):
    def test_lsm_and_neural_roundtrip(self):
        import numpy as np
        import stopping as st
        import m5_runner as runner
        st.limit_threads(1)
        opt = st.validation_put()
        train = st.gbm_paths(opt, 500, 172)
        val = st.gbm_paths(opt, 200, 173)
        policies = [st.fit_lsm(opt, train, val, 1, 1e-6),
                    st.fit_neural(opt, train, val,
                        st.NeuralSettings(max_steps=2, batch=64), seed=174)]
        for policy in policies:
            encoded = json.loads(json.dumps(runner.encode_policy(policy)))
            restored = runner.decode_policy(encoded, opt)
            for guarded in (True, False):
                a = st.discounted_payoffs(opt, val, policy, guarded)
                b = st.discounted_payoffs(opt, val, restored, guarded)
                for x, y in zip(a, b):
                    np.testing.assert_array_equal(x, y)

    def test_dual_chunk_resume_and_policy_contract(self):
        from unittest.mock import patch
        import numpy as np
        import m5_runner as runner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unit = dict(id='dual-test', kind='dual', inner=[3200, 6400], outer=2000,
                        outer_indices=[0, 1], rng_seed=123, group=200)
            plan = dict(full_namespace=str(root))
            results = [dict(metadata={'inner': n}, upper_samples=np.array([3., 4.]),
                       initial_value_samples=np.array([2., 2.]),
                       lower_samples=np.array([1., 5.])) for n in unit['inner']]
            with patch.object(runner, 'load_plan', return_value=(plan, 'digest')), \
                 patch.object(runner, 'unit_contract', side_effect=lambda *a: {'unit': unit}), \
                 patch.object(runner, 'persisted_dual_policy', return_value=(None, None, 'policy1')) as pol, \
                 patch('dual.nested_policy_dual', return_value=results) as execute:
                result = runner.run_dual_chunk('unused', unit['id'], root)
                self.assertEqual(result['samples'][0]['C0'], [2., 2.])
                runner.run_dual_chunk('unused', unit['id'], root)
                self.assertEqual(execute.call_count, 1)
                pol.return_value = (None, None, 'policy2')
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    runner.run_dual_chunk('unused', unit['id'], root)

class StudyIdentityTests(unittest.TestCase):
    def test_saved_study_policy_is_reused_and_matches_values(self):
        import numpy as np
        import stopping as st
        import m5_runner as runner
        from unittest.mock import patch
        st.limit_threads(1)
        opt = st.paper_max_call(2)
        test = st.gbm_paths(opt, 150, 591)
        settings = st.NeuralSettings(max_steps=2, batch=64)
        args = (opt, 200, 0, ['sorted+payoff'], settings, 100,
                test, st.reference_for(opt), 451)
        result = runner.study_with_policies(*args)
        original = st.run_unit(*args)
        for a, b in zip(result['runs'], original['runs']):
            self.assertEqual(a['value'], b['value'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = dict(id='study', kind='study', d=2, budget=200, seed=0,
                         variant='sorted+payoff')
            plan = dict(manifest=[study])
            contract = {'identity': 'frozen'}
            runner.checkpoint(root / 'checkpoints-v2/study_units/study.json', contract, lambda: result)
            for method in ('lsm', 'sorted+payoff'):
                unit = dict(d=2, budget=200, seed=0, method=method)
                with patch.object(runner, 'unit_contract', return_value=contract), \
                     patch.object(st, 'select_lsm', side_effect=AssertionError('refit')), \
                     patch.object(st, 'fit_neural', side_effect=AssertionError('refit')):
                    _, policy, digest = runner.persisted_dual_policy(plan, 'hash', unit, root)
                values, _ = st.discounted_payoffs(opt, test, policy)
                row = next(r for r in result['runs'] if r['method'] == runner.policy_key(method) and r['guarded'])
                self.assertEqual(float(values.mean()), row['value'])
                self.assertEqual(digest, st.stable_hash(result['policies'][runner.policy_key(method)]))


class AggregationTests(unittest.TestCase):
    def test_real_manifest_every_dual_has_one_policy(self):
        import m5_runner as r
        plan, _ = r.load_plan(ROOT / 'results/quick/m4-combined-freeze-rev5/frozen_plan.json')
        for dual in [u for u in plan['manifest'] if u['kind'] == 'dual']:
            key = r.policy_key(dual['method'])
            matches = [s for s in plan['manifest'] if s['kind'] == 'study'
                       and all(s[k] == dual[k] for k in ('d', 'budget', 'seed'))
                       and (key == 'lsm' or 'nn_'+s['variant'] == key)]
            self.assertEqual(len(matches), 1, dual['id'])
            self.assertIn(key, ('lsm', 'nn_'+matches[0]['variant']))

    def test_indexed_gap_and_prefix_rejection(self):
        import m5_runner as r
        import numpy as np
        base = dict(outer=500, outer_indices=list(range(500)), inner=[3200])
        u = np.arange(500, dtype=float)+3
        c = u-.08
        z = u + np.sin(u)
        row = dict(inner=3200, U=u.tolist(), C0=c.tolist(), Ztau=z.tolist())
        result = dict(outer_indices=base['outer_indices'], samples=[row])
        double = dict(base, inner=[3200, 6400])
        high = dict(row, inner=6400, U=(u-.001).tolist())
        doubled = dict(outer_indices=base['outer_indices'], samples=[dict(row), high])
        entries = [(base, result), (double, doubled)]
        summary = r.dual_statistics(entries)
        self.assertAlmostEqual(summary['gap']['mean'], .08)
        self.assertLess(summary['gap']['half_width'], 1e-12)
        self.assertAlmostEqual(summary['doubling']['mean'], -.001)
        self.assertTrue(all(summary['h3'].values()))
        doubled['samples'][0] = dict(row, U=(u+1e-10).tolist())
        with self.assertRaisesRegex(ValueError, 'prefix mismatch'):
            r.dual_statistics(entries)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            r.dual_statistics([(dict(base, outer=501), result)])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            r.dual_statistics([(base, result), (base, result)])

    def test_aggregate_from_checkpoints_and_repeat_exactly(self):
        import m5_runner as r
        from unittest.mock import patch
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = []
            for seed in range(3):
                manifest.append(dict(id=f's{seed}', kind='study', d=5, budget=1000,
                                     seed=seed, variant='sorted+payoff'))
            dual = dict(id='dual', kind='dual', d=5, budget=1000, seed=0,
                        method='sorted+payoff', outer=4, outer_indices=list(range(4)), inner=[3200])
            manifest.append(dual)
            plan = dict(manifest=manifest, full_namespace=str(root), level=2)
            def contract(p, digest, uid):
                return {'unit': next(u for u in manifest if u['id'] == uid)}
            for unit in manifest[:3]:
                runs = [dict(d=5, budget=1000, seed=unit['seed'], method=m, guarded=True,
                             value=26+unit['seed']*.01, gap=.159-unit['seed']*.01,
                             test_se=.08, train_s=1., selection_s=2.)
                        for m in ('lsm', 'nn_sorted+payoff')]
                pair = dict(d=5, budget=1000, seed=unit['seed'], guarded=True,
                            contrast='nn_sorted+payoff - lsm', diff=unit['seed']*.02, paired_se=.01)
                r.checkpoint(root/'checkpoints-v2/study_units'/(unit['id']+'.json'), contract(plan, '', unit['id']),
                             lambda runs=runs, pair=pair: dict(runs=runs, paired=[pair], dates=[]))
            dc = dict(contract(plan, '', 'dual'), policy_sha256='p')
            result = dict(policy_sha256='p', outer_indices=list(range(4)),
                          samples=[dict(inner=3200, U=[1.,2.,3.,4.], C0=[.9,1.9,2.9,3.9], Ztau=[0.,3.,1.,4.])])
            r.checkpoint(root/'checkpoints-v2/dual_units/dual.json', dc, lambda: result)
            with patch.object(r, 'load_plan', return_value=(plan, 'digest')), \
                 patch.object(r, 'unit_contract', side_effect=contract), \
                 patch.object(r, 'persisted_dual_policy', return_value=(None, None, 'p')):
                summary = r.aggregate('unused', root)
                self.assertAlmostEqual(summary['H2'][0]['mean'], .02)
                self.assertAlmostEqual(summary['H3'][0]['gap']['mean'], .1)
                self.assertAlmostEqual(summary['H3'][0]['raw_gap']['mean'], .5)
                self.assertAlmostEqual(summary['H2'][0]['ci_high'], r.sample_ci([0,.02,.04])['ci_high'])
                saved = {p.name:p.read_bytes() for p in root.iterdir() if p.is_file()}
                r.aggregate('unused', root)
                self.assertEqual(saved, {p.name:p.read_bytes() for p in root.iterdir() if p.is_file()})
                path = root/'checkpoints-v2/dual_units/dual.json'
                path.unlink()
                with self.assertRaisesRegex(ValueError, 'missing required checkpoint'):
                    r.aggregate('unused', root)

    def test_controller_stages_follow_frozen_manifest(self):
        import m5_runner as r
        from unittest.mock import patch
        plan, digest = r.load_plan(ROOT / 'results/quick/m4-combined-freeze-rev5/frozen_plan.json')
        for stage, kinds in [('study', ('validation', 'study')), ('dual', ('dual',))]:
            called = []
            def run(path, uid, root):
                called.append(uid)
            with patch.object(r, 'run_validation_item', side_effect=run), \
                 patch.object(r, 'run_study_unit', side_effect=run), \
                 patch.object(r, 'run_dual_chunk', side_effect=run), \
                 patch.object(r, 'inventory_superseded'), \
                 patch('builtins.print'):
                r.run_manifest_stage(stage)
            self.assertEqual(called, [u['id'] for u in plan['manifest'] if u['kind'] in kinds])
        with patch.object(r, 'inventory_superseded'), \
             patch.object(r, 'aggregate', return_value='summary') as agg:
            self.assertEqual(r.run_manifest_stage('aggregate'), 'summary')
            self.assertEqual(agg.call_count, 1)
