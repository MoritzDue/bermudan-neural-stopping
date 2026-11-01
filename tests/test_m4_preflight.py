import unittest
from pathlib import Path
from m4_preflight import quick_destination, dual_projection, budget_ledger, production_projection


class TimingProbeTests(unittest.TestCase):
    def test_measured_base_projection_does_not_assume_half_cost(self):
        p = production_projection(60., 100, base_seconds=40.)
        self.assertEqual(p['base_2000_s'], 960.)
        self.assertEqual(p['base_block_100_s'], 48.)
        self.assertEqual(p['doubling_500_s'], 360.)
        self.assertTrue(p['base_measured'])
        for value in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                production_projection(60., 100, base_seconds=value)

    def test_production_coupled_projection_and_blocks(self):
        p = production_projection(60., 100)
        self.assertEqual(p['base_2000_s'], 720.)
        self.assertEqual(p['doubling_500_s'], 360.)
        self.assertEqual(p['base_block_100_s'], 36.)
        self.assertEqual(p['doubling_block_100_s'], 72.)
        for seconds, outer, safety in [(1,2,1.2),(1,100,1.1),(float('nan'),100,1.2)]:
            with self.assertRaises(ValueError):
                production_projection(seconds,outer,safety)

    def test_ledger_reserves_and_live_balance(self):
        args = dict(notebook_seconds=300, confirmed_remaining_seconds=10000,
                    snapshot_timestamp=1793547000, remaining_m4_seconds=100,
                    cumulative_costs=[1000, 2000, 3000, 5000, 7000, 9000])
        ledger = budget_ledger(**args)
        self.assertEqual(ledger['selected_level'], 4)
        self.assertEqual(ledger['levels'][0]['A'], .8 * (10000-100-420-300))
        self.assertEqual(ledger['levels'][4]['A'], .8 * (10000-100-420-700))
        self.assertEqual(ledger['B'], 10000)
        self.assertTrue(budget_ledger(**(args | {'confirmed_remaining_seconds': 700}))['escalate'])
        for change in ({'notebook_seconds': float('nan')},
                       {'cumulative_costs': [1, 2]},
                       {'cumulative_costs': [1, 2, 3, 2, 5, 6]}):
            with self.assertRaises(ValueError):
                budget_ledger(**(args | change))

    def test_ledger_equality_and_eight_hour_cap(self):
        ledger = budget_ledger(notebook_seconds=0, confirmed_remaining_seconds=100000,
            snapshot_timestamp=1, remaining_m4_seconds=0,
            cumulative_costs=[1000, 2000, 3000, 4000, 28800, 28801])
        self.assertEqual(ledger['selected_level'], 4)
        self.assertEqual(ledger['levels'][4]['A'], 28800)

    def test_output_boundary(self):
        for path in ('results/full/probe', 'results/quick', '../probe'):
            with self.assertRaises(ValueError):
                quick_destination(path)
        self.assertEqual(quick_destination('results/quick/m4-probe').name, 'm4-probe')

    def test_projection_includes_coupled_doubling_cost(self):
        result = dual_projection(3., 2)
        self.assertEqual(result['base_2000_s'], 3000.)
        self.assertEqual(result['doubling_500_s'], 1500.)
        with self.assertRaises(ValueError):
            dual_projection(0., 2)


class ChunkResumeTests(unittest.TestCase):
    def test_interruption_resume_and_coupled_prefix(self):
        import tempfile
        import numpy as np
        from stopping import Option, Policy
        from dual import nested_policy_dual, _summary
        from m4_preflight import checkpoint_dual_chunk
        opt = Option(kind='put', d=1, dates=3)
        policy = Policy('never')
        contract = dict(code_hash='code', config_hash='config', policy_hash='never',
                        outer=6, outer_indices=[0, 1, 2], seed=773,
                        budgets=[4, 8], group=2)
        # Saving the first chunk and leaving the second absent models an
        # interrupted job: resuming must reuse it without evaluating the policy.
        # Honour the controller's writable TMPDIR; source is read-only in jobs.
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp)/'first.npz'
            u1, l1, z1 = checkpoint_dual_chunk(first, opt, policy, contract=contract)
            with np.load(first, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files),
                                 {'contract', 'upper', 'initial_value', 'realised_payoff'})
                np.testing.assert_array_equal(saved['initial_value'], l1)
                np.testing.assert_array_equal(saved['realised_payoff'], z1)
            # Older checkpoints lack C0 and cannot establish the revised gap.
            import json
            legacy = Path(tmp)/'legacy.npz'
            np.savez(legacy, contract=json.dumps(contract, sort_keys=True,
                     separators=(',', ':')), upper=u1, lower=z1)
            before = legacy.read_bytes()
            with self.assertRaises(ValueError):
                checkpoint_dual_chunk(legacy, opt, policy, contract=contract)
            self.assertEqual(legacy.read_bytes(), before)

            from unittest.mock import patch
            with patch('m4_preflight.nested_policy_dual', side_effect=AssertionError('rerun')):
                resumed = checkpoint_dual_chunk(first, opt, policy, contract=contract)
            np.testing.assert_array_equal(resumed[0], u1)
            with self.assertRaises(ValueError):
                checkpoint_dual_chunk(first, opt, policy,
                                      contract=contract | {'policy_hash': 'changed'})
            u2, l2, z2 = checkpoint_dual_chunk(Path(tmp)/'second.npz', opt, policy,
                contract=contract | {'outer_indices': [3, 4, 5]})
            full = nested_policy_dual(opt, policy, outer=6, seed=773,
                                     budgets=(4, 8), group=2)
            for i, result in enumerate(full):
                u, l = np.concatenate([u1[i], u2[i]]), np.concatenate([l1[i], l2[i]])
                np.testing.assert_array_equal(u, result['upper_samples'])
                np.testing.assert_array_equal(l, result['initial_value_samples'])
                z = np.concatenate([z1[i], z2[i]])
                np.testing.assert_array_equal(z, result['lower_samples'])
                np.testing.assert_array_equal(u-l, result['value_gap_samples'])
                self.assertEqual(_summary(u-l), result['value_gap'])
                self.assertEqual(_summary(u-z), result['gap'])
            prefix = nested_policy_dual(opt, policy, outer=6, seed=773,
                outer_indices=[0, 1, 2], budgets=(4,), group=2)
            np.testing.assert_array_equal(prefix[0]['upper_samples'], u1[0])
            for indices in ([1, 1], [-1, 0], [0., 1.], [5, 6]):
                with self.assertRaises(ValueError):
                    nested_policy_dual(opt, policy, outer=6, outer_indices=indices,
                                       budgets=(4,), group=2)


class ResumePilotHarnessTests(unittest.TestCase):
    def test_pilot_records_both_hashes_and_excludes_only_timing(self):
        import json
        import tempfile
        import numpy as np
        from unittest.mock import patch
        from stopping import Policy
        from m4_preflight import resume_pilot
        policy = Policy('lsm', info={'degree': 1, 'ridge': 1e-6})
        def nested(*args, **kwargs):
            n = len(kwargs['outer_indices'])
            return [dict(upper_samples=np.arange(n, dtype=float)+j,
                         lower_samples=np.arange(n, dtype=float),
                         initial_value_samples=np.arange(n, dtype=float)+.25)
                    for j in range(len(kwargs['budgets']))]
        def unit(seconds):
            return dict(runs=[dict(value=2., train_s=seconds)],
                        paired=[dict(diff=1., paired_se=.1)], dates=[])
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'pilot'
            with patch('m4_preflight.quick_destination', return_value=out), \
                 patch('m4_preflight.gbm_paths', return_value=np.zeros((2, 10, 5))), \
                 patch('m4_preflight.select_lsm', return_value=(policy, None)), \
                 patch('m4_preflight.nested_policy_dual', side_effect=nested), \
                 patch('stopping.run_unit', side_effect=[unit(1.), unit(2.)]), \
                 patch('stopping.reference_for', return_value={'value': 2.}):
                resume_pilot(out)
            report = json.loads((out/'resume_evidence.json').read_text())
            self.assertTrue(report['interruption_observed'])
            for result in [report['study'], *report['dual'].values()]:
                self.assertTrue(result['identical'])
                self.assertEqual(result['resumed_sha256'], result['uninterrupted_sha256'])


class ValidationProbeTests(unittest.TestCase):
    def test_exact_full_sections_and_restart_cost(self):
        import validation_sections  # Import plotting stack before mocking subprocess.run.
        import json
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from m4_preflight import validation_probe
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'probe'
            with patch('m4_preflight.quick_destination', return_value=out), \
                 patch('subprocess.run', return_value=SimpleNamespace(stdout='2.5\n')), \
                 patch('validation_sections.validation_checks') as checks, \
                 patch('validation_sections.validation_put_fit') as put, \
                 patch('validation_sections.validation_convergence') as convergence, \
                 patch('validation_sections.validation_dual') as dual, \
                 patch('stopping.reference_for', return_value={}):
                validation_probe(out)
            report = json.loads((out/'validation_timings.json').read_text())
            self.assertEqual([r['component'] for r in report['components']],
                             ['checks', 'put', 'convergence', 'dual'])
            self.assertEqual(checks.call_count, 1)
            for call in (put, convergence, dual):
                self.assertEqual(call.call_args.args[0], 'full')
                self.assertEqual(call.call_count, 1)
            expected = sum(1.2*(r['seconds']+2.5+10) for r in report['components'])
            self.assertEqual(report['projection']['seconds'], expected)
            self.assertEqual(len(report['units']), 4)
