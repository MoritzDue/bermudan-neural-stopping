"""Deterministic checks for dual-validation reference extraction and uncertainty accounting."""
import unittest
import numpy as np
from stopping import validation_put, crr_bermudan_put
from dual_validation import (crr_boundary_policy, independent_gap,
                             budget_diagnostic, terminal_drift_decomposition)
from dual import policy_dual, _summary
from stopping import Policy


from dual_validation import grid_inner_error


class ValidationTests(unittest.TestCase):
    def test_boundary_matches_tree(self):
        o=validation_put()
        p,b,v=crr_boundary_policy(o,80)
        self.assertAlmostEqual(v,crr_bermudan_put(o,80),places=10)
        self.assertEqual(set(b),set(range(1,o.dates)))
        self.assertTrue(all(0<b[k]<o.strike for k in b))
        for k in b:
            np.testing.assert_array_equal(p.rules[k](np.array([[b[k]-.1],[b[k]+.1]])),[True,False])

    def test_independent_gap_error(self):
        u=np.array([1.,2.,3.]); l=np.array([0.,2.,4.,6.])
        r=independent_gap(u,l)
        self.assertAlmostEqual(r['mean'],-1.)
        self.assertAlmostEqual(r['se']**2,u.var(ddof=1)/3+l.var(ddof=1)/4)

    def test_budget_gate_rejects_large_paired_drop(self):
        x = np.array([1., 2., 3., 4.])
        current = dict(upper_samples=x, upper=_summary(x))
        previous = dict(upper_samples=x+.03, upper=_summary(x+.03))
        r = budget_diagnostic(previous, current, 800, 1600)
        self.assertFalse(r['stable'])
        self.assertAlmostEqual(r['residual_bias_c_over_n'], .03)
        with self.assertRaises(ValueError):
            budget_diagnostic(previous, current, 800, 3200)

    def test_terminal_decomposition_reports_rare_bins(self):
        o = validation_put()
        r = policy_dual(o, Policy('never'), outer=100, inner=16, seed=940)
        rows = terminal_drift_decomposition(o, r)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(not row['total_drift_normal_testable'] for row in rows))
        for row in rows:
            self.assertGreaterEqual(row['expected_positive_payoffs'], 0.)
            self.assertLessEqual(row['expected_positive_payoffs'], 20.)
            self.assertTrue(np.isfinite(row['inner_error']['mean']))

    def test_grid_oracle_last_date_against_black_scholes(self):
        from scipy.stats import norm
        o = validation_put()
        r = policy_dual(o, Policy('never'), outer=100, inner=16, seed=940)
        paths, error = grid_inner_error(o, Policy('never'), r, .0005)
        k = o.dates-1
        spot = paths[:, k, 0]
        sd = o.sigma*np.sqrt(o.dt)
        d1 = (np.log(spot/o.strike)+(o.rate-o.dividend+o.sigma**2/2)*o.dt)/sd
        c = o.discount(k)*(o.strike*np.exp(-o.rate*o.dt)*norm.cdf(-d1+sd)
                          -spot*np.exp(-o.dividend*o.dt)*norm.cdf(-d1))
        terminal = o.payoff(paths[:, -1])*o.discount(o.dates)
        exact_error = np.diff(r['martingale_samples'], axis=1)[:, k]-terminal+c
        np.testing.assert_allclose(error[:, k], exact_error, atol=1e-4, rtol=0)


class NestedCouplingTests(unittest.TestCase):
    def test_prefix_and_independent_outer(self):
        from dual import nested_policy_dual
        o = validation_put(); p = Policy('never')
        a = nested_policy_dual(o,p,budgets=(8,16),outer=20,group=8,seed=948)
        b = nested_policy_dual(o,p,budgets=(8,),outer=20,group=8,seed=948)
        np.testing.assert_array_equal(a[0]['upper_samples'], b[0]['upper_samples'])
        np.testing.assert_array_equal(a[0]['martingale_samples'], b[0]['martingale_samples'])
        self.assertNotEqual(a[0]['metadata']['outer_seed'], a[0]['metadata']['inner_seed'])
        self.assertFalse(np.array_equal(a[0]['upper_samples'],a[1]['upper_samples']))
        # Recover C_hat_0 by telescoping the never-exercise martingale.
        from dual import continuation
        from stopping import gbm_paths
        paths = gbm_paths(o,20,seed=a[1]['metadata']['outer_seed'])
        manual = []
        for g in range(2):
            rng = np.random.default_rng(np.random.SeedSequence(
                [a[1]['metadata']['inner_seed'],0,0,g]))
            manual.append(continuation(o,paths[0:1,0],0,p,8,rng)[0])
        terminal = o.payoff(paths[0:1,-1])[0]*o.discount(o.dates)
        recovered = terminal-a[1]['martingale_samples'][0,-1]
        self.assertAlmostEqual(recovered,np.mean(manual),places=11)
        with self.assertRaises(ValueError):
            nested_policy_dual(o,p,budgets=(8,12),group=8)

    def test_nested_deterministic_martingale(self):
        from dataclasses import replace
        from dual import nested_policy_dual
        o = replace(validation_put(), sigma=0.)
        with self.assertRaises(TimeoutError):
            nested_policy_dual(o,Policy('never'),budgets=(4,),outer=10,group=4,max_seconds=1e-15)
        r = nested_policy_dual(o,Policy('never'),budgets=(4,8),outer=10,group=4)
        for v in r:
            np.testing.assert_allclose(v['martingale_samples'],0.,atol=1e-12)
        with self.assertRaises(ValueError):
            nested_policy_dual(o,Policy('never'),budgets=(4,),outer=10,group=4,memory_mb=.0001)


if __name__=='__main__': unittest.main()


class CumulativeSizingTests(unittest.TestCase):
    def test_persistent_small_drops_and_largest_rejected(self):
        from dual_validation import cumulative_sizing_rows
        x=np.array([-.02,0.,.02])
        results=[dict(metadata=dict(inner=n),upper=_summary(x+b),upper_samples=x+b)
                 for n,b in zip((800,1600,3200,6400),(0.024,0.016,0.008,0.))]
        rows=cumulative_sizing_rows(results)
        self.assertTrue(rows[0]['adjacent_gate'])
        self.assertFalse(rows[0]['cumulative_gate'])
        self.assertFalse(rows[-1]['eligible'])
        self.assertTrue(rows[-2]['eligible'])
        self.assertAlmostEqual(rows[0]['cumulative_drop']['mean'],.024)
        self.assertLess(rows[0]['cumulative_drop']['se'],1e-16)

    def test_invalid_cumulative_comparisons_rejected(self):
        from dual_validation import cumulative_sizing_rows
        def result(n, size=3):
            x = np.arange(size, dtype=float)
            return dict(metadata=dict(inner=n), upper=_summary(x), upper_samples=x)
        for results in ([], [result(800)], [result(1600),result(800)],
                        [result(800),result(3200)], [result(800),result(1600,4)]):
            with self.assertRaises(ValueError):
                cumulative_sizing_rows(results)

    def test_followup_rejects_nonquick_output_before_writing(self):
        from dual_validation import run_cumulative_followup
        with self.assertRaises(ValueError):
            run_cumulative_followup(preflight_only=True, output_dir='results/full/forbidden')

class BlockCheckpointTests(unittest.TestCase):
    def test_resume_and_pairing(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from dual_validation import checkpoint_sizing
        calls=[]
        def fake(option,policy,**kw):
            calls.append(kw['seed'])
            x=np.arange(kw['outer'],dtype=float)
            return [dict(upper_samples=x+1/n) for n in kw['budgets']]
        manifest=dict(evaluation_seed=96301,outer=12,group=8,budgets=[8,16])
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with patch('dual.nested_policy_dual',side_effect=fake):
                partial=checkpoint_sizing(None,[('lsm',None),('neural',None)],manifest,out,work_seconds=0)
                self.assertEqual(partial['blocks_remaining'],8)
                result=checkpoint_sizing(None,[('lsm',None),('neural',None)],manifest,out)
                self.assertEqual(result['blocks_remaining'],0)
                checkpoint_sizing(None,[('lsm',None),('neural',None)],manifest,out)
            self.assertEqual(len(calls),8)
            self.assertEqual(len(set(calls)),8)
            rows=__import__('json').loads((out/'followup-lsm.json').read_text())['rows']
            self.assertAlmostEqual(rows[0]['adjacent_drop']['mean'],1/16)
            self.assertFalse(rows[-1]['eligible'])

    def test_interruption_retains_completed_blocks(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from dual_validation import checkpoint_sizing
        manifest=dict(evaluation_seed=96301,outer=12,group=8,budgets=[8,16])
        good=[dict(upper_samples=np.arange(3.))]*2
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with patch('dual.nested_policy_dual',side_effect=[good,KeyboardInterrupt()]):
                r=checkpoint_sizing(None,[('lsm',None),('neural',None)],manifest,out)
            self.assertEqual(r['blocks_remaining'],7)
            self.assertTrue((out/'lsm-block-0.npz').exists())
            self.assertEqual(len(list(out.glob('interruption-*.json'))),1)

    def test_sigterm_records_signal_and_restores_handler(self):
        import json
        import signal
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from dual_validation import checkpoint_sizing
        manifest = dict(evaluation_seed=96301, outer=12, group=8, budgets=[8,16])
        original = signal.getsignal(signal.SIGTERM)
        def terminate(*args, **kwargs):
            signal.raise_signal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch('dual.nested_policy_dual', side_effect=terminate):
                result = checkpoint_sizing(None, [('lsm',None),('neural',None)], manifest, out)
            self.assertEqual(result['blocks_remaining'], 8)
            records = list(out.glob('interruption-*.json'))
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0].read_text())['signal'], [signal.SIGTERM])
            self.assertEqual(signal.getsignal(signal.SIGTERM), original)
            self.assertEqual(len(list(out.glob('invocation-*.json'))), 1)
