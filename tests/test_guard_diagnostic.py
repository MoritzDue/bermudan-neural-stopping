"""Independent hand-computed checks of the development guard diagnostic."""
import unittest
import numpy as np
from stopping import Option, Policy, guard_diagnostic


class GuardDiagnosticTests(unittest.TestCase):
    def test_guarded_suffix_and_discounting(self):
        opt = Option(kind="put", d=1, strike=100, dates=3, maturity=3, rate=0.1)
        # Date 2 rule is rejected: continuation from date 1 must use maturity.
        paths = np.array([[[100], [80], [70], [90]],
                          [[100], [90], [60], [80]]], dtype=float)
        pol = Policy("toy", rules={1: lambda x: np.ones(len(x), bool),
                                   2: lambda x: np.ones(len(x), bool)},
                     guard={1: False, 2: True}, val_gain={1: 0.2, 2: -0.1})
        rows = {r['date']: r for r in guard_diagnostic(opt, paths, pol)}
        g1 = np.array([20, 10])*np.exp(-0.1) - np.array([10, 20])*np.exp(-0.3)
        g2 = np.array([30, 40])*np.exp(-0.2) - np.array([10, 20])*np.exp(-0.3)
        self.assertAlmostEqual(rows[1]['fresh_gain'], g1.mean())
        self.assertAlmostEqual(rows[1]['fresh_se'], g1.std(ddof=1)/np.sqrt(2))
        self.assertAlmostEqual(rows[2]['fresh_gain'], g2.mean())
        self.assertEqual(rows[2]['retained_fresh_gain'], 0)
        self.assertEqual(pol.guard, {1: False, 2: True})
        self.assertEqual(pol.val_gain, {1: 0.2, 2: -0.1})

    def test_no_rule_and_out_of_money(self):
        opt = Option(kind="put", d=1, dates=2)
        paths = np.full((3, 3, 1), 110.)
        pol = Policy("toy", rules={1: lambda x: np.ones(len(x), bool)})
        row = guard_diagnostic(opt, paths, pol)[0]
        self.assertEqual(row['fresh_gain'], 0)
        self.assertEqual(row['fresh_stop_share'], 0)
        row = guard_diagnostic(opt, paths, Policy("never"))[0]
        self.assertEqual(row['fresh_gain'], 0)
        with self.assertRaises(ValueError):
            guard_diagnostic(opt, paths[:1], pol)
