"""Fixed-seed estimator and smoke-fallback regression checks."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import dual
from stopping import (Option, Policy, CONFIG, NeuralSettings, asdict,
                      gbm_paths, discounted_payoffs, load_study_settings)


class M4EstimatorTests(unittest.TestCase):
    def test_initial_value_identity_and_independent_policy_mean(self):
        opt = Option(kind='put', d=1, dates=3)
        policy = Policy('never')
        calls = []
        original = dual.continuation
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            if args[2] == 0:
                calls.append(float(result[0]))
            return result
        with patch('dual.continuation', side_effect=capture):
            r = dual.nested_policy_dual(opt, policy, budgets=(64,),
                                       outer=400, group=64, seed=881)[0]
        # Each captured rollout already discounts to zero: no extra discount.
        np.testing.assert_array_equal(r['initial_value_samples'], calls)
        np.testing.assert_array_equal(r['value_gap_samples'],
                                     r['upper_samples']-np.array(calls))
        fresh = discounted_payoffs(opt, gbm_paths(opt, 30000, seed=99181), policy)[0]
        initial = r['initial_value_samples']
        se = np.sqrt(initial.var(ddof=1)/len(initial)+fresh.var(ddof=1)/len(fresh))
        self.assertLess(abs(initial.mean()-fresh.mean()), 3*se)
        difference = (r['upper_samples']-r['lower_samples'])-r['value_gap_samples']
        self.assertLess(abs(difference.mean()), 3*difference.std(ddof=1)/np.sqrt(len(difference)))

    def test_stale_fallback_is_quick_only_and_preserves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'frozen_settings.json'
            path.write_text(json.dumps(dict(provenance={'old': True}, neural={})))
            before = path.read_bytes()
            with patch('stopping.calibration_provenance', return_value={'current': True}):
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    result = load_study_settings(path, 'quick')
                self.assertIn('WARNING', stream.getvalue())
                self.assertEqual(result['neural'], asdict(NeuralSettings(**CONFIG['quick']['validation']['nn_settings'])))
                self.assertEqual(result['provenance'], {'current': True})
                self.assertIn('CONFIG', result['settings_source'])
                with self.assertRaises(RuntimeError):
                    load_study_settings(path, 'full')
            self.assertEqual(before, path.read_bytes())
