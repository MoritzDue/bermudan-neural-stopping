"""Fixed-seed numerical regression checks; smoke evidence, not study results."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from stopping import *


def european_max(opt):
    """Independent survival-integral formula for independent lognormal assets."""
    mu = np.log(opt.s0) + (opt.rate - opt.dividend - opt.sigma**2 / 2) * opt.maturity
    sd = opt.sigma * np.sqrt(opt.maturity)
    integral = quad(lambda y: -np.expm1(opt.d * np.log(ndtr((np.log(y)-mu)/sd))),
                    opt.strike, np.inf, epsabs=1e-8)[0]
    return np.exp(-opt.rate * opt.maturity) * integral


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        limit_threads(1)

    def test_european_max_never_exercise(self):
        for d in (2, 5):
            o = paper_max_call(d)
            v, tau = discounted_payoffs(o, gbm_paths(o, 150_000, seed=110+d), Policy('never'))
            exact = european_max(o)
            z = (v.mean()-exact)/(v.std(ddof=1)/np.sqrt(len(v)))
            print(f'European {d}D: exact={exact:.8f}, MC={v.mean():.8f}, z={z:.3f}')
            self.assertLess(abs(z), 4)
            self.assertTrue((tau == o.dates).all())

    def test_crr_independent_convolution(self):
        o = validation_put()
        ref = reference_for(o)
        fine = convolution_bermudan_put(o, 0.00025)
        print(f'Put reference: {ref}; finer convolution={fine:.9f}')
        self.assertLess(abs(ref['value']-fine), 5e-4)
        self.assertLess(abs(ref['convolution']-fine), 5e-5)
        self.assertGreaterEqual(ref['grid_band'], abs(ref['value']-fine))

    def test_gbm_moments_correlation_nesting(self):
        o = replace(paper_max_call(3), rho=0.3)
        x = gbm_paths(o, 100_000, seed=91)
        terminal = x[:, -1]
        expected = o.s0 * np.exp((o.rate-o.dividend)*o.maturity)
        np.testing.assert_array_less(abs(terminal.mean(0)-expected), 4*terminal.std(0)/np.sqrt(len(x)))
        returns = np.diff(np.log(x), axis=1).reshape(-1, o.d)
        self.assertLess(abs(np.corrcoef(returns.T)[0,1]-o.rho), 0.01)
        self.assertLess(abs(returns.var()-o.sigma**2*o.dt), 0.0001)
        np.testing.assert_array_equal(gbm_paths(o, 23, seed=3), gbm_paths(o, 100, seed=3)[:23])

    def test_discounting_and_mutation_detection(self):
        o = validation_put()
        x = np.full((2,o.dates+1,1), 90.)
        never = Policy('never')
        first = Policy('first', rules={1: lambda z: np.ones(len(z), bool)})
        v,tau = discounted_payoffs(o,x,never)
        np.testing.assert_allclose(v, 10*np.exp(-o.rate*o.maturity))
        self.assertTrue((tau==o.dates).all())
        def check_early():
            v,tau = discounted_payoffs(o,x,first)
            np.testing.assert_allclose(v, 10*np.exp(-o.rate*o.dt), rtol=1e-12)
            self.assertTrue((tau==1).all())
        check_early()
        bw = Backward(o, x)
        np.testing.assert_allclose(bw.future(2), 10*np.exp(-o.rate*(o.dates-2)*o.dt))
        bw.update(1, np.ones(len(x), bool))
        np.testing.assert_allclose(bw.future(0), 10*np.exp(-o.rate*o.dt))
        v,tau = discounted_payoffs(o,x,Policy('zero', rules={0:lambda z:np.ones(len(z),bool)}))
        self.assertTrue((tau==o.dates).all())
        # Isolated mutation: discount at maturity regardless of actual stopping date.
        with patch.object(Option, 'discount', lambda self,k: np.exp(-self.rate*self.maturity)):
            with self.assertRaises(AssertionError):
                check_early()
        # No t=0 exercise, even with an explicit rule; no OTM exercise.
        v,tau = discounted_payoffs(o,np.full_like(x,110),Policy('all',rules={k:lambda z:np.ones(len(z),bool) for k in range(o.dates)}))
        self.assertTrue((v==0).all())
        self.assertTrue((tau==o.dates).all())

    def test_multidate_threshold_discounting(self):
        # Reviewer-supplied independent Gaussian-kernel policy evaluation.
        from scipy.signal import fftconvolve
        o = validation_put()
        h = 0.00025
        n = int(3/h)
        x = np.arange(-n, n+1)*h
        spot = o.s0*np.exp(x)
        sd = o.sigma*np.sqrt(o.dt)
        drift = (o.rate-o.dividend-o.sigma**2/2)*o.dt
        kernel = np.exp(-0.5*((x+drift)/sd)**2)/(sd*np.sqrt(2*np.pi))*h
        payoff = np.maximum(o.strike-spot, 0)
        value = payoff.copy()
        for k in range(o.dates-1, -1, -1):
            value = np.exp(-o.rate*o.dt)*fftconvolve(value, kernel, mode='same')
            if k > 0:
                value = np.where((spot < 92)&(payoff > 0), payoff, value)
        exact = value[n]
        policy = Policy('threshold', rules={k: lambda z: z[:,0] < 92 for k in range(1,o.dates)})
        paths = gbm_paths(o, 400_000, seed=987654)
        def z_score():
            v, tau = discounted_payoffs(o, paths, policy)
            return (v.mean()-exact)/(v.std(ddof=1)/np.sqrt(len(v))), tau
        z, tau = z_score()
        print(f'Threshold policy: grid={exact:.8f}, z={z:.3f}')
        self.assertLess(abs(z), 4)
        self.assertTrue((np.bincount(tau, minlength=o.dates+1)[1:o.dates] > 0).all())
        mutations = {
            'off-by-one': lambda self,k: np.exp(-self.rate*(np.asarray(k)+1)*self.dt),
            'half-rate': lambda self,k: np.exp(-0.5*self.rate*np.asarray(k)*self.dt),
            'maturity': lambda self,k: np.exp(-self.rate*self.maturity)*np.ones_like(np.asarray(k),float),
        }
        for name, discount in mutations.items():
            with patch.object(Option, 'discount', discount):
                mutated_z, _ = z_score()
            print(f'Threshold mutation {name}: z={mutated_z:.3f}')
            self.assertGreaterEqual(abs(mutated_z), 4)

    def test_backward_multidate_targets(self):
        o = validation_put()
        # Each path first goes ITM at a different date; the third waits to maturity.
        paths = np.full((3, o.dates+1, 1), 110.)
        paths[0, 2, 0] = 90.
        paths[1, 5, 0] = 80.
        paths[2, -1, 0] = 70.
        bw = Backward(o, paths)
        for k in range(o.dates-1, 0, -1):
            bw.update(k, paths[:, k, 0] < 100)
        # Independent calendar-time oracle, with simple per-path cash-flow amounts.
        expected = np.array([10., 20., 30.])*np.exp(-0.05*np.array([2/9, 5/9, 1.]))
        np.testing.assert_allclose(bw.future(0), expected, rtol=1e-12)
        np.testing.assert_allclose(bw.future(1), expected*np.exp(0.05/9), rtol=1e-12)
        with patch.object(Backward, 'future', lambda self,k: self.pay*np.exp(-self.opt.rate*(self.tau-k+1)*self.opt.dt)):
            with self.assertRaises(AssertionError):
                np.testing.assert_allclose(bw.future(0), expected, rtol=1e-12)

    def test_deterministic_limit(self):
        o = replace(validation_put(), s0=80, sigma=1e-6)
        x = gbm_paths(o, 1000, seed=8)
        p = Policy('first',rules={1:lambda z:np.ones(len(z),bool)})
        v,tau = discounted_payoffs(o,x,p)
        expected = o.strike*np.exp(-o.rate*o.dt)-o.s0
        self.assertAlmostEqual(v.mean(), expected, delta=2e-5)
        self.assertTrue((tau==1).all())

    def test_seed_roles(self):
        seeds = [seed_for(2026,r,d,s,e) for r in ROLES for d in (1,2,5) for s in (0,1,100,500) for e in (0,1)]
        self.assertEqual(len(seeds),len(set(seeds)))

    def test_put_fitted_policies(self):
        o=validation_put()
        train=gbm_paths(o,10_000,seed=501)
        val=gbm_paths(o,5_000,seed=502)
        test=gbm_paths(o,200_000,seed=503)
        lsm,_=select_lsm(o,train,val)
        nn=fit_neural(o,train,val,NeuralSettings(max_steps=1500,lr=1e-3,batch=512),seed=504)
        ref=reference_for(o)
        for p in (lsm,nn):
            v,_=discounted_payoffs(o,test,p)
            se=v.std(ddof=1)/np.sqrt(len(v))
            print(f'Put {p.method}: {v.mean():.8f}, SE={se:.8f}, gap={ref["value"]-v.mean():.8f}')
            self.assertLessEqual(v.mean(),ref['value']+ref['grid_band']+3*se)
            self.assertLessEqual(ref['value']-v.mean(),0.10)

    def test_no_dividend_lsm_adversarial(self):
        o=replace(paper_max_call(5),dividend=0.)
        train=gbm_paths(o,10_000,seed=601)
        val=gbm_paths(o,5_000,seed=602)
        test=gbm_paths(o,200_000,seed=603)
        p,_=select_lsm(o,train,val)
        v,_=discounted_payoffs(o,test,p)
        never,_=discounted_payoffs(o,test,Policy('never'))
        exact=european_max(o)
        se=v.std(ddof=1)/np.sqrt(len(v))
        delta=v-never
        paired_se=delta.std(ddof=1)/np.sqrt(len(delta))
        print(f'No dividend: exact={exact:.8f}, LSM={v.mean():.8f}, SE={se:.8f}, paired loss={delta.mean():.8f} +/- {1.96*paired_se:.8f}')
        self.assertLessEqual(v.mean(),exact+3*se)
        self.assertLessEqual(delta.mean()+1.96*paired_se,0.)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_result_roundtrip_tail_and_namespace(self):
        folder=self.root/'study'
        unit=dict(runs=[dict(value=1)],paired=[],dates=[])
        s=ResultStore(folder,dict(frozen={'a':1}),'quick')
        s.append(unit,('case',1,0))
        with s.path.open('ab') as f: f.write(b'{"unfinished":')
        s.append(unit,('case',2,0))
        self.assertEqual(len(s.done()),2)
        self.assertEqual(len(s.load('runs')),2)
        old=s.path.read_bytes()
        other=ResultStore(folder,dict(frozen={'a':2}),'quick')
        self.assertNotEqual(s.folder,other.folder)
        self.assertEqual(s.path.read_bytes(),old)
        self.assertEqual(json.loads((folder/'run_info.json').read_text())['frozen'],{'a':1})
        self.assertEqual(other.done(),set())
        self.assertEqual(ResultStore(folder,dict(frozen={'a':2}),'quick').folder,other.folder)
        with self.assertRaises(RuntimeError): ResultStore(folder,dict(frozen={'a':3}),'full')
        # Valid final record without newline survives repair.
        s.path.write_bytes(old.rstrip(b'\n'))
        s.append(unit,('case',3,0))
        self.assertEqual(len(s.done()),3)

    def test_calibration_provenance(self):
        legacy=self.root/'calibration.csv'
        legacy.write_text('old evidence\n')
        path=calibration_path(self.root,'quick')
        self.assertNotEqual(path,legacy)
        self.assertEqual(calibration_path(self.root,'quick'),path)
        path.write_text('current evidence\n')
        with patch.dict(CONFIG['quick'],base_seed=100):
            changed=calibration_path(self.root,'quick')
            self.assertNotEqual(changed,path)
        with patch('stopping.code_hash',return_value='changed'):
            self.assertNotEqual(calibration_path(self.root,'quick'),path)
        self.assertEqual(legacy.read_text(),'old evidence\n')
        self.assertNotEqual(calibration_path(self.root,'full'),path)
        with patch('stopping.calibration_provenance', return_value={'environment': 'changed'}):
            self.assertNotEqual(calibration_path(self.root,'quick'),path)
        meta=path.with_suffix('.json')
        meta.write_text('{}')
        with self.assertRaises(RuntimeError): calibration_path(self.root,'quick')

    def test_results_root_override(self):
        with patch.dict(os.environ,BERMUDAN_RESULTS_ROOT=str(self.root)):
            self.assertEqual(results_dir('quick'), self.root/'quick')
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(results_dir('quick'), PROJECT_ROOT/'results'/'quick')
        with self.assertRaises(ValueError): results_dir('typo')

    def test_reference_fields(self):
        for d in (2,5):
            r=reference_for(paper_max_call(d))
            self.assertAlmostEqual(r['midpoint'],(r['lower']+r['upper'])/2)
            self.assertEqual(r['value'],r['midpoint'])
        self.assertEqual(PAPER_REFERENCES[2]['lattice'],13.902)
        with self.assertRaises(ValueError): reference_for(paper_max_call(10))


class SharedTestUncertaintyTests(unittest.TestCase):
    def test_perfectly_correlated_test_errors_do_not_shrink_with_seeds(self):
        from stopping import absolute_gap_uncertainty
        r = absolute_gap_uncertainty([0.1]*10, [0.04]*10)
        self.assertAlmostEqual(r['shared_test_se_bound'], 0.04)
        self.assertAlmostEqual(r['combined_ci_high']-0.1, 2.241402727604947*0.04)
        # Bound dominates the exact SE for unequal scales under any correlation.
        scales = np.array([0.02, 0.05, 0.08])
        covariance = np.outer(scales, scales) * np.array([[1,.3,-.1],[.3,1,.2],[-.1,.2,1]])
        exact = np.sqrt(covariance.sum())/3
        self.assertGreaterEqual(absolute_gap_uncertainty([1,2,3], scales)['shared_test_se_bound'], exact)
        with self.assertRaises(ValueError): absolute_gap_uncertainty([1], [0.1])
        with self.assertRaises(ValueError): absolute_gap_uncertainty([1,2], [-.1,.1])


if __name__=='__main__':
    unittest.main(verbosity=2)
