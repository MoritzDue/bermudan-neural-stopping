"""Small fixed-seed correctness experiments, not full research evidence."""
import unittest
from dataclasses import replace
import numpy as np
from stopping import *
from dual import policy_dual
from tests.test_stopping import european_max


class DualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        limit_threads(1)

    def test_deterministic_and_perfect_foresight_control(self):
        o = replace(validation_put(), s0=80., sigma=0.)
        p = Policy('first', rules={k:lambda x:np.ones(len(x),bool) for k in range(1,o.dates)})
        r = policy_dual(o,p,outer=40,inner=10)
        exact = 100*np.exp(-.05/9)-80
        np.testing.assert_allclose(r['upper_samples'],exact,atol=1e-12)
        np.testing.assert_allclose(r['lower_samples'],exact,atol=1e-12)
        # Stochastic negative control: perfect foresight is not this dual.
        o = validation_put()
        r = policy_dual(o,Policy('never'),outer=1200,inner=200,seed=77)
        paths = gbm_paths(o,1200,seed=r['metadata']['outer_seed'])
        foresight = (o.payoff(paths[:,1:])*o.discount(np.arange(1,10))).max(1)
        difference = foresight-r['upper_samples']
        print('Perfect foresight minus nested dual:', difference.mean(), 'SE',difference.std(ddof=1)/np.sqrt(len(difference)))
        self.assertGreater(difference.mean(),3*difference.std(ddof=1)/np.sqrt(len(difference)))

    def test_no_dividend_inner_sweep_and_drift(self):
        o = replace(paper_max_call(5), dividend=0.)
        exact = european_max(o)
        rows=[]
        for n in (32,512):
            r = policy_dual(o,Policy('never'),outer=1200,inner=n,seed=42)
            u=r['upper']; rows.append(u)
            print('No dividend inner',n,'U',u,'exact',exact)
            self.assertGreaterEqual(u['mean']+3*u['se'],exact)
            m=r['martingale']
            self.assertTrue(np.all(abs(m['mean'])<=4*m['se']+1e-12))
        self.assertLess(abs(rows[1]['mean']-exact),abs(rows[0]['mean']-exact))

    def test_bad_policies(self):
        o=validation_put(); ref=reference_for(o)['value']
        for p in (Policy('never'),Policy('immediate',rules={k:lambda x:np.ones(len(x),bool) for k in range(1,o.dates)})):
            r=policy_dual(o,p,outer=1500,inner=200,seed=3)
            print('Bad policy',p.method,'U',r['upper'],'gap',r['gap'])
            self.assertGreaterEqual(r['upper']['mean']+3*r['upper']['se'],ref)
            self.assertGreater(r['gap']['mean'],0.)

    def test_seeds_reproduction_and_memory_cap(self):
        seeds=[seed_for(2026,r,d,s,e) for r in ROLES for d in (1,2,5) for s in (0,1,100) for e in (0,1)]
        self.assertEqual(len(seeds),len(set(seeds)))
        o=validation_put(); p=Policy('never')
        a=policy_dual(o,p,outer=20,inner=10,outer_chunk=3)
        b=policy_dual(o,p,outer=20,inner=10,outer_chunk=3)
        np.testing.assert_array_equal(a['upper_samples'],b['upper_samples'])
        with self.assertRaises(ValueError): policy_dual(o,p,memory_mb=1e-10)
        self.assertLess(a['metadata']['estimated_chunk_bytes'],256*1024**2)


if __name__=='__main__': unittest.main(verbosity=2)
