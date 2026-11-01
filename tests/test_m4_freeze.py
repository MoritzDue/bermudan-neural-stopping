"""Manifest checks independent of expensive timing measurements."""
import unittest
from m4_preflight import LADDER_VARIANTS, ladder_manifest
from stopping import VARIANTS, NeuralSettings, with_variant


class ManifestTests(unittest.TestCase):
    def test_cumulative_ladder_and_primary_pair(self):
        unit={}
        dual={}
        for d,b in ((5,1000),(5,10000),(5,50000),(2,10000)):
            variants=['sorted+payoff']
            if d==5 and b!=10000:
                variants+=['raw','raw+payoff','sorted']
            for v in variants:
                unit[d,b,v]=10.
            for v in ['lsm']+variants:
                for coupled in (False,True):
                    dual[d,b,v,coupled]=30. if coupled else 20.
        manifest,costs=ladder_manifest(unit,dual,100.)
        names = {u['variant'] for u in manifest if u['kind']=='study'}
        names.update(u['method'] for u in manifest if u['kind']=='dual' and u['method']!='lsm')
        self.assertEqual(names, set(LADDER_VARIANTS))
        for name in names:
            self.assertIn(name, VARIANTS)
            with_variant(NeuralSettings(max_steps=10, lr=1e-3, batch=8), name)
        for unit_row in manifest:
            if unit_row['kind']=='study':
                self.assertEqual(unit_row['produces'], ['lsm', unit_row['variant']])
        self.assertEqual(len({u['id'] for u in manifest}),len(manifest))
        self.assertEqual([sum(u['kind']=='study' and u['level']<=l for u in manifest)
                          for l in range(6)],[10,20,30,90,100,100])
        primary=[u for u in manifest if u['kind']=='dual' and u['level']==0]
        self.assertEqual(len(primary),50)
        extras=[u for u in manifest if u['kind']=='dual' and u['level']==5]
        self.assertEqual(len(extras),24*20)
        self.assertTrue(all(u['inner']==[3200] for u in extras))
        for method in ('lsm','sorted+payoff'):
            for inner,n in (([3200],2000),([3200,6400],500)):
                rows=[u for u in primary if u['method']==method and u['inner']==inner]
                self.assertEqual([i for u in rows for i in u['outer_indices']],list(range(n)))
                self.assertEqual(len({u['rng_seed'] for u in rows}),1)
        self.assertEqual(costs,[sum(u['projected_s'] for u in manifest if u['level']<=l)
                                for l in range(6)])
        self.assertTrue(all(b>=a for a,b in zip(costs,costs[1:])))


if __name__=='__main__':
    unittest.main()
