import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from itertools import combinations, product
from math import comb
from importlib.metadata import version

from bornlens import Sample, compare, shots_per_circuit

STATS = importlib.util.find_spec("scipy") is not None


@unittest.skipUnless(STATS, "requires stats")
class Acceptance(unittest.TestCase):
    def setUp(self):
        from bornlens import advanced
        self.f = advanced
        self.a = Sample("Z", 8, 3)

    def test_invalid_analyses(self):
        f, a = self.f, self.a
        for call in (
            lambda: f.adjust_pvalues([]), lambda: f.adjust_pvalues([.1], "bad"),
            lambda: f.compare_many({}, tolerance=0), lambda: f.sequential_compare(a,a,tolerance=-1),
            lambda: f.paired_compare([],[],tolerance=.1), lambda: f.paired_compare([0],[1],tolerance=.1),
            lambda: f.effect_size(a,Sample("X",2,1),tolerance=.1),
            lambda: f.bayesian_compare(a,a,tolerance=0,prior=(0,1)),
            lambda: f.bootstrap_interval(a,a,confidence=1),
            lambda: f.distribution_compare([],{}), lambda: f.distribution_compare({},{}),
            lambda: f.readout_diagnostics(Sample("X",1,0),Sample("X",1,0)),
            lambda: f.hierarchical_effect([1],[1]),lambda: f.hierarchical_effect([1,2],[1,float("nan")]),
            lambda: shots_per_circuit(0),lambda: compare(a,a,tolerance=3),
            lambda: compare(a,{},tolerance=1),lambda: compare(a,a,tolerance=1,method="unknown"),
        ):
            with self.subTest(call=call), self.assertRaises((ValueError,TypeError)):
                call()
        eye = Sample("I",3,0)
        self.assertEqual(f.sequential_compare(eye,eye,tolerance=0).interval,(0,0))
        self.assertEqual(f.bayesian_compare(eye,eye,tolerance=0)["within_probability"],1)
        self.assertEqual(compare(eye,eye,tolerance=0,method="exact").interval,(0,0))
        from bornlens.statistics import classification_power,plan_classification
        for fn in [lambda:classification_power(-.1,.2,3,4,tolerance=1),
                   lambda:plan_classification(0,1,[],target=3,tolerance=.1)]:
            with self.assertRaises(ValueError):
                fn()

    def test_conditional_enumeration(self):
        a,b = [0,0,0,1],[0,1,1,1,2]
        import numpy as np
        all_labels=np.array(a+b)
        def tv(indices):
            mask=np.zeros(len(all_labels),bool)
            mask[list(indices)]=True
            x,y=np.bincount(all_labels[mask],minlength=3),np.bincount(all_labels[~mask],minlength=3)
            return abs(x/len(a)-y/len(b)).sum()/2
        observed=tv(range(len(a)))
        exact=sum(tv(i)>=observed-1e-14 for i in combinations(range(9),4))/comb(9,4)
        r=self.f.distribution_compare({0:3,1:1,2:0},{0:1,1:3,2:1},draws=30000)
        self.assertAlmostEqual(r["total_variation"],observed)
        self.assertLess(abs(r["pvalue"]-exact),.016)
        diffs=[1,1,1,-1]
        expected=sum(abs(sum(s))>=abs(sum(diffs)) for s in product((-1,1),repeat=4))/16
        self.assertAlmostEqual(self.f.randomization_test([-x for x in diffs],diffs,paired=True),expected)

    def test_beta_reference(self):
        from scipy.integrate import quad
        from scipy.stats import beta
        for x,y in [(0.5,.5),(1,1),(8,2),(.2,5)]:
            a,b=Sample("Z",2,5),Sample("Z",3,7)
            aa,bb=beta(a.plus+x,a.minus+y),beta(b.plus+x,b.minus+y)
            prob=quad(lambda t: aa.pdf(t)*(bb.cdf(min(1,t+.1))-bb.cdf(max(0,t-.1))),0,1,points=[.1,.9])[0]
            r=self.f.bayesian_compare(a,b,tolerance=.2,prior=(x,y),draws=80000)
            self.assertLess(abs(r["within_probability"]-prob),5*r["mc_se"])
            self.assertLess(abs(r["mean"]-2*(bb.mean()-aa.mean())),.008)

    @unittest.skipUnless(importlib.util.find_spec("statsmodels"), "independent library")
    def test_external_references(self):
        import numpy as np
        from statsmodels.stats.multitest import multipletests
        from statsmodels.stats.meta_analysis import combine_effects
        for method in ("bonferroni","holm","fdr_bh","fdr_by"):
            p=[0,.31,.03,.01,.01,1]
            np.testing.assert_allclose(self.f.adjust_pvalues(p,method),multipletests(p,method=method)[1])
        effects, se = np.array([-1.,.4,.1,1.7]),np.array([.3,.2,.1,.4])
        ref=combine_effects(effects,se**2,method_re="dl")
        r=self.f.hierarchical_effect(effects,se)
        np.testing.assert_allclose([r["mean"],r["between_variance"]],[ref.mean_effect_re,ref.tau2])

    def test_json_roundtrip(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"r.json"
            self.f.write_report(p,inputs=[np.int64(3)],settings={},results="ok")
            self.assertEqual(json.loads(p.read_text())["inputs"],[3])
            with self.assertRaises(TypeError):
                self.f.write_report(p,inputs=object(),settings={},results=None)


@unittest.skipUnless(importlib.util.find_spec("shotweave") and importlib.util.find_spec("qiskit"),"published extras")
class Interoperability(unittest.TestCase):
    def test_published_shotweave(self):
        import shotweave as sw
        from qiskit import QuantumCircuit
        from bornlens.qiskit import sample_counts
        from bornlens.shotweave import compare_final
        self.assertEqual(version("shotweave"),"0.2.0")
        circuits=[QuantumCircuit(1),QuantumCircuit(1)]
        circuits[1].ry(.8,0)
        for strategy in ("uniform","adaptive","bayesian","robust","noise-aware"):
            batches=[]
            def submit(shots):
                counts=[int(sample_counts(c,"Z",n,seed=47+10*len(batches)+i).get("1",0))
                        for i,(n,c) in enumerate(zip(shots,circuits))]
                batches.append(counts)
                return counts
            opts={"readout":[sw.Readout(.02,.03)]*2} if strategy=="noise-aware" else {}
            exp=sw.run_experiment(512,[1,1],submit,strategy=strategy,pilot_size=8,**opts)
            self.assertEqual(exp.total_shots,512)
            samples=[Sample("Z",n-k,k) for n,k in zip(exp.final_shots,exp.successes)]
            ref=compare(*samples,tolerance=.2)
            if exp.pilot_shots:
                rows=sw.bornlens_rows("sim-pilot","sim-final",exp.pilot_shots,exp.final_shots,exp.successes)
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/"rows"
                    path.write_text("\n".join(map(json.dumps,rows)))
                    self.assertEqual(compare_final(path,0,1,tolerance=.2),ref)
            self.assertEqual(ref.shots_a+ref.shots_b,sum(exp.final_shots))
