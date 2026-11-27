import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from bornlens import Sample, compare


@unittest.skipUnless(importlib.util.find_spec("scipy"), "stats extra")
class AdvancedTests(unittest.TestCase):
    def setUp(self):
        from bornlens import advanced
        self.api = advanced
        self.a = Sample("Z",1000,0)
        self.b = Sample("Z",0,1000)

    def test_family_and_regression(self):
        pairs = {"Z":(self.a,self.b),"X":(Sample("X",500,500),Sample("X",500,500))}
        r = self.api.regression_check(pairs,tolerance=.2)
        self.assertEqual(r["status"],"fail")
        self.assertEqual(r["observables"]["X"].confidence,.975)
        for tol,status in [(2,"pass"),(0,"inconclusive")]:
            self.assertEqual(self.api.regression_check({"Z":(self.a,self.a)},tolerance=tol)["status"],status)
        with self.assertRaises(ValueError):
            self.api.compare_many({"X":(self.a,self.b)},tolerance=.2)

    def test_adjustments(self):
        import numpy as np
        p = [.04,.01,.03]
        for method,expected in [("bonferroni",[.12,.03,.09]),("holm",[.06,.03,.06]),
                                ("fdr_bh",[.04,.03,.04]),("fdr_by",np.array([.04,.03,.04])*11/6)]:
            np.testing.assert_allclose(self.api.adjust_pvalues(p,method),expected)

    def test_sequential(self):
        q = self.api.sequential_compare(self.a,self.b,tolerance=.1)
        self.assertEqual(q.verdict,"beyond_tolerance")
        self.assertLess(compare(self.a,self.b,tolerance=.1).interval[1],q.interval[1])
        with self.assertRaises(ValueError):
            self.api.sequential_compare(self.a,Sample("Z",1,0),tolerance=.1)

    def test_bayes_and_bootstrap(self):
        r = self.api.bayesian_compare(self.a,self.b,tolerance=.1,seed=4)
        self.assertEqual(r["within_probability"],0)
        self.assertTrue(-2 < r["credible_interval"][0] < r["credible_interval"][1] < -1.9)
        self.assertEqual(self.api.bootstrap_interval(self.a,self.b),(-2,-2))
        self.assertEqual(self.api.bootstrap_interval([1,-1],[1,-1],paired=True),(0,0))

    def test_paired_and_randomization(self):
        a,b = [1]*10,[-1]*10
        self.assertEqual(self.api.paired_compare(a,b,tolerance=.1)["delta"],-2)
        self.assertAlmostEqual(self.api.randomization_test(a,b,paired=True),2/1024)
        self.assertEqual(self.api.randomization_test(a,a,paired=True),1)
        self.assertAlmostEqual(self.api.randomization_test(Sample("X",3,0),Sample("X",0,3)),.1)

    def test_distributions(self):
        f = self.api.distribution_compare
        self.assertEqual(f({"00":4},{"00":11})["pvalue"],1)
        r = f({"01":30},{"10":40},seed=4,draws=499)
        self.assertEqual(r["total_variation"],1)
        self.assertEqual(r["pvalue"],1/500)
        self.assertEqual(r,f({"01":30},{"10":40},seed=4,draws=499))

    def test_noise_and_drift(self):
        r = self.api.readout_diagnostics(self.a,self.b)
        self.assertEqual(r["contrast"],1)
        self.assertGreater(r["zero_to_one"]["interval"][1],0)
        history = self.api.drift_monitor(self.a,[self.a,self.b])
        self.assertEqual([h["alarm"] for h in history],[False,True])
        self.assertAlmostEqual(history[1]["threshold"],.05/6)

    def test_design_and_effect(self):
        e = self.api.effect_size(self.a,self.b,tolerance=.1)
        self.assertEqual(e["margin_excess"],1.9)
        self.assertEqual(e["risk_difference"],-1)
        design,p = self.api.optimize_design(0,1,[(1,1),(100,100),(10,10)],budget=20,tolerance=.1)
        self.assertEqual(design,(10,10))
        self.assertAlmostEqual(p.correct,1)

    def test_hierarchy_and_report(self):
        f = self.api.hierarchical_effect
        self.assertEqual(f([.2,.2],[.1,.1])["between_variance"],0)
        r = f([-1,1],[.1,.1])
        self.assertAlmostEqual(r["mean"],0)
        self.assertAlmostEqual(r["between_variance"],1.99)
        self.assertTrue(0<r["shrunk_effects"][1]<1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"report.json"
            self.api.write_report(path,inputs={"a":self.a,"b":self.b},settings={"tolerance":.1},
                                  results=compare(self.a,self.b,tolerance=.1))
            result = json.loads(path.read_text())
            self.assertEqual(result["inputs"]["a"]["plus"],1000)
            self.assertEqual(result["results"]["verdict"],"beyond_tolerance")
            self.assertIn("scipy",result["versions"])
