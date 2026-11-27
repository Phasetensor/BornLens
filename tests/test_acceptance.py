import importlib.util
import json
import tempfile
import unittest
from importlib.metadata import version
from itertools import combinations, product
from math import comb
from pathlib import Path

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
            lambda: f.adjust_pvalues([]),
            lambda: f.adjust_pvalues([0.1], "bad"),
            lambda: f.compare_many({}, tolerance=0),
            lambda: f.sequential_compare(a, a, tolerance=-1),
            lambda: f.paired_compare([], [], tolerance=0.1),
            lambda: f.paired_compare([0], [1], tolerance=0.1),
            lambda: f.effect_size(a, Sample("X", 2, 1), tolerance=0.1),
            lambda: f.bayesian_compare(a, a, tolerance=0, prior=(0, 1)),
            lambda: f.bootstrap_interval(a, a, confidence=1),
            lambda: f.distribution_compare([], {}),
            lambda: f.distribution_compare({}, {}),
            lambda: f.readout_diagnostics(Sample("X", 1, 0), Sample("X", 1, 0)),
            lambda: f.hierarchical_effect([1], [1]),
            lambda: f.hierarchical_effect([1, 2], [1, float("nan")]),
            lambda: shots_per_circuit(0),
            lambda: compare(a, a, tolerance=3),
            lambda: compare(a, {}, tolerance=1),
            lambda: compare(a, a, tolerance=1, method="unknown"),
        ):
            with self.subTest(call=call), self.assertRaises((ValueError, TypeError)):
                call()
        eye = Sample("I", 3, 0)
        self.assertEqual(f.sequential_compare(eye, eye, tolerance=0).interval, (0, 0))
        self.assertEqual(
            f.bayesian_compare(eye, eye, tolerance=0)["within_probability"], 1
        )
        self.assertEqual(
            compare(eye, eye, tolerance=0, method="exact").interval, (0, 0)
        )
        from bornlens.statistics import classification_power, plan_classification

        for fn in [
            lambda: classification_power(-0.1, 0.2, 3, 4, tolerance=1),
            lambda: plan_classification(0, 1, [], target=3, tolerance=0.1),
        ]:
            with self.assertRaises(ValueError):
                fn()

    def test_conditional_enumeration(self):
        a, b = [0, 0, 0, 1], [0, 1, 1, 1, 2]
        import numpy as np

        all_labels = np.array(a + b)

        def tv(indices):
            mask = np.zeros(len(all_labels), bool)
            mask[list(indices)] = True
            x, y = (
                np.bincount(all_labels[mask], minlength=3),
                np.bincount(all_labels[~mask], minlength=3),
            )
            return abs(x / len(a) - y / len(b)).sum() / 2

        observed = tv(range(len(a)))
        exact = sum(
            tv(i) >= observed - 1e-14 for i in combinations(range(9), 4)
        ) / comb(9, 4)
        r = self.f.distribution_compare(
            {0: 3, 1: 1, 2: 0}, {0: 1, 1: 3, 2: 1}, draws=30000
        )
        self.assertAlmostEqual(r["total_variation"], observed)
        self.assertLess(abs(r["pvalue"] - exact), 0.016)
        diffs = [1, 1, 1, -1]
        expected = (
            sum(abs(sum(s)) >= abs(sum(diffs)) for s in product((-1, 1), repeat=4)) / 16
        )
        self.assertAlmostEqual(
            self.f.randomization_test([-x for x in diffs], diffs, paired=True), expected
        )

    def test_beta_reference(self):
        from scipy.integrate import quad
        from scipy.stats import beta

        for x, y in [(0.5, 0.5), (1, 1), (8, 2), (0.2, 5)]:
            a, b = Sample("Z", 2, 5), Sample("Z", 3, 7)
            aa, bb = beta(a.plus + x, a.minus + y), beta(b.plus + x, b.minus + y)
            prob = quad(
                lambda t: (
                    aa.pdf(t) * (bb.cdf(min(1, t + 0.1)) - bb.cdf(max(0, t - 0.1)))
                ),
                0,
                1,
                points=[0.1, 0.9],
            )[0]
            r = self.f.bayesian_compare(a, b, tolerance=0.2, prior=(x, y), draws=80000)
            self.assertLess(abs(r["within_probability"] - prob), 5 * r["mc_se"])
            self.assertLess(abs(r["mean"] - 2 * (bb.mean() - aa.mean())), 0.008)

    @unittest.skipUnless(importlib.util.find_spec("statsmodels"), "independent library")
    def test_external_references(self):
        import numpy as np
        from statsmodels.stats.meta_analysis import combine_effects
        from statsmodels.stats.multitest import multipletests

        for method in ("bonferroni", "holm", "fdr_bh", "fdr_by"):
            p = [0, 0.31, 0.03, 0.01, 0.01, 1]
            np.testing.assert_allclose(
                self.f.adjust_pvalues(p, method), multipletests(p, method=method)[1]
            )
        effects, se = np.array([-1.0, 0.4, 0.1, 1.7]), np.array([0.3, 0.2, 0.1, 0.4])
        ref = combine_effects(effects, se**2, method_re="dl")
        r = self.f.hierarchical_effect(effects, se)
        np.testing.assert_allclose(
            [r["mean"], r["between_variance"]], [ref.mean_effect_re, ref.tau2]
        )

    def test_json_roundtrip(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.json"
            self.f.write_report(p, inputs=[np.int64(3)], settings={}, results="ok")
            self.assertEqual(json.loads(p.read_text())["inputs"], [3])
            with self.assertRaises(TypeError):
                self.f.write_report(p, inputs=object(), settings={}, results=None)


@unittest.skipUnless(
    importlib.util.find_spec("shotweave") and importlib.util.find_spec("qiskit"),
    "published extras",
)
class Interoperability(unittest.TestCase):
    def test_published_shotweave(self):
        import shotweave as sw
        from qiskit import QuantumCircuit

        from bornlens.qiskit import sample_counts
        from bornlens.shotweave import compare_final

        self.assertEqual(version("shotweave"), "0.2.0")
        circuits = [QuantumCircuit(1), QuantumCircuit(1)]
        circuits[1].ry(0.8, 0)
        for strategy in ("uniform", "adaptive", "bayesian", "robust", "noise-aware"):
            batches = []

            def submit(shots):
                counts = [
                    int(
                        sample_counts(c, "Z", n, seed=47 + 10 * len(batches) + i).get(
                            "1", 0
                        )
                    )
                    for i, (n, c) in enumerate(zip(shots, circuits))
                ]
                batches.append(counts)
                return counts

            opts = (
                {"readout": [sw.Readout(0.02, 0.03)] * 2}
                if strategy == "noise-aware"
                else {}
            )
            exp = sw.run_experiment(
                512, [1, 1], submit, strategy=strategy, pilot_size=8, **opts
            )
            self.assertEqual(exp.total_shots, 512)
            samples = [
                Sample("Z", n - k, k) for n, k in zip(exp.final_shots, exp.successes)
            ]
            ref = compare(*samples, tolerance=0.2)
            if exp.pilot_shots:
                rows = sw.bornlens_rows(
                    "sim-pilot",
                    "sim-final",
                    exp.pilot_shots,
                    exp.final_shots,
                    exp.successes,
                )
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "rows"
                    path.write_text("\n".join(map(json.dumps, rows)))
                    self.assertEqual(compare_final(path, 0, 1, tolerance=0.2), ref)
            self.assertEqual(ref.shots_a + ref.shots_b, sum(exp.final_shots))


@unittest.skipUnless(STATS, "stats extra")
class AcceptanceEdges(unittest.TestCase):
    def test_adapter_and_bounds(self):
        from bornlens import advanced as a

        s = Sample("Z", 2, 3)
        for family in [[], {}, {"X": (s, s)}, {"Z": (s, Sample("X", 1, 2))}]:
            with self.assertRaises(ValueError):
                a.compare_many(family, tolerance=0.1)
        with self.assertRaises(ValueError):
            a.distribution_compare({"A": 10**30}, {"B": 1})
        self.assertIsNone(
            a.optimize_design(0.2, 0.3, [(5, 5)], budget=4, tolerance=0.1)
        )
        with self.assertRaises(ValueError):
            a.bayesian_compare(s, s, tolerance=float("nan"))
        self.assertTrue(
            a.hierarchical_effect([0.1, 0.2], [1e-150, 0.01])["between_variance"] >= 0
        )

    def test_qiskit_regression(self):
        if not importlib.util.find_spec("qiskit"):
            self.skipTest("Qiskit extra")
        from qiskit import QuantumCircuit

        from bornlens.advanced import regression_check
        from bornlens.qiskit import from_counts, sample_counts

        a = QuantumCircuit(1)
        for n, angle, label in [
            (10000, 0, "pass"),
            (10000, 0.01, "pass"),
            (10000, 2.8, "fail"),
            (8, 0, "inconclusive"),
        ]:
            b = QuantumCircuit(1)
            b.x(0)
            b.x(0)
            b.ry(angle, 0)
            family = {
                p: tuple(
                    from_counts(sample_counts(c, p, n, seed=i + 42), p)
                    for i, c in enumerate([a, b])
                )
                for p in ("X", "Z")
            }
            self.assertEqual(regression_check(family, tolerance=0.1)["status"], label)

    def test_design_oracle(self):
        from scipy.stats import binom

        from bornlens.advanced import optimize_design

        designs = [(6, 8), (7, 7), (8, 6)]

        def brute(na, nb):
            return sum(
                binom.pmf(i, na, 0.02) * binom.pmf(j, nb, 0.97)
                for i in range(na + 1)
                for j in range(nb + 1)
                if compare(
                    Sample("Z", i, na - i), Sample("Z", j, nb - j), tolerance=0.2
                ).verdict
                == "beyond_tolerance"
            )

        best, result = optimize_design(0.02, 0.97, designs, budget=14, tolerance=0.2)
        self.assertAlmostEqual(result.correct, max(brute(*d) for d in designs))
        self.assertAlmostEqual(result.correct, brute(*best))


if importlib.util.find_spec("hypothesis") and STATS:
    from hypothesis import given
    from hypothesis import strategies as st

    class Properties(unittest.TestCase):
        @given(
            st.lists(
                st.floats(min_value=0, max_value=1, allow_nan=False),
                min_size=1,
                max_size=40,
            )
        )
        def test_holm_monotone(self, p):
            from bornlens.advanced import adjust_pvalues

            adjusted = adjust_pvalues(sorted(p))
            self.assertEqual(adjusted, sorted(adjusted))
            self.assertTrue(
                all(raw <= adj <= 1 for raw, adj in zip(sorted(p), adjusted))
            )
