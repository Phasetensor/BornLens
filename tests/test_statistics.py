import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from bornlens import Sample, compare
from bornlens.shotweave import compare_final


class InteropTests(unittest.TestCase):
    def test_stage_separation(self):
        rows = [
            dict(job_id="a", shots=[10, 10]),
            dict(stage="final", job_id="b", shots=[11, 20]),
            dict(
                total_shots=51,
                final_shots=[11, 20],
                successes=[2, 5],
                pilot_successes=[10, 0],
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            path.write_text("\n".join(map(json.dumps, rows)))
            self.assertEqual(
                compare_final(path, 0, 1, tolerance=0.2),
                compare(Sample("Z", 9, 2), Sample("Z", 15, 5), tolerance=0.2),
            )
            with self.assertRaises(ValueError):
                compare_final(path, 1, 1, tolerance=0.2)
            rows[1]["job_id"] = "a"
            path.write_text("\n".join(map(json.dumps, rows)))
            with self.assertRaises(ValueError):
                compare_final(path, 0, 1, tolerance=0.2)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "stats extra")
class InferenceTests(unittest.TestCase):
    def test_power_matches_joint_enumeration(self):
        from scipy.stats import binom

        from bornlens.statistics import classification_power

        for method in ("hoeffding", "newcombe", "exact"):
            for pa, pb in ((0, 1), (0.2, 0.75), (0.5, 0.5)):
                expected = dict(
                    within_tolerance=0.0, beyond_tolerance=0.0, inconclusive=0.0
                )
                for i in range(8):
                    for j in range(12):
                        r = compare(
                            Sample("Z", i, 7 - i),
                            Sample("Z", j, 11 - j),
                            tolerance=0.8,
                            method=method,
                        )
                        expected[r.verdict] += binom.pmf(i, 7, pa) * binom.pmf(
                            j, 11, pb
                        )
                p = classification_power(pa, pb, 7, 11, tolerance=0.8, method=method)
                for value, reference in zip(
                    (p.within, p.beyond, p.inconclusive), expected.values()
                ):
                    self.assertAlmostEqual(value, reference)

    def test_endpoint_and_swap(self):
        for method in ("exact", "newcombe"):
            a, b = Sample("Z", 0, 15), Sample("Z", 20, 0)
            r = compare(a, b, tolerance=0.1, method=method)
            s = compare(b, a, tolerance=0.1, method=method)
            self.assertEqual(r.interval[1], 2)
            self.assertAlmostEqual(r.interval[0], -s.interval[1])
            self.assertEqual(r.verdict, "beyond_tolerance")

    def test_design_search(self):
        from bornlens.statistics import plan_classification

        settings = dict(target=0.9, tolerance=0.2)
        self.assertEqual(
            plan_classification(0, 1, [(100, 100), (10, 10)], **settings)[0], (10, 10)
        )
        self.assertIsNone(plan_classification(0.5, 0.5, [(3, 3)], **settings))
