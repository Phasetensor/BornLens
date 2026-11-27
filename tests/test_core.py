import math
import unittest

from bornlens import Sample, _verdict, compare, shots_per_circuit


class CoreTests(unittest.TestCase):
    def test_three_decisions(self):
        a = Sample("Z", 7500, 2500)
        for b, verdict in [(Sample("Z", 7550, 2450), "within_tolerance"),
                           (Sample("Z", 2500, 7500), "beyond_tolerance"),
                           (Sample("Z", 7750, 2250), "inconclusive")]:
            self.assertEqual(compare(a, b, tolerance=.05).verdict, verdict)

    def test_boundaries(self):
        for sign in (-1, 1):
            edge = sign * .25
            self.assertEqual(_verdict(edge, edge, .25), "within_tolerance")
            lo, hi = sorted((edge, sign * .5))
            self.assertEqual(_verdict(lo, hi, .25), "inconclusive")

    def test_unequal_sizes(self):
        a, b = Sample("X", 3, 1), Sample("X", 2, 6)
        r = compare(a, b, tolerance=.2)
        self.assertEqual((r.delta, r.shots_a, r.shots_b), (-1, 4, 8))
        self.assertEqual(r.interval[0], -2)
        self.assertAlmostEqual(r.interval[1], -1 + math.sqrt(.75 * math.log(40)))

    def test_validation(self):
        for args in [("z", 1, 1), ("Z", 0, 0), ("Z", True, 0),
                     ("I", 0, 1), ("X", -1, 3), ("Z", 1.2, 1)]:
            with self.assertRaises(ValueError):
                Sample(*args)
        a = Sample("I", 10, 0)
        self.assertEqual(compare(a, a, tolerance=0).interval, (0, 0))
        for b, tol, conf in [(Sample("X", 1, 0), 1, .95),
                             (a, float("nan"), .95), (a, 1, 1)]:
            with self.assertRaises(ValueError):
                compare(a, b, tolerance=tol, confidence=conf)

    def test_planner(self):
        n = shots_per_circuit(.05)
        self.assertEqual(n, 5903)
        for shots in (n, n - 1):
            a = Sample("Y", shots, 0)
            self.assertEqual(compare(a, a, tolerance=.05).interval[1] <= .05, shots == n)
