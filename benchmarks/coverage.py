"""Reproduce CSV with python benchmarks/coverage.py from an installed checkout."""
import csv
import sys

import numpy as np

from bornlens import Sample, compare

rng = np.random.default_rng(20261127)
writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow(("delta", "shots_each", "trials", "covered", "incorrect", "inconclusive"))
# For Ry(arccos(mu)) applied to |0>, Z has mean mu. Sample this circuit law exactly.
for delta in (0, 0.05, 0.1, 0.15, -0.3):
    for n in (250, 2000, 10000):
        trials = 5000
        before = rng.binomial(n, .5, trials)
        after = rng.binomial(n, (1 + delta) / 2, trials)
        covered = incorrect = undecided = 0
        truth = "within_tolerance" if abs(delta) <= .1 else "beyond_tolerance"
        for a, b in zip(before, after):
            r = compare(Sample("Z", a, n-a), Sample("Z", b, n-b), tolerance=.1)
            covered += r.interval[0] <= delta <= r.interval[1]
            undecided += r.verdict == "inconclusive"
            incorrect += r.verdict not in (truth, "inconclusive")
        writer.writerow((delta, n, trials, covered, incorrect, undecided))
