"""Run after installing bornlens[stats]. CSV counts and marginal Wilson MC intervals."""
import csv
import sys
import numpy as np
from bornlens.statistics import interval, _bounds

rng = np.random.default_rng(47)
writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow("pa pb na nb method metric count trials rate mc95_lo mc95_hi".split())
for pa,pb in [(0,0),(0,1),(.005,.015),(.5,.5),(.5,.55),(.5,.58),(.9,.3)]:
    for na,nb in [(250,250),(100,2000),(2000,100),(5000,5000)]:
        trials = 10000
        a,b = rng.binomial(na,pa,trials),rng.binomial(nb,pb,trials)
        delta = round(2*(pb-pa),12)
        for method in ("hoeffding","newcombe","exact"):
            lo,hi = interval(a,na,b,nb,.95,method)
            within = (lo >= -.1) & (hi <= .1)
            beyond = (lo > .1) | (hi < -.1)
            metrics = {"coverage": (lo <= delta) & (hi >= delta),
                       "inconclusive": ~(within | beyond)}
            if abs(delta) <= .1:
                metrics["false_beyond"] = beyond
            else:
                metrics["false_equivalence"] = within
            for name, mask in metrics.items():
                k = int(mask.sum())
                low,high = _bounds(k,trials,.05,"newcombe")
                writer.writerow((pa,pb,na,nb,method,name,k,trials,k/trials,float(low),float(high)))
