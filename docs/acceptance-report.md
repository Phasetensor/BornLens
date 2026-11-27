# Scientific acceptance report for BornLens 0.3.1

Candidate branch: advanced-final-validation. Baseline tag: v0.3.0 at
7e95beb874de855cadd3ac041d9f72df2ffeb1a2. The approved commit is `4db29f5c26deb3206a46fb36e21f747fdac54a27`. The [acceptance run](https://github.com/Phasetensor/BornLens/actions/runs/46012347525) records the test log and coverage counts.

## Scientific evidence

The [matrix](traceability.md) covers all fifteen previously released APIs.
[Recorded metrics](../benchmarks/acceptance/metrics.csv) and
[figures](../benchmarks/acceptance/validation.svg) accompany the protocol.
Tests compare against statsmodels multiplicity and meta-analysis routines,
adaptive quadrature for posterior probabilities and credible endpoints,
exhaustive sign assignments and label permutations, and joint-binomial
enumeration for design choices. The small categorical permutation estimator
is compared to exact enumeration within a prespecified Monte Carlo tolerance.

For mixed null/alternative tests in the independent and positive-dependence
scenarios, observed Holm FWER ranged from 3.07% to 4.43%. BH FDR ranged from
2.49% to 3.28%. The two metrics have different denominators. Under the
0.5/0.5 sequential null, terminal fixed-sample coverage was 99.17% and
whole-path repeated fixed-sample coverage was 91.03% (MC SE 0.29 percentage
points). All 10,000 paths were covered by the confidence sequence in this
scenario. Across the six scenarios the minimum observed sequence coverage
was 99.99%. Finite simulation does not prove an error-control theorem.

The independent-pair bootstrap is inappropriate when measurements share
within-pair dependence. Even with correct pair resampling, coverage was
83.4% in one 12-pair scenario. With three heterogeneous groups the normal
plug-in hierarchical interval reached only 79.22% in one scenario. Both
results limit nominal-coverage interpretation. Single-example regression
decisions, posterior examples and analytic identities are labelled separately
from repeated-experiment operating characteristics.

## Engineering and publication

CI exercises the dependency-free core and optional statistics/Qiskit stacks.
The reference job installs ShotWeave 0.2.0 from PyPI and uses actual Qiskit
measurements in fixed, adaptive, Bayesian, robust and noise-aware experiments.
It validates raw observed counts, stage separation and budget accounting.
The workflow builds wheel and sdist, checks metadata, installs both outside
the checkout and runs the advanced guide against both installed packages.
Statement and branch coverage are gated separately; exact counts are in the
workflow log. Release promotion requires explicit approval and the reusable
acceptance job. The release workflow checks downloaded GitHub and PyPI bytes
against its local validated build. The [release run](https://github.com/Phasetensor/BornLens/actions/runs/46013266006) verifies both public distributions against the [approved hashes](../releases/0.3.1.sha256).

No new hardware acquisitions were made. Readout diagnostics do not identify
the physical cause of observed discrepancies. Drift delay is conditional on
an alarm and the model assumes fixed disjoint batches. Numeric reference and
simulation validation does not imply certification across hardware backends.
