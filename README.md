# BornLens

BornLens compares a Pauli expectation before and after a quantum circuit change.
It reports the measured change, a finite-sample confidence interval and a tolerance verdict.
Part of the PhaseTensor Quantum Computing Suite.

Install from a checkout with `python -m pip install .`. For ideal Qiskit simulation use
`python -m pip install '.[qiskit]'`.

```python
from bornlens import Sample, compare, shots_per_circuit

before = Sample("Z", plus=7500, minus=2500)
after = Sample("Z", plus=7550, minus=2450)
result = compare(before, after, tolerance=0.05)
print(result.verdict, result.delta, result.interval)
# within_tolerance, approximately 0.01 and (-0.0284, 0.0484)
print(shots_per_circuit(0.05))  # 5903 shots per circuit at 95% confidence
```

## Statistical protocol

Fix the observable, tolerance, confidence and sample sizes before collecting data.
Assume independent shots within and between circuits, with a stable outcome distribution
for each circuit. Each measured eigenvalue is +1 or -1. The sample mean is
(plus - minus) / shots. The estimated change is mean B minus mean A.

For confidence 1 - alpha, apply the two-sided Hoeffding bound to the weighted sum
of the independent signed outcomes. Each A contribution has range width 2/nA and
each B contribution has width 2/nB. The resulting half-width is
sqrt(2 log(2/alpha) (1/nA + 1/nB)). Intersect the interval with [-2, 2].
For the identity observable the change is exactly zero. Hoeffding gives a
conservative finite-sample bound even for unanimous outcomes. A normal approximation
using empirical variance can give zero width for those samples.

An interval contained in the closed tolerance band gives within_tolerance.
An interval strictly above or below the band gives beyond_tolerance. All other
intervals give inconclusive. Touching a boundary from outside is inconclusive.
Coverage concerns one prespecified comparison. Drift, correlated shots, repeated
looks and multiple comparisons require separate treatment.

For equal sample sizes the sufficient shots per circuit for half-width h are
ceil(4 log(2/alpha) / h²). This controls precision. The verdict also depends
on the distance of the true change from the tolerance boundary.

Reference: [Hoeffding (1963), Probability Inequalities for Sums of Bounded Random Variables](https://doi.org/10.1080/01621459.1963.10500830).

## Qiskit

```python
from qiskit import QuantumCircuit
from bornlens.qiskit import compare_circuits

a = QuantumCircuit(1)
b = a.copy()
b.x(0)
result = compare_circuits(a, b, "Z", shots_a=1000, shots_b=1500,
                          tolerance=0.1, seed=17)
print(result.verdict)  # beyond_tolerance
```

Pass bound gate circuits before measurements. Rightmost Pauli factor acts on qubit 0.
The adapter rotates X with H and Y with S-dagger followed by H, then computes parity
over nonidentity factors. For external integer counts use from_counts(counts, label)
after applying these basis rotations and measuring qubit i into classical bit i
of one full-width register. The caller supplies the measurement basis and mapping.
The simulator uses separate random streams for A and B. Run the checks with
`python -m unittest discover -s tests -v`.

## Validation and release 0.1.0

Install the release with `python -m pip install bornlens==0.2.0`.
For development install the checkout with the Qiskit extra and run
`python benchmarks/coverage.py` to reproduce [the benchmark counts](benchmarks/results.csv).
Each row contains 5,000 independent trials with a fixed seed, tolerance 0.1 and
nominal coverage 95%. The samples follow the exact Z measurement law of ideal
single-qubit Ry circuits. Incorrect counts exclude inconclusive decisions.
The benchmark covers five true changes and three shot budgets. It measures
ideal fixed-shot behavior; hardware drift and correlated noise require separate experiments.

## Statistical extensions

Install the stats extra with `python -m pip install 'bornlens[stats]'`. See [methods, power and hardware evidence](docs/methods.md) and the [changelog](CHANGELOG.md).
