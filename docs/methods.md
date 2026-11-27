# Methods and validation

The estimand is twice the difference in plus-outcome probabilities (B minus A).
All procedures assume independent Bernoulli shots, stable per-arm probabilities,
prespecified arms, a fixed measurement basis and a fixed analysis time.

| compare(..., method=...) | Construction | Interpretation |
|---|---|---|
| hoeffding (default) | Bounded independent sum | Finite-sample coverage at least the nominal level |
| newcombe | Newcombe method 10, combining asymmetric Wilson score distances by root sum of squares | Approximate coverage, including possible undercoverage |
| exact | Clopper–Pearson marginal intervals at confidence 1-alpha/2; subtract opposite limits and multiply by two | Finite-sample conservative joint bound by Bonferroni; no inversion of an unconditional risk-difference test |

[Newcombe (1998)](https://pubmed.ncbi.nlm.nih.gov/9595617/).
[Clopper and Pearson (1934)](https://doi.org/10.1093/biomet/26.4.404).

## Precision, classification and power

The precision planner targets a Hoeffding half-width regardless of the true effect.
Classification power depends on both probabilities, the method and the tolerance.
The power engine integrates the independent binomial laws using monotone interval
endpoints to locate decision regions. It reports all three verdict probabilities.
Correct classification at either band endpoint means a within verdict. An
inconclusive result does not count as correct classification.

```python
from bornlens.statistics import classification_power, plan_classification

p = classification_power(0.5, 0.65, 2000, 2000, tolerance=0.1)
design, power = plan_classification(
    0.5, 0.65, [(250, 250), (2000, 2000), (5000, 5000)], target=0.9, tolerance=0.1
)
```

The planner searches the supplied designs in increasing total shots. It returns
None if none qualifies. Results are conditional on the input probabilities;
repeat across plausible baselines and effects to assess sensitivity.

## Benchmark

Run `python benchmarks/methods.py` with the stats extra installed. The CSV
contains 28 scenarios (7 probability pairs and 4 allocations), 10,000 trials
each. All three methods share the same random counts within a scenario.
Metrics use all trials within the applicable truth class as denominator.
False equivalence is a within verdict outside the true band. False beyond is a
beyond verdict inside the true band. The third outcome is reported separately.
Wilson 95% intervals quantify Monte Carlo uncertainty for each rate. They are
marginal intervals; shared samples induce dependence across methods.

Minimum observed coverage across the grid: Hoeffding 99.10%, exact 99.25%,
Newcombe 94.36%. These minima describe the simulated grid. Degenerate samples,
near-boundary probabilities and unequal allocations are included in the CSV.

## ShotWeave and hardware

For a fixed pair of arms and fixed tolerance, call
`bornlens.shotweave.compare_final(path, 0, 1, tolerance=.1)` on the single-qubit
Z JSONL format emitted by ShotWeave's hardware example. The pilot may determine
the final allocations. Conditional on the pilot, the final sizes are fixed and
fresh outcomes must follow the assumed independent laws. Do not select the pair
after inspecting the final outcomes, pool the pilot counts, use the adaptive
total as a fixed sample, or stop early on the verdict. Correct for multiplicity
if several comparisons inform a joint claim.

Retrospective application to the existing IBM Marrakesh ShotWeave experiment,
final job dc4llkeq4uuc73a92eeg, angles 0 and 0.7:
359 and 2604 final shots, 2 and 312 minus outcomes. With tolerance 0.1,
the default estimate is -0.22849, 95% interval [-0.38141, -0.07557],
inconclusive. This illustrates real-data ingestion. Hardware drift and readout
bias can shift expectations relative to ideal circuits; a single execution
does not establish frequentist hardware coverage.

For a prospective run, install qiskit-ibm-runtime and the qiskit extra,
authenticate an IBM Quantum account, then run
`python examples/ibm_fixed_design.py --backend BACKEND`. One job measures Z
for two Ry circuits on physical qubit 0 with 4096 shots each. The ideal change
is -0.5, tolerance is 0.1 and confidence is 95%. Save the printed job ID;
use --job-id to resume collection. This protocol awaits fresh hardware data.

## Software checks

The 12-test suite includes brute-force joint enumeration as an independent
check of the power algorithm and fixtures for disjoint pilot/final stages.
Python trace with missing-line counting recorded 219/228 executable package
lines (96.1%) on Python 3.12. CI also measures branches using coverage.py;
these metrics use different denominators.

## Advanced validation

The acceptance study, independent reference tests, Monte Carlo errors, runtime
and memory are indexed in [the traceability matrix](traceability.md).
The sequential implementation uses a summable union bound; see
[Howard et al. on confidence sequences](https://par.nsf.gov/servlets/purl/10251927)
for broader constructions. The permutation Monte Carlo test counts the
observed assignment in the reference set; see
[Phipson and Smyth](https://pubmed.ncbi.nlm.nih.gov/21044043/).
