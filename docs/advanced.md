# Advanced analysis (0.3.1)

Install `python -m pip install 'bornlens[stats]==0.3.1'`. All functions below are
importable from `bornlens.advanced`. Tolerances and signed changes use Pauli
expectation units, B minus A. Declare the family, pairing, decision margins,
look schedule and model before observing outcomes.

| Function | Inputs and interpretation |
|---|---|
| compare_many | Map Pauli labels to pairs of Sample objects. Simultaneous Bonferroni intervals with Hoeffding or exact marginal intervals. Across-label dependence is allowed. |
| adjust_pvalues | One prespecified family of valid raw p-values. Bonferroni and Holm control familywise error. Benjamini–Hochberg (fdr_bh) assumes independence or positive regression dependence. Benjamini–Yekutieli (fdr_by) allows arbitrary dependence. A large adjusted difference-test p-value does not establish equivalence. |
| sequential_compare | Supply cumulative counts for a single fixed pair with equal sample sizes n. Hoeffding error spending alpha/[n(n+1)] sums to alpha, allowing a decision after any matched look. Assumes independent stationary shots within and across circuits. Do not reset counts between looks. |
| bayesian_compare | Independent Beta priors on the plus probabilities, default Jeffreys. Reports posterior mean, equal-tail credible interval and posterior mass of the tolerance band. Monte Carlo standard error measures numerical sampling uncertainty. |
| paired_compare | Two aligned +/-1 sequences. Dependence within a pair is allowed. Pairs must be independent. Hoeffding on differences in [-2,2]. Aggregated marginal counts lose the pairing. |
| bootstrap_interval | Percentile interval, independent count resampling or whole-pair resampling. Use a fixed seed. Coverage is approximate; unanimous samples can yield a zero-width interval. |
| distribution_compare | Two categorical count mappings, common encoding, independent samples. Total variation in [0,1] and Monte Carlo permutation p-value conditional on pooled counts, with plus-one correction. |
| regression_check | One fixed reference/candidate observable family. Fail on any beyond verdict, pass only if every simultaneous interval is within its band. Remaining cases are inconclusive. |
| readout_diagnostics | Known prepared zero and one measured in Z. Directional flip rates, simultaneous exact marginal bounds and contrast. Preparation and measurement errors may both contribute. |
| drift_monitor | Prespecified disjoint batches against a fixed independent reference. Two-sided exact tests use alpha/[t(t+1)] at window t. Supply the complete history on update. Detects distribution changes without attributing their physical cause. |
| effect_size | Signed Pauli change, plus-probability difference, Cohen's h and absolute change minus tolerance. These are point estimates. |
| optimize_design | Supply assumed plus probabilities, candidate shot pairs, total budget and comparison settings. Maximizes computed probability of correct classification over feasible candidates. Ties prefer fewer shots. |
| randomization_test | Independent binary samples use the exact conditional Fisher test. Matched exchangeable labels use a two-sided sign-flip test conditional on discordant pairs. A difference-test p-value is not an equivalence verdict. |
| hierarchical_effect | Independent study/batch estimates and positive within-study standard errors. Normal random effects, DerSimonian–Laird heterogeneity, plug-in normal interval and partially pooled estimates. Approximation can be poor with few batches or uncertain standard errors. |
| write_report | Strict JSON containing the supplied inputs, analysis settings, results and installed package versions. Include the seed, shot order or pairing and acquisition identifiers needed to rerun the analysis. |

## Reproduce a regression check

```python
from bornlens import Sample
from bornlens.advanced import regression_check, write_report

family = {
    "Z": (Sample("Z", 8000, 2000), Sample("Z", 8050, 1950)),
    "X": (Sample("X", 5000, 5000), Sample("X", 4990, 5010)),
}
settings = {"tolerance": 0.1, "confidence": 0.95}
result = regression_check(family, **settings)
assert result["status"] == "pass"
write_report("regression.json", inputs=family, settings=settings, results=result)
```

## Method references

- [Multiplicity procedures and dependence assumptions](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html).
- [Random-effects analysis and uncertainty in heterogeneity](https://training.cochrane.org/handbook/current/chapter-10).

See the [per-method validation and limitations](traceability.md) and
[operating characteristics](../benchmarks/acceptance/README.md).
Time complexity is linear in shots for paired observations, O(m log m)
for sorting m p-values, and O(Rk) for R conditional draws over k outcome
categories. Posterior sampling costs O(R). The power planner performs
O(nA log(nB+1)) vector work per candidate; large asymmetric designs can
be expensive. Fixed-reference monitoring reuses the reference and controls
the family via spending; it is not an estimator of change time or physical cause.

## Minimal calls for all methods

With the stats extra installed, these calls use counts or signed eigenvalues as
indicated. User-supplied experiment identifiers should accompany actual data.

```python
from bornlens import Sample
from bornlens import advanced as bl

a, b = Sample("Z", 81, 19), Sample("Z", 70, 30)
family = {"Z": (a, b)}
pairs_a, pairs_b = [1, -1, 1, 1], [1, -1, -1, 1]
settings = {"tolerance": 0.15}
bl.compare_many(family, **settings)
bl.adjust_pvalues([0.008, 0.03, 0.4], "holm")
bl.sequential_compare(a, b, **settings)
bl.bayesian_compare(a, b, seed=47, **settings)
bl.paired_compare(pairs_a, pairs_b, **settings)
bl.bootstrap_interval(pairs_a, pairs_b, paired=True, seed=47)
bl.distribution_compare({"00": 81, "11": 19}, {"00": 70, "11": 30}, seed=47)
result = bl.regression_check(family, **settings)
bl.readout_diagnostics(Sample("Z", 99, 1), Sample("Z", 3, 97))
bl.drift_monitor(a, [b])
bl.effect_size(a, b, **settings)
bl.optimize_design(0.81, 0.7, [(100, 100), (150, 50)], budget=200, **settings)
bl.randomization_test(pairs_a, pairs_b, paired=True)
bl.hierarchical_effect([0.1, 0.2, -0.03], [0.02, 0.04, 0.03])
bl.write_report("family.json", inputs=family, settings=settings, results=result)
```
