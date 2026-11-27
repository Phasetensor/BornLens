# BornLens Advanced acceptance matrix

All functions below were released in 0.3.0. This matrix describes the
consolidation candidate. Evidence is method-specific: exact enumeration,
analytic identities, independent-library checks and simulation answer different
questions. The benchmark is an ideal sampling experiment. There is no claim of
a new device execution. See [API and assumptions](advanced.md),
[metrics](../benchmarks/acceptance/metrics.csv) and [test implementation](../tests/test_acceptance.py).

| Public API (bornlens.advanced) | Formula or estimand | Verification and benchmark series | Scope and limitation |
|---|---|---|---|
| compare_many | For m labels use confidence 1-alpha/m for each component | family coverage and both directional errors; test_family_and_regression | Prespecified family; marginal labels may be dependent. |
| adjust_pvalues | Bonferroni mp; Holm cumulative maxima; BH reverse minima mp/rank; BY adds harmonic factor | statsmodels multipletests; Hypothesis monotonicity; multiplicity FWER/FDR/power | BH under independence or PRDS. Negative correlation is a stress scenario. FDR and FWER differ under mixed truth. |
| sequential_compare | Hoeffding radius sqrt(4 log(2n(n+1)/alpha)/n) | all 256 looks on each trajectory; terminal, repeated and first-decision policies | Independent streams, equal cumulative n, fixed pair. Conservative near the boundary. |
| bayesian_compare | Draw probabilities from independent updated Beta laws and transform 2(pB-pA) | numerical quadrature for band mass and analytic posterior mean; bayes series | Credible bounds are posterior quantiles. MC standard errors describe integration noise. Prior-sensitive for small samples. |
| paired_compare | Mean of B-A in [-2,2], radius sqrt(8 log(2/alpha)/n) | paired series: matched, rare and anticorrelated pairs; test_paired_and_randomization | Independence across pairs; pairing information is required. |
| bootstrap_interval | Percentile quantiles of empirical resamples of pairs or independent outcomes | paired/unpaired coverage, bias, RMSE and width | A degenerate empirical law gives zero width. Coverage may fall below target. |
| distribution_compare | Total variation = half the L1 distance; conditional pooled-category permutations | exhaustive 126 assignments; categorical null, sparse and separated alternatives | Fixed category encoding. Finite draws limit p-value resolution. |
| regression_check | Fail on any outside interval, pass on all inside, otherwise inconclusive | Qiskit identity, cancelling gates, perturbations, multiple labels and shot budgets; regression series | Validates only measured observables. |
| readout_diagnostics | Conditional flip proportions, contrast 1-p01-p10 | exact binomial marginal bounds; readout series, zero and asymmetric errors | Preparation errors can contaminate readout estimates. |
| drift_monitor | Fixed-reference Fisher tests with alpha/[t(t+1)] | stationary, step and ramp; two reference/batch sizes, conditional delay | Disjoint windows. Conditional delay excludes undetected paths and requires detection probability alongside it. |
| effect_size | Delta=2(pB-pA); Cohen h=2(asin sqrt(pB)-asin sqrt(pA)); excess=abs(delta)-epsilon | endpoint analytic identities in test_design_and_effect | Point estimates; magnitude alone does not establish statistical significance. |
| optimize_design | Maximize correct-classification mass over supplied feasible designs | independent joint binomial enumeration in test_design_oracle; design curves | Conditional on assumed probabilities and candidate set; tie-break on total shots. |
| randomization_test | Fisher conditional test, or sign-flips among discordant pairs | exhaustive sign assignments, known Fisher probability and categorical permutation oracle | Exchangeability under the null; failure to reject cannot establish equivalence. |
| hierarchical_effect | DL moment tau², inverse-total-variance pooling, normal plug-in CI | statsmodels independent reference; hierarchy bias, coverage, raw and shrunk MSE | Independent batches, supplied positive within-batch SE. Few-group plug-in intervals can undercover. |
| write_report | Serialize supplied input, settings, results and dependency versions | strict JSON roundtrip and unsupported-input rejection | Caller must include acquisition identifiers and seeds; serialization cannot reconstruct omitted information. |

## Evidence classification

All 15 APIs have implementation and test evidence. Sampling guarantees are
conditional on the assumptions above. Monte Carlo validates only the reported
scenarios. Reporting is checked by serialization, and effect sizes by analytic
identities; a coverage rate for either would be meaningless. Hardware
generalization is outside this acceptance experiment.
