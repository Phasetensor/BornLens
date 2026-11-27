"""Prespecified validation at seed 4703. Run from an installed checkout."""

import argparse
import csv
import json
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import beta, norm

from bornlens import Sample
from bornlens import advanced as api
from bornlens.statistics import classification_power

ROOT = Path(__file__).parent / "acceptance"


def validate(reps, out):
    rng = np.random.default_rng(4703)
    rows = []
    timings = {}

    def record(area, scenario, metric, samples):
        x = np.asarray(samples, dtype=float).ravel()
        rows.append(
            [
                area,
                scenario,
                metric,
                len(x),
                x.mean(),
                x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else "",
            ]
        )

    def timed(name, fn):
        tracemalloc.start()
        start = time.perf_counter()
        fn()
        timings[name] = [
            time.perf_counter() - start,
            tracemalloc.get_traced_memory()[1],
        ]
        tracemalloc.stop()

    def sequential():
        for pa, pb in [
            (0, 0),
            (0, 1),
            (0.001, 0.01),
            (0.5, 0.5),
            (0.5, 0.55),
            (0.5, 0.7),
        ]:
            d = round(2 * (pb - pa), 12)
            scenario = f"{pa}/{pb}"
            ca = np.zeros(reps, dtype=int)
            cb = ca.copy()
            holds = {k: np.ones(reps, bool) for k in ("sequence", "naive")}
            stopped = np.zeros(reps, bool)
            stop_n = np.full(reps, 256)
            at_stop = np.ones(reps, bool)
            for n in range(1, 257):
                ca += rng.binomial(1, pa, reps)
                cb += rng.binomial(1, pb, reps)
                estimate = 2 * (cb - ca) / n
                radius = np.sqrt(4 * np.log(40 * n * (n + 1)) / n)
                fixed = np.sqrt(4 * np.log(40) / n)
                covered = abs(estimate - d) <= radius
                holds["sequence"] &= covered
                holds["naive"] &= abs(estimate - d) <= fixed
                first = (~stopped) & (
                    (abs(estimate) + radius <= 0.1) | (abs(estimate) - radius > 0.1)
                )
                stop_n[first] = n
                at_stop[first] = covered[first]
                stopped |= first
                if n in (1, 16, 64, 256):
                    r = api.sequential_compare(
                        Sample("Z", ca[0], n - ca[0]),
                        Sample("Z", cb[0], n - cb[0]),
                        tolerance=0.1,
                    )
                    np.testing.assert_allclose(
                        r.interval,
                        [max(-2, estimate[0] - radius), min(2, estimate[0] + radius)],
                    )
                    record("sequence-path", scenario, f"halfwidth-{n}", [radius])
            at_stop[~stopped] = covered[~stopped]
            for key, values in {
                **holds,
                "fixed_at_horizon": abs(estimate - d) <= fixed,
                "stopping_coverage": at_stop,
                "early_decision": stopped,
                "sample_size": stop_n,
            }.items():
                record("sequential", scenario, key, values)

    def multiple():
        for m in (4, 12):
            for rho in (0, 0.65, -0.06):
                corr = np.full((m, m), float(rho))
                np.fill_diagonal(corr, 1)
                z = rng.multivariate_normal(np.zeros(m), corr, reps)
                alt = np.arange(m) % 3 == 0
                for mix in (False, True):
                    truth = alt & mix
                    p = norm.sf(z + truth * 2.8)
                    for method in ("bonferroni", "holm", "fdr_bh", "fdr_by"):
                        order = p.argsort(axis=1)
                        ordered = np.take_along_axis(p, order, axis=1)
                        ranks = np.arange(1, m + 1)
                        if method == "bonferroni":
                            adjusted = ordered * m
                        elif method == "holm":
                            adjusted = np.maximum.accumulate(
                                ordered * (m + 1 - ranks), axis=1
                            )
                        else:
                            factor = np.sum(1 / ranks) if method == "fdr_by" else 1
                            adjusted = np.minimum.accumulate(
                                (ordered * m * factor / ranks)[:, ::-1], axis=1
                            )[:, ::-1]
                        adj = np.take_along_axis(
                            np.minimum(1, adjusted), order.argsort(axis=1), axis=1
                        )
                        np.testing.assert_allclose(
                            adj[0], api.adjust_pvalues(p[0], method)
                        )
                        reject = adj <= 0.05
                        false = reject[:, ~truth].sum(axis=1)
                        scenario = f"{method}/{m}/{rho}/{mix}"
                        record("multiplicity", scenario, "FWER", false > 0)
                        record(
                            "multiplicity",
                            scenario,
                            "FDR",
                            false / np.maximum(1, reject.sum(axis=1)),
                        )
                        if mix:
                            record(
                                "multiplicity",
                                scenario,
                                "power",
                                reject[:, truth].mean(axis=1),
                            )
                # Correlated binomial counts via Gaussian copula with exact marginal laws.
                from scipy.stats import binom

                pa = np.resize([0, 0.5, 0.02, 0.95], m)
                pb = np.resize([0, 0.5, 0.15, 0.65], m)
                na = np.resize([60, 400], m)
                nb = np.resize([250, 100], m)
                a = binom.ppf(np.clip(norm.cdf(z), 1e-15, 1 - 1e-15), na, pa)
                b = binom.ppf(
                    np.clip(
                        norm.cdf(rng.multivariate_normal(np.zeros(m), corr, reps)),
                        1e-15,
                        1 - 1e-15,
                    ),
                    nb,
                    pb,
                )
                est = 2 * (b / nb - a / na)
                radius = np.sqrt(2 * np.log(40 * m) * (1 / na + 1 / nb))
                delta = 2 * (pb - pa)
                record(
                    "family",
                    f"{m}/{rho}",
                    "simultaneous_coverage",
                    (abs(est - delta) <= radius).all(axis=1),
                )
                record(
                    "family",
                    f"{m}/{rho}",
                    "false_equivalence",
                    ((abs(est) + radius <= 0.1) & (abs(delta) > 0.1)).any(axis=1),
                )
                record(
                    "family",
                    f"{m}/{rho}",
                    "false_beyond",
                    ((abs(est) - radius > 0.1) & (abs(delta) <= 0.1)).any(axis=1),
                )

    def paired():
        # Joint cells (--,-+,+-,++), preserving empirical discordant-pair counts.
        for probs in (
            [0.4, 0.05, 0.15, 0.4],
            [0.05, 0.4, 0.5, 0.05],
            [0.25] * 4,
            [0.5, 0, 0, 0.5],
            [0.98, 0.005, 0.005, 0.01],
        ):
            for n in (12, 80):
                scenario = f"{probs}/{n}"
                values = {
                    k: []
                    for k in (
                        "paired_boot_coverage",
                        "unpaired_boot_coverage",
                        "paired_width",
                        "unpaired_width",
                        "paired_bound_coverage",
                        "bias",
                        "squared_error",
                    )
                }
                truth = 2 * (probs[1] - probs[2])
                for cells in rng.multinomial(n, probs, size=min(reps, 500)):
                    a = np.repeat([-1, -1, 1, 1], cells)
                    b = np.repeat([-1, 1, -1, 1], cells)
                    seed = int(rng.integers(2**32))
                    ci = api.bootstrap_interval(a, b, paired=True, draws=499, seed=seed)
                    ind = api.bootstrap_interval(
                        Sample("Z", cells[2] + cells[3], cells[0] + cells[1]),
                        Sample("Z", cells[1] + cells[3], cells[0] + cells[2]),
                        draws=499,
                        seed=seed,
                    )
                    bound = api.paired_compare(a, b, tolerance=0.1)["interval"]
                    err = (b - a).mean() - truth
                    for k, v in zip(
                        values,
                        (
                            ci[0] <= truth <= ci[1],
                            ind[0] <= truth <= ind[1],
                            ci[1] - ci[0],
                            ind[1] - ind[0],
                            bound[0] <= truth <= bound[1],
                            err,
                            err**2,
                        ),
                    ):
                        values[k].append(v)
                for k, v in values.items():
                    record("paired", scenario, k, v)
                record(
                    "paired",
                    scenario,
                    "RMSE",
                    [np.sqrt(np.mean(values["squared_error"]))],
                )

    def drift():
        from functools import lru_cache

        @lru_cache(None)
        def pv(k, n, j, m):
            return api.randomization_test(Sample("Z", k, n - k), Sample("Z", j, m - j))

        for n, m in [(32, 16), (128, 32)]:
            for kind in ("stationary", "abrupt", "gradual"):
                ref = rng.binomial(n, 0.1, reps)
                first = np.zeros(reps, int)
                for t in range(1, 13):
                    prob = (
                        0.1
                        if kind == "stationary" or t < 4
                        else (
                            0.55
                            if kind == "abrupt"
                            else min(0.6, 0.1 + 0.065 * (t - 3))
                        )
                    )
                    batch = rng.binomial(m, prob, reps)
                    pairs, inv = np.unique(
                        np.stack((ref, batch), axis=1), axis=0, return_inverse=True
                    )
                    p = np.array([pv(int(x), n, int(y), m) for x, y in pairs])[
                        inv.ravel()
                    ]
                    hit = (p <= 0.05 / (t * (t + 1))) & (first == 0)
                    first[hit] = t
                scenario = f"{n}/{m}/{kind}"
                record("drift", scenario, "alarm_probability", first > 0)
                if kind != "stationary":
                    record("drift", scenario, "false_alarm", (first > 0) & (first < 4))
                    record("drift", scenario, "detection_probability", first >= 4)
                    if np.any(first >= 4):
                        record(
                            "drift",
                            scenario,
                            "conditional_delay",
                            first[first >= 4] - 4,
                        )

    def hierarchy():
        for k in (3, 24):
            se = np.linspace(0.08, 0.4, k)
            for tau in (0, 0.1, 0.8):
                theta = rng.normal(0, tau, (reps, k))
                y = theta + rng.normal(0, se, (reps, k))
                for outlier in (False, True):
                    data = y.copy()
                    if outlier:
                        data[:, 0] += 2
                    w = 1 / se**2
                    fixed = (data * w).sum(axis=1) / w.sum()
                    q = ((data - fixed[:, None]) ** 2 * w).sum(axis=1)
                    t2 = np.maximum(
                        0, (q - k + 1) / (w.sum() - np.sum(w * w) / w.sum())
                    )
                    v = 1 / (se**2 + t2[:, None])
                    mu = (data * v).sum(axis=1) / v.sum(axis=1)
                    half = norm.ppf(0.975) / np.sqrt(v.sum(axis=1))
                    shrunk = mu[:, None] + t2[:, None] * v * (data - mu[:, None])
                    ref = api.hierarchical_effect(data[0], se)
                    np.testing.assert_allclose(
                        [mu[0], t2[0]],
                        [ref["mean"], ref["between_variance"]],
                        atol=1e-12,
                    )
                    for name, vals in [
                        ("bias", mu),
                        ("tau2_bias", t2 - tau * tau),
                        ("coverage", abs(mu) <= half),
                        ("shrinkage_mse", np.mean((shrunk - theta) ** 2, axis=1)),
                        ("raw_mse", np.mean((data - theta) ** 2, axis=1)),
                    ]:
                        record("hierarchy", f"{k}/{tau}/{outlier}", name, vals)

    def other():
        from scipy.integrate import quad

        for a, b in [
            (Sample("Z", 0, 5), Sample("Z", 1, 8)),
            (Sample("Z", 25, 25), Sample("Z", 28, 22)),
        ]:
            for x, y in [(0.5, 0.5), (1, 1), (10, 2), (0.2, 5)]:
                aa, bb = beta(a.plus + x, a.minus + y), beta(b.plus + x, b.minus + y)
                prob = quad(
                    lambda t, aa=aa, bb=bb: (
                        aa.pdf(t)
                        * (bb.cdf(min(1, t + 0.05)) - bb.cdf(max(0, t - 0.05)))
                    ),
                    0,
                    1,
                    points=[0.05, 0.95],
                )[0]
                r = api.bayesian_compare(a, b, prior=(x, y), tolerance=0.1, seed=47)
                scenario = f"{a.plus}/{b.plus}/{x}/{y}"
                record("bayes", scenario, "posterior_mass", [r["within_probability"]])
                record("bayes", scenario, "quadrature", [prob])
                record("bayes", scenario, "posterior_mcse", [r["mc_se"]])
        for pa, pb in [(0.5, 0.5), (0.01, 0.02), (0.5, 0.55), (0.2, 0.6)]:
            for n in (20, 100, 400):
                r = classification_power(pa, pb, n, n * 2, tolerance=0.1)
                for name in ("within", "beyond", "inconclusive", "correct"):
                    record("design", f"{pa}/{pb}/{n}", name, [getattr(r, name)])
        for probs in [[0.2, 0.3, 0.5], [0.02, 0.05, 0.93], [0, 0, 1]]:
            alt = np.array(probs)
            base = np.array([0.2, 0.3, 0.5])
            hit = []
            for a, b in zip(
                rng.multinomial(16, base, min(reps, 500)),
                rng.multinomial(27, alt, min(reps, 500)),
            ):
                r = api.distribution_compare(
                    dict(enumerate(a)),
                    dict(enumerate(b)),
                    draws=399,
                    seed=int(rng.integers(2**32)),
                )
                hit.append(r["pvalue"] <= 0.05)
            record("distribution", str(probs), "rejection_rate", hit)
        for p0, p1 in [(0, 0.001), (0.01, 0.04), (0.2, 0.05)]:
            n = 100
            k = rng.binomial(n, p0, reps)
            j = rng.binomial(n, p1, reps)

            def intervals(counts, true, n=n):
                from scipy.stats import binomtest

                bounds = [
                    binomtest(int(v), n).proportion_ci(0.975) for v in range(n + 1)
                ]
                return np.array([lo <= true <= hi for lo, hi in bounds])[counts]

            record(
                "readout",
                f"{p0}/{p1}",
                "joint_coverage",
                intervals(k, p0) & intervals(j, p1),
            )
            record(
                "readout",
                f"{p0}/{p1}",
                "contrast_bias",
                (1 - (k + j) / n) - (1 - p0 - p1),
            )
        for n, angle in [(10, 0), (1000, 0), (5000, 0.02), (5000, 1.57)]:
            from math import cos

            family = {}
            for label, p in [
                ("Z", (1 + cos(angle)) / 2),
                ("X", (1 + np.sin(angle)) / 2),
            ]:
                pa = 1 if label == "Z" else 0.5
                a = int(rng.binomial(n, pa))
                b = int(rng.binomial(n, p))
                family[label] = (Sample(label, a, n - a), Sample(label, b, n - b))
            status = api.regression_check(family, tolerance=0.2)["status"]
            record("regression", f"{n}/{angle}", status, [1])

    for name, fn in [
        ("sequential", sequential),
        ("multiplicity", multiple),
        ("paired", paired),
        ("drift", drift),
        ("hierarchy", hierarchy),
        ("other", other),
    ]:
        timed(name, fn)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["area", "scenario", "metric", "replicates", "estimate", "mc_se"]
        )
        writer.writerows(rows)
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "seed": 4703,
                "main_replicates": reps,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "runtime_seconds_and_peak_traced_bytes": timings,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    validate(40 if args.smoke else 10000, args.out)
