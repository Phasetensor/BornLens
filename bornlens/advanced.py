"""Advanced finite-shot analysis. Install bornlens[stats]."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from math import log, sqrt
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, fisher_exact, norm

from . import Comparison, Sample, _integer, _real, _verdict, compare
from .statistics import Power, classification_power


def _prob(value, name):
    value = _real(value, name)
    if not 0 < value < 1:
        raise ValueError(f"{name} must be in (0,1)")
    return value


def _tol(value):
    value = _real(value, "tolerance")
    if not 0 <= value <= 2:
        raise ValueError("tolerance must be in [0,2]")
    return value


def _pair(a, b):
    if (
        not isinstance(a, Sample)
        or not isinstance(b, Sample)
        or a.observable != b.observable
    ):
        raise ValueError("expected samples with matching Pauli labels")


def adjust_pvalues(values: Sequence[float], method: str = "holm") -> list[float]:
    """Adjusted p-values in input order. fdr_bh and fdr_by control FDR."""
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not p.size or not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)):
        raise ValueError("expected a nonempty vector of p-values")
    order = np.argsort(p)
    ranks = np.arange(1, len(p) + 1)
    ordered = p[order]
    if method == "bonferroni":
        adjusted = ordered * len(p)
    elif method == "holm":
        adjusted = np.maximum.accumulate(ordered * (len(p) + 1 - ranks))
    elif method in ("fdr_bh", "fdr_by"):
        factor = np.sum(1 / ranks) if method == "fdr_by" else 1
        adjusted = np.minimum.accumulate((ordered * len(p) * factor / ranks)[::-1])[
            ::-1
        ]
    else:
        raise ValueError("unknown multiple testing method")
    result = np.empty_like(p)
    result[order] = np.minimum(1, adjusted)
    return result.tolist()


def compare_many(
    pairs: Mapping[str, tuple[Sample, Sample]],
    *,
    tolerance: float,
    confidence: float = 0.95,
    method: str = "hoeffding",
) -> dict[str, Comparison]:
    """Simultaneous Bonferroni intervals, valid under arbitrary across-label dependence."""
    if (
        not isinstance(pairs, Mapping)
        or not pairs
        or method not in ("hoeffding", "exact")
    ):
        raise ValueError("provide a nonempty family and a finite-sample method")
    marginal = 1 - (1 - _prob(confidence, "confidence")) / len(pairs)
    results = {}
    for label, (a, b) in pairs.items():
        _pair(a, b)
        if label != a.observable:
            raise ValueError("family key must match the observable")
        results[label] = compare(
            a, b, tolerance=tolerance, confidence=marginal, method=method
        )
    return results


def regression_check(
    pairs: Mapping[str, tuple[Sample, Sample]], **settings: Any
) -> dict[str, Any]:
    results = compare_many(pairs, **settings)
    verdicts = {r.verdict for r in results.values()}
    status = (
        "fail"
        if "beyond_tolerance" in verdicts
        else ("pass" if verdicts == {"within_tolerance"} else "inconclusive")
    )
    return {"status": status, "observables": results}


def sequential_compare(
    a: Sample, b: Sample, *, tolerance: float, confidence: float = 0.95
) -> Comparison:
    """Anytime-valid at every matched cumulative shot count n, for one fixed pair."""
    _pair(a, b)
    n = a.shots
    if n != b.shots:
        raise ValueError("matched cumulative shot counts required")
    alpha = 1 - _prob(confidence, "confidence")
    tolerance = _tol(tolerance)
    d = b.mean - a.mean
    # Spend alpha/(n*(n+1)) at cumulative n. Sum over all n is alpha.
    r = sqrt(4 * (log(2 / alpha) + log(n) + log(n + 1)) / n)
    lo, hi = max(-2, d - r), min(2, d + r)
    if set(a.observable) == {"I"}:
        lo = hi = 0.0
    return Comparison(
        a.observable,
        a.mean,
        b.mean,
        d,
        (lo, hi),
        tolerance,
        confidence,
        n,
        n,
        _verdict(lo, hi, tolerance),
        "sequential_hoeffding",
    )


def bayesian_compare(
    a: Sample,
    b: Sample,
    *,
    tolerance: float,
    prior: tuple[float, float] = (0.5, 0.5),
    confidence: float = 0.95,
    draws: int = 50000,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Independent beta posteriors on plus probabilities; posterior Monte Carlo."""
    _pair(a, b)
    tolerance, confidence = _tol(tolerance), _prob(confidence, "confidence")
    x, y = (_real(v, "prior shape") for v in prior)
    if min(x, y) <= 0:
        raise ValueError("beta prior shapes must be positive")
    draws = _integer(draws, "draws", 2)
    rng = np.random.default_rng(seed)
    d = 2 * (
        rng.beta(x + b.plus, y + b.minus, draws)
        - rng.beta(x + a.plus, y + a.minus, draws)
    )
    if set(a.observable) == {"I"}:
        d.fill(0)
    within = float(np.mean(abs(d) <= tolerance))
    tail = (1 - confidence) / 2
    return {
        "mean": float(d.mean()),
        "credible_interval": np.quantile(d, [tail, 1 - tail]).tolist(),
        "within_probability": within,
        "mc_se": sqrt(within * (1 - within) / draws),
        "draws": draws,
        "seed": seed,
    }


def _paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if (
        a.ndim != 1
        or a.size == 0
        or a.shape != b.shape
        or not (np.isin(a, [-1, 1]).all() and np.isin(b, [-1, 1]).all())
    ):
        raise ValueError("expected aligned nonempty +/-1 vectors")
    return b.astype(float) - a.astype(float)


def paired_compare(
    a: Sequence[int], b: Sequence[int], *, tolerance: float, confidence: float = 0.95
) -> dict[str, Any]:
    d = _paired(a, b)
    tolerance, confidence = _tol(tolerance), _prob(confidence, "confidence")
    radius = sqrt(8 * log(2 / (1 - confidence)) / len(d))
    mean = float(d.mean())
    lo, hi = max(-2, mean - radius), min(2, mean + radius)
    return {
        "delta": mean,
        "interval": (lo, hi),
        "pairs": len(d),
        "verdict": _verdict(lo, hi, tolerance),
    }


def bootstrap_interval(
    a: Any,
    b: Any,
    *,
    paired: bool = False,
    confidence: float = 0.95,
    draws: int = 10000,
    seed: int | None = 0,
) -> tuple[float, float]:
    """Percentile bootstrap of the mean change, using the empirical discrete laws."""
    rng = np.random.default_rng(seed)
    tail = (1 - _prob(confidence, "confidence")) / 2
    draws = _integer(draws, "draws", 2)
    if paired:
        d = _paired(a, b)
        probs = np.array([(d == k).mean() for k in (-2, 0, 2)])
        counts = rng.multinomial(len(d), probs, size=draws)
        delta = 2 * (counts[:, 2] - counts[:, 0]) / len(d)
    else:
        _pair(a, b)
        delta = 2 * (
            rng.binomial(b.shots, b.plus / b.shots, draws) / b.shots
            - rng.binomial(a.shots, a.plus / a.shots, draws) / a.shots
        )
    return tuple(np.quantile(delta, [tail, 1 - tail]).tolist())


def randomization_test(a: Any, b: Any, *, paired: bool = False) -> float:
    """Exact two-sided test of exchangeability for binary outcomes."""
    if paired:
        d = _paired(a, b)
        discordant = int(np.count_nonzero(d))
        return (
            float(binomtest(int((d > 0).sum()), discordant).pvalue)
            if discordant
            else 1.0
        )
    _pair(a, b)
    return float(fisher_exact([[a.plus, a.minus], [b.plus, b.minus]])[1])


def distribution_compare(
    a: Mapping[str, int],
    b: Mapping[str, int],
    *,
    draws: int = 10000,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Total variation and a conditional random-label test for common categories."""
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        raise TypeError("use count mappings")
    labels = sorted(set(a) | set(b))
    rows = [[_integer(m.get(k, 0), "count") for k in labels] for m in (a, b)]
    if sum(map(sum, rows)) >= 10**9:
        raise ValueError("permutation sampler requires pooled count below one billion")
    counts = np.asarray(rows, dtype=np.int64)
    na, nb = counts.sum(axis=1)
    if min(na, nb) <= 0:
        raise ValueError("empty distribution")
    totals = counts.sum(axis=0)
    totals = totals[totals > 0]
    observed = float(np.abs(counts[0] / na - counts[1] / nb).sum() / 2)
    rng = np.random.default_rng(seed)
    draws = _integer(draws, "draws", 1)
    extreme = 0
    for start in range(0, draws, 256):
        x = rng.multivariate_hypergeometric(
            totals, int(na), size=min(256, draws - start)
        )
        tv = np.abs(x / na - (totals - x) / nb).sum(axis=1) / 2
        extreme += int((tv >= observed - 1e-14).sum())
    return {
        "total_variation": observed,
        "pvalue": (extreme + 1) / (draws + 1),
        "draws": draws,
        "seed": seed,
    }


def readout_diagnostics(
    prepared_zero: Sample, prepared_one: Sample, *, confidence: float = 0.95
) -> dict[str, Any]:
    """Single-qubit Z calibration: conditional flip rates and readout contrast."""
    _pair(prepared_zero, prepared_one)
    if prepared_zero.observable != "Z":
        raise ValueError("expected single-qubit Z calibration")
    confidence = _prob(confidence, "confidence")
    out: dict[str, Any] = {}
    for key, k, n in (
        ("zero_to_one", prepared_zero.minus, prepared_zero.shots),
        ("one_to_zero", prepared_one.plus, prepared_one.shots),
    ):
        ci = binomtest(k, n).proportion_ci(1 - (1 - confidence) / 2)
        out[key] = {"rate": k / n, "interval": (ci.low, ci.high)}
    out["contrast"] = 1 - out["zero_to_one"]["rate"] - out["one_to_zero"]["rate"]
    return out


def drift_monitor(
    reference: Sample, windows: Sequence[Sample], *, alpha: float = 0.05
) -> list[dict[str, Any]]:
    """Compare fixed disjoint time batches with a fixed reference, spending alpha."""
    alpha = _prob(alpha, "alpha")
    history = []
    for t, sample in enumerate(windows, 1):
        p = randomization_test(reference, sample)
        threshold = alpha / (t * (t + 1))
        history.append(
            {"window": t, "pvalue": p, "threshold": threshold, "alarm": p <= threshold}
        )
    return history


def effect_size(a: Sample, b: Sample, *, tolerance: float) -> dict[str, float]:
    _pair(a, b)
    tolerance = _tol(tolerance)
    pa, pb = a.plus / a.shots, b.plus / b.shots
    return {
        "delta": b.mean - a.mean,
        "risk_difference": pb - pa,
        "cohens_h": float(2 * (np.arcsin(np.sqrt(pb)) - np.arcsin(np.sqrt(pa)))),
        "margin_excess": abs(b.mean - a.mean) - tolerance,
    }


def optimize_design(
    p_a: float,
    p_b: float,
    allocations: Sequence[tuple[int, int]],
    *,
    budget: int,
    **settings: Any,
) -> tuple[tuple[int, int], Power] | None:
    """Maximize correct classification among admissible candidate designs."""
    budget = _integer(budget, "budget", 2)
    best = None
    for na, nb in allocations:
        na, nb = _integer(na, "n_a", 1), _integer(nb, "n_b", 1)
        if na + nb > budget:
            continue
        power = classification_power(p_a, p_b, na, nb, **settings)
        if best is None or (power.correct, -na - nb) > (best[1].correct, -sum(best[0])):
            best = ((na, nb), power)
    return best


def hierarchical_effect(
    effects: Sequence[float],
    standard_errors: Sequence[float],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Normal random-effects meta-analysis; DerSimonian-Laird moment estimate."""
    y, se = np.asarray(effects, dtype=float), np.asarray(standard_errors, dtype=float)
    if (
        y.ndim != 1
        or y.size < 2
        or se.shape != y.shape
        or not (np.isfinite(y).all() and np.isfinite(se).all() and (se > 0).all())
    ):
        raise ValueError(
            "provide at least two independent effects with positive standard errors"
        )
    y = y.astype(np.longdouble)
    se = se.astype(np.longdouble)
    w = 1 / se**2
    fixed = np.average(y, weights=w)
    q = np.sum(w * (y - fixed) ** 2)
    tau2 = max(
        0.0, float((q - len(y) + 1) / (2 * np.sum(w[1:] * np.cumsum(w[:-1])) / w.sum()))
    )
    if not np.isfinite(tau2):
        raise ValueError("between-study variance exceeds floating-point range")
    weights = 1 / (se**2 + tau2)
    mu = float(np.average(y, weights=weights))
    radius = -norm.ppf((1 - _prob(confidence, "confidence")) / 2) / np.sqrt(
        weights.sum()
    )
    return {
        "mean": mu,
        "between_variance": tau2,
        "interval": (float(mu - radius), float(mu + radius)),
        "shrunk_effects": (mu + tau2 * weights * (y - mu)).astype(float).tolist(),
    }


def write_report(path: str | Path, *, inputs: Any, settings: Any, results: Any) -> None:
    """Write machine-readable inputs, analysis configuration and results as strict JSON."""
    from importlib.metadata import version

    def encode(value):
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported report type {type(value).__name__}")

    report = {
        "schema_version": 1,
        "versions": {p: version(p) for p in ("bornlens", "numpy", "scipy")},
        "inputs": inputs,
        "settings": settings,
        "results": results,
    }
    content = json.dumps(
        report, default=encode, sort_keys=True, indent=2, allow_nan=False
    )
    Path(path).write_text(content + "\n", encoding="utf-8")
