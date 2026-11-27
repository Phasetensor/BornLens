"""Binomial interval and fixed-design power calculations (install the stats extra)."""
from __future__ import annotations
from typing import Any, Optional, Sequence

from dataclasses import dataclass
from math import sqrt
import numpy as np
from scipy.special import betaincinv, ndtri
from scipy.stats import binom
from . import _integer, _log_factor, _real


def _bounds(k, n, alpha, method):
    p = np.asarray(k) / n
    if method == "newcombe":
        z = -ndtri(alpha / 2)
        den = 1 + z*z/n
        center = (p + z*z/(2*n)) / den
        radius = z*np.sqrt(p*(1-p)/n + (z/(2*n))**2) / den
        return np.where(p == 0, 0, center-radius), np.where(p == 1, 1, center+radius)
    tail = alpha / 4  # two marginal intervals, each with error alpha/2
    lo = np.where(p == 0, 0, betaincinv(k, n-k+1, tail))
    hi = np.where(p == 1, 1, 1-betaincinv(n-k, k+1, tail))
    return lo, hi


def interval(ka: Any, na: int, kb: Any, nb: int, confidence: float, method: str) -> tuple[Any,Any]:
    factor = _log_factor(confidence)
    a, b = np.asarray(ka)/na, np.asarray(kb)/nb
    d = (2*kb-nb)/nb - (2*ka-na)/na
    if method == "hoeffding":
        r = sqrt(2*factor*(1/na + 1/nb))
        return np.maximum(-2, d-r), np.minimum(2, d+r)
    if method not in ("newcombe", "exact"):
        raise ValueError("method must be hoeffding, newcombe or exact")
    al, au = _bounds(ka, na, 1-confidence, method)
    bl, bu = _bounds(kb, nb, 1-confidence, method)
    if method == "exact":
        return 2*(bl-au), 2*(bu-al)
    return (np.maximum(-2, d-2*np.hypot(b-bl, au-a)),
            np.minimum(2, d+2*np.hypot(bu-b, a-al)))


@dataclass(frozen=True)
class Power:
    within: float
    beyond: float
    inconclusive: float
    correct: float


def classification_power(p_a: float, p_b: float, n_a: int, n_b: int, *, tolerance: float, confidence: float=.95,
                         method: str="hoeffding") -> Power:
    """Sum joint binomial probabilities. Inputs p_a,p_b are P(+1)."""
    pa, pb = _real(p_a, "p_a"), _real(p_b, "p_b")
    tol = _real(tolerance, "tolerance")
    if not (0 <= pa <= 1 and 0 <= pb <= 1 and 0 <= tol <= 2):
        raise ValueError("probabilities must lie in [0,1] and tolerance in [0,2]")
    na, nb = _integer(n_a, "n_a", 1), _integer(n_b, "n_b", 1)
    # Interval endpoints are nondecreasing in kb for fixed ka. Locate the
    # decision boundaries by binary search; then integrate each binomial tail.
    ka = np.arange(na+1)
    weights = binom.pmf(ka, na, pa)

    def first(endpoint, threshold, strict):
        left, right = np.zeros_like(ka), np.full_like(ka, nb+1)
        while np.any(left < right):
            mid = (left+right)//2
            values = interval(ka, na, np.minimum(mid, nb), nb, confidence, method)[endpoint]
            hit = values > threshold if strict else values >= threshold
            active = left < right
            right = np.where(active & hit, mid, right)
            left = np.where(active & ~hit, mid+1, left)
        return left

    start = first(0, -tol, False)
    stop = first(1, tol, True)
    within = np.maximum(0, np.where(start > nb*pb,
        binom.sf(start-1, nb, pb)-binom.sf(stop-1, nb, pb),
        binom.cdf(stop-1, nb, pb)-binom.cdf(start-1, nb, pb)))
    beyond = (binom.cdf(first(1, -tol, False)-1, nb, pb)
              + binom.sf(first(0, tol, True)-1, nb, pb))
    w, b = float(weights @ within), float(weights @ beyond)
    # Decimal inputs at the tolerance boundary use their stated decimal values.
    from decimal import Decimal
    inside = 2*abs(Decimal(str(pb))-Decimal(str(pa))) <= Decimal(str(tol))
    return Power(w, b, max(0., 1-w-b), w if inside else b)


def plan_classification(p_a: float, p_b: float, designs: Sequence[tuple[int,int]], *, target: float, **settings: Any) -> Optional[tuple[tuple[int,int],Power]]:
    """Return lowest total shots meeting target among supplied (n_a,n_b) designs."""
    target = _real(target, "target")
    if not 0 < target <= 1:
        raise ValueError("target must be in (0,1]")
    for a, b in sorted(designs, key=lambda pair: (sum(pair), pair)):
        result = classification_power(p_a, p_b, a, b, **settings)
        if result.correct >= target:
            return (a, b), result
    return None
