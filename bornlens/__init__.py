"""Compare independent, fixed-size samples of the same Pauli observable."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, log, sqrt
from numbers import Integral, Real


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _log_factor(confidence):
    confidence = _real(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    return log(2 / (1 - confidence))


@dataclass(frozen=True)
class Sample:
    """Pauli label in Qiskit order (rightmost factor acts on qubit 0)."""

    observable: str
    plus: int
    minus: int

    def __post_init__(self) -> None:
        label = self.observable
        if (
            not isinstance(label, str)
            or not label
            or any(p not in "IXYZ" for p in label)
        ):
            raise ValueError("observable must be a nonempty uppercase Pauli string")
        for name in ("plus", "minus"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.shots == 0:
            raise ValueError("sample is empty")
        if set(label) == {"I"} and self.minus:
            raise ValueError("identity has only positive outcomes")

    @property
    def shots(self) -> int:
        return self.plus + self.minus

    @property
    def mean(self) -> float:
        return (self.plus - self.minus) / self.shots


@dataclass(frozen=True)
class Comparison:
    observable: str
    mean_a: float
    mean_b: float
    delta: float
    interval: tuple
    tolerance: float
    confidence: float
    shots_a: int
    shots_b: int
    verdict: str
    method: str = "hoeffding"


def _verdict(lower, upper, tolerance):
    if lower >= -tolerance and upper <= tolerance:
        return "within_tolerance"
    if upper < -tolerance or lower > tolerance:
        return "beyond_tolerance"
    return "inconclusive"


def compare(
    a: Sample,
    b: Sample,
    *,
    tolerance: float,
    confidence: float = 0.95,
    method: str = "hoeffding",
) -> Comparison:
    """Return a two-sided Hoeffding interval and its tolerance verdict."""
    if not isinstance(a, Sample) or not isinstance(b, Sample):
        raise TypeError("both inputs must be Sample instances")
    if a.observable != b.observable:
        raise ValueError("observables must match")
    tolerance = _real(tolerance, "tolerance")
    if not 0 <= tolerance <= 2:
        raise ValueError("tolerance must be between zero and two")
    factor = _log_factor(confidence)
    delta = b.mean - a.mean
    radius = sqrt(2 * factor * (1 / a.shots + 1 / b.shots))
    if set(a.observable) == {"I"}:
        radius = 0.0
    lower, upper = max(-2, delta - radius), min(2, delta + radius)
    if method != "hoeffding":
        from .statistics import interval

        lower, upper = interval(a.plus, a.shots, b.plus, b.shots, confidence, method)
        lower, upper = float(lower), float(upper)
        if set(a.observable) == {"I"}:
            lower = upper = 0.0
    return Comparison(
        a.observable,
        a.mean,
        b.mean,
        delta,
        (lower, upper),
        tolerance,
        confidence,
        a.shots,
        b.shots,
        _verdict(lower, upper, tolerance),
        method,
    )


def shots_per_circuit(half_width: float, *, confidence: float = 0.95) -> int:
    """Equal fixed sample sizes sufficient for the requested interval precision."""
    half_width = _real(half_width, "half_width")
    if not 0 < half_width <= 2:
        raise ValueError("half_width must be in (0, 2]")
    return ceil(4 * _log_factor(confidence) / half_width / half_width)
