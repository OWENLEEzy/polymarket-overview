import math
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import NormalDist, median, pstdev

import numpy as np

from app.analysis.aggregators import threshold_cdf_probability
from app.analysis.parsers import (
    UNIT_TO_BILLIONS,
    MoneyBracket,
    canonical_unit,
    parse_deadline_date,
    parse_money_bracket,
    to_billions,
)
from app.models import ObjectType, PointEstimate, StructuredMarket

_NORMAL = NormalDist()
_HIGH_TAIL_THRESHOLD = 0.30
_MIN_R_SQUARED = 0.85


@dataclass(frozen=True)
class Interval:
    lower: float | None  # canonical billions
    upper: float | None  # canonical billions
    probability: float


@dataclass
class NumericEstimate:
    expected_value: float | None
    p10: float | None
    p50: float | None
    p90: float | None
    unit: str
    fit_method: str
    anomalies: list[str] = field(default_factory=list)


def numeric_estimate(intervals: list[Interval], unit: str) -> NumericEstimate:
    total = sum(item.probability for item in intervals)
    normalized = [
        Interval(item.lower, item.upper, item.probability / total)
        for item in intervals
        if total > 0
    ]
    anomalies: list[str] = []
    if len(normalized) < 2:
        anomalies.append("insufficient_data")
        return NumericEstimate(None, None, None, None, unit, "midpoint", anomalies)

    tail = next((item for item in normalized if item.upper is None), None)
    if tail is not None and tail.probability > _HIGH_TAIL_THRESHOLD:
        anomalies.append("high_tail_probability")

    fit = _fit_lognormal(normalized)
    if fit is not None:
        mu, sigma = fit
        return NumericEstimate(
            expected_value=_from_billions(math.exp(mu + sigma * sigma / 2), unit),
            p10=_from_billions(math.exp(mu + sigma * _NORMAL.inv_cdf(0.10)), unit),
            p50=_from_billions(math.exp(mu + sigma * _NORMAL.inv_cdf(0.50)), unit),
            p90=_from_billions(math.exp(mu + sigma * _NORMAL.inv_cdf(0.90)), unit),
            unit=unit,
            fit_method="lognormal",
            anomalies=anomalies,
        )

    anomalies.append("lognormal_fit_failed")
    expected = _midpoint_expected_value(normalized)
    return NumericEstimate(
        expected_value=_from_billions(expected, unit),
        p10=None,
        p50=_from_billions(expected, unit),
        p90=None,
        unit=unit,
        fit_method="midpoint",
        anomalies=anomalies,
    )


def _fit_lognormal(intervals: list[Interval]) -> tuple[float, float] | None:
    cumulative = 0.0
    thresholds: list[float] = []
    cdfs: list[float] = []
    for item in sorted(intervals, key=lambda i: (i.upper is None, i.upper or 0.0)):
        cumulative += item.probability
        if item.upper is not None and item.upper > 0 and 0.0 < cumulative < 1.0:
            thresholds.append(item.upper)
            cdfs.append(cumulative)
    if len(thresholds) < 3:
        return None
    y = np.log(np.array(thresholds))
    z = np.array([_NORMAL.inv_cdf(c) for c in cdfs])
    # Index rather than unpack the ndarray so mypy --strict doesn't demand an
    # annotation for the iterable. polyfit(deg=1) returns [slope, intercept].
    coefficients = np.polyfit(z, y, 1)
    sigma = float(coefficients[0])
    mu = float(coefficients[1])
    if sigma <= 0:
        return None
    predicted = mu + sigma * z
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if r_squared < _MIN_R_SQUARED:
        return None
    return mu, sigma


def _midpoint_expected_value(intervals: list[Interval]) -> float:
    total = 0.0
    for item in intervals:
        if item.lower is not None and item.upper is not None:
            representative = (item.lower + item.upper) / 2
        elif item.upper is not None:  # open lower tail
            representative = item.upper / 2
        elif item.lower is not None:  # open upper tail
            representative = item.lower * 1.5
        else:
            representative = 0.0
        total += item.probability * representative
    return total


def _from_billions(value: float, unit: str) -> float:
    factor = UNIT_TO_BILLIONS.get(canonical_unit(unit), 1.0)
    return round(value / factor, 4)


_HETEROGENEOUS_STD = 0.30


def boolean_estimate(probabilities: list[float]) -> tuple[float | None, list[str]]:
    if not probabilities:
        return None, ["insufficient_data"]
    anomalies: list[str] = []
    if len(probabilities) > 1 and pstdev(probabilities) > _HETEROGENEOUS_STD:
        anomalies.append("heterogeneous_group")
    return round(median(probabilities), 6), anomalies


def argmax_estimate(grouped: dict[str, list[float]]) -> tuple[str | None, float | None]:
    if not grouped:
        return None, None
    averages = {name: sum(values) / len(values) for name, values in grouped.items()}
    total = sum(averages.values())
    if total <= 0:
        return None, None
    normalized = {name: value / total for name, value in averages.items()}
    top = max(normalized, key=lambda name: normalized[name])
    return top, round(normalized[top], 6)


def median_date_estimate(
    dated_probabilities: list[tuple[str, float]],
) -> tuple[str | None, list[str]]:
    if not dated_probabilities:
        return None, ["insufficient_data"]
    ordered = sorted(dated_probabilities, key=lambda item: item[0])
    cumulative = 0.0
    for date, probability in ordered:
        cumulative += probability
        if cumulative >= 0.5:
            return date, []
    return ordered[-1][0], ["median_date_not_reached"]


_RESIDUAL_MASS_THRESHOLD = 0.05


def compute_point_estimate(
    object_name: str,
    object_type: ObjectType,
    markets: list[StructuredMarket],
) -> PointEstimate:
    included = [market for market in markets if market.include]
    object_id = included[0].object_id if included else None
    if object_type == ObjectType.BOOLEAN:
        estimate = _boolean_point_estimate(object_name, object_type, included)
    elif object_type == ObjectType.TIME:
        estimate = _date_point_estimate(object_name, object_type, included)
    elif object_type == ObjectType.CONTINUOUS:
        estimate = _continuous_point_estimate(object_name, object_type, included)
    else:
        estimate = _categorical_point_estimate(object_name, object_type, included)
    # Stamp the source object_id so the frontend can fetch the matching
    # distribution unambiguously even when two objects share a display name.
    return estimate.model_copy(update={"object_id": object_id})


def _boolean_point_estimate(
    object_name: str, object_type: ObjectType, markets: list[StructuredMarket]
) -> PointEstimate:
    value, anomalies = boolean_estimate([m.probability for m in markets])
    return PointEstimate(
        object_name=object_name,
        object_type=object_type,
        role="boolean",
        boolean_probability=value,
        anomalies=anomalies,
        fit_method="median",
    )


def _date_point_estimate(
    object_name: str, object_type: ObjectType, markets: list[StructuredMarket]
) -> PointEstimate:
    dated: list[tuple[str, float]] = []
    for market in markets:
        date = (market.resolution_date or "").split("T")[0] or parse_deadline_date(
            market.category_value or market.question
        )
        if date:
            dated.append((date, market.probability))
    median_date, anomalies = median_date_estimate(dated)
    return PointEstimate(
        object_name=object_name,
        object_type=object_type,
        role="time",
        median_date=median_date,
        anomalies=anomalies,
        fit_method="median",
    )


def _categorical_point_estimate(
    object_name: str, object_type: ObjectType, markets: list[StructuredMarket]
) -> PointEstimate:
    brackets = [(parse_money_bracket(m.category_value), m.probability) for m in markets]
    parseable = [(bracket, prob) for bracket, prob in brackets if bracket is not None]
    if len(parseable) >= 2:
        dropped_mass = sum(prob for bracket, prob in brackets if bracket is None)
        return _numeric_from_brackets(object_name, object_type, parseable, dropped_mass)

    grouped: dict[str, list[float]] = defaultdict(list)
    for market in markets:
        grouped[market.category_value or market.question].append(market.probability)
    top, probability = argmax_estimate(grouped)
    return PointEstimate(
        object_name=object_name,
        object_type=object_type,
        role="context",
        top_category=top,
        top_category_probability=probability,
        anomalies=[],
        fit_method="argmax",
    )


def _numeric_from_brackets(
    object_name: str,
    object_type: ObjectType,
    parseable: list[tuple[MoneyBracket, float]],
    dropped_mass: float = 0.0,
) -> PointEstimate:
    unit = _dominant_unit([bracket.unit for bracket, _ in parseable])
    intervals = [
        Interval(
            lower=to_billions(bracket.lower, bracket.unit) if bracket.lower is not None else None,
            upper=to_billions(bracket.upper, bracket.unit) if bracket.upper is not None else None,
            probability=prob,
        )
        for bracket, prob in parseable
    ]
    estimate = numeric_estimate(intervals, unit)
    point = _numeric_to_point_estimate(object_name, object_type, estimate)
    # Non-numeric buckets (e.g. "No IPO") are excluded from the fit; if their
    # combined probability is material, flag it rather than silently dropping it.
    if dropped_mass > _RESIDUAL_MASS_THRESHOLD:
        return point.model_copy(update={"anomalies": [*point.anomalies, "residual_mass_dropped"]})
    return point


def _continuous_point_estimate(
    object_name: str, object_type: ObjectType, markets: list[StructuredMarket]
) -> PointEstimate:
    points = sorted(
        [
            (
                to_billions(m.threshold_value, m.threshold_unit or "B"),
                threshold_cdf_probability(m.operator, m.probability),
            )
            for m in markets
            if m.threshold_value is not None
        ],
        key=lambda item: item[0],
    )
    intervals: list[Interval] = []
    previous_threshold: float | None = None
    previous_cdf = 0.0
    for threshold, cdf in points:
        intervals.append(
            Interval(
                lower=previous_threshold, upper=threshold, probability=max(cdf - previous_cdf, 0.0)
            )
        )
        previous_threshold = threshold
        previous_cdf = cdf
    if points:
        intervals.append(
            Interval(lower=points[-1][0], upper=None, probability=max(1.0 - points[-1][1], 0.0))
        )
    estimate = numeric_estimate(intervals, "B")
    point = _numeric_to_point_estimate(object_name, object_type, estimate)
    # A CDF that decreases as the threshold rises is a market inconsistency.
    # max(cdf - previous_cdf, 0.0) above hides it from the fit, so detect it on
    # the raw CDF series and surface it (the detail view flags the same thing).
    if any(later[1] < earlier[1] for earlier, later in zip(points, points[1:])):
        return point.model_copy(update={"anomalies": [*point.anomalies, "monotonicity_conflict"]})
    return point


def _numeric_to_point_estimate(
    object_name: str, object_type: ObjectType, estimate: NumericEstimate
) -> PointEstimate:
    return PointEstimate(
        object_name=object_name,
        object_type=object_type,
        role="numeric",
        expected_value=estimate.expected_value,
        p10=estimate.p10,
        p50=estimate.p50,
        p90=estimate.p90,
        unit=estimate.unit,
        anomalies=estimate.anomalies,
        fit_method=estimate.fit_method,  # type: ignore[arg-type]
    )


def _dominant_unit(units: list[str]) -> str:
    if not units:
        return "B"
    return max(set(units), key=units.count)


def describe_point_estimates(point_estimates: list[PointEstimate]) -> str:
    lines: list[str] = []
    for estimate in ordered_by_role(point_estimates):
        if estimate.role == "boolean" and estimate.boolean_probability is not None:
            line = (
                f"- [boolean] {estimate.object_name}: {round(estimate.boolean_probability * 100)}%"
            )
        elif estimate.role == "time" and estimate.median_date:
            line = f"- [time] {estimate.object_name}: median {estimate.median_date}"
        elif estimate.role == "numeric" and estimate.expected_value is not None:
            line = f"- [numeric] {estimate.object_name}: ~{estimate.expected_value}{estimate.unit or ''}"
        elif estimate.role == "context" and estimate.top_category:
            line = f"- [context] {estimate.object_name}: {estimate.top_category} ({round((estimate.top_category_probability or 0) * 100)}%)"
        else:
            # No representable value (e.g. a TIME object with no resolvable date).
            # Still emit a line so anomalies attach to THIS estimate, never to the
            # previous one — and never index an empty `lines` list.
            line = f"- [{estimate.role}] {estimate.object_name}: insufficient data"
        if estimate.anomalies:
            line += f" [anomalies: {', '.join(estimate.anomalies)}]"
        lines.append(line)
    return "\n".join(lines)


_ROLE_ORDER = {"boolean": 0, "time": 1, "numeric": 2, "context": 3}


def ordered_by_role(point_estimates: list[PointEstimate]) -> list[PointEstimate]:
    return sorted(point_estimates, key=lambda pe: _ROLE_ORDER.get(pe.role, 9))
