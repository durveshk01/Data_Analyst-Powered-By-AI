"""Trend, growth, period comparison and driver (contribution) analysis.

Two correctness rules run through this module:

1. **Never compare an incomplete period with a complete one.** If the data
   ends on 10 October, October is not "down 68%" — it is a third of a month.
   :func:`build_series` detects partial first/last periods from the data's real
   coverage, analyses that need complete periods drop them, and
   :func:`compare_periods` switches to a like-for-like window (e.g. 2026
   Jan–Sep vs 2025 Jan–Sep) instead of comparing nine months with twelve.

2. **"Why" is arithmetic, not narrative.** :func:`explain_change` splits a
   period-over-period change exactly across the values of a dimension, and when
   no dimension is named it tries several and picks the one where the change is
   most concentrated — so the answer to "why did revenue fall?" is a ranked
   decomposition with a checkable sum.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.engine.query import apply_filters, resolve_column, time_series
from app.engine.types import Aggregation, FilterSpec, TimeGrain
from app.engine.utils import compact_number, guess_unit, pct_change, safe_div, safe_float
from app.errors import AnalysisError, BadRequestError
from app.obs.logging import get_logger

log = get_logger(__name__)

_MIN_TREND_POINTS = 4
_ADDITIVE_AGGS = {Aggregation.SUM, Aggregation.COUNT}

MONTH_TOKENS: dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

# How far short of a period boundary the data may stop and the period still
# count as complete. A month whose last record is on the 29th is complete;
# one whose last record is on the 10th is not.
_COMPLETENESS_TOLERANCE: dict[TimeGrain, pd.Timedelta] = {
    TimeGrain.HOUR: pd.Timedelta(minutes=5),
    TimeGrain.DAY: pd.Timedelta(hours=1),
    TimeGrain.WEEK: pd.Timedelta(days=1),
    TimeGrain.MONTH: pd.Timedelta(days=2),
    TimeGrain.QUARTER: pd.Timedelta(days=5),
    TimeGrain.YEAR: pd.Timedelta(days=14),
}


# ------------------------------------------------------------ period helpers --
def period_offset(grain: TimeGrain) -> pd.DateOffset:
    return {
        TimeGrain.HOUR: pd.DateOffset(hours=1),
        TimeGrain.DAY: pd.DateOffset(days=1),
        TimeGrain.WEEK: pd.DateOffset(weeks=1),
        TimeGrain.MONTH: pd.DateOffset(months=1),
        TimeGrain.QUARTER: pd.DateOffset(months=3),
        TimeGrain.YEAR: pd.DateOffset(years=1),
    }[grain]


def period_end(start: pd.Timestamp, grain: TimeGrain) -> pd.Timestamp:
    """Exclusive end of the period that begins at `start`."""
    return pd.Timestamp(start) + period_offset(grain)


def _periods_to_labels(series: pd.Series, grain: TimeGrain) -> list[str]:
    ts = pd.to_datetime(series)
    if grain == TimeGrain.HOUR:
        return [t.strftime("%Y-%m-%d %H:00") for t in ts]
    if grain == TimeGrain.DAY:
        return [t.strftime("%Y-%m-%d") for t in ts]
    if grain == TimeGrain.WEEK:
        # ISO week *and ISO year*: 29 Dec 2025 is W01 of 2026, not of 2025.
        return [
            f"W{t.isocalendar().week:02d} {t.isocalendar().year}" for t in ts
        ]
    if grain == TimeGrain.MONTH:
        return [t.strftime("%b %Y") for t in ts]
    if grain == TimeGrain.QUARTER:
        return [f"Q{((t.month - 1) // 3) + 1} {t.year}" for t in ts]
    return [str(t.year) for t in ts]


def data_coverage(
    df: pd.DataFrame, time_column: str, filters: list[FilterSpec] | None = None
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(covered_from, covered_until) — the real time span the data covers.

    Day-precision data (every timestamp at midnight) covers the *whole* of its
    last day, so `covered_until` is pushed to the following midnight.
    """
    frame, _ = apply_filters(df, filters or [])
    tcol = resolve_column(time_column, frame)
    times = frame[tcol]
    if not pd.api.types.is_datetime64_any_dtype(times):
        times = pd.to_datetime(times, errors="coerce")
    times = times.dropna()
    if times.empty:
        return None
    start, end = times.min(), times.max()
    if bool((times.dt.normalize() == times).all()):
        end = end + pd.Timedelta(days=1)
    return pd.Timestamp(start), pd.Timestamp(end)


def build_series(
    df: pd.DataFrame,
    *,
    time_column: str,
    value_column: str,
    grain: TimeGrain,
    agg: Aggregation = Aggregation.SUM,
    filters: list[FilterSpec] | None = None,
    complete_only: bool = False,
) -> pd.DataFrame:
    """A regular time series with `period`, `label` and `value` columns.

    `frame.attrs` records coverage: `partial_first`, `partial_last`,
    `covered_from`, `covered_until` and, when `complete_only` dropped anything,
    `excluded_partial` (the labels removed). Analyses that compare periods ask
    for `complete_only=True`; charts keep every period.
    """
    frame = time_series(
        df,
        time_column=time_column,
        value_column=value_column,
        grain=grain,
        agg=agg,
        filters=filters,
    )
    if frame.empty:
        raise AnalysisError(
            f"No dated rows available to build a {grain.label} series of "
            f"'{value_column}' over '{time_column}'."
        )
    frame["period"] = pd.to_datetime(frame["period"])
    frame["label"] = _periods_to_labels(frame["period"], grain)

    coverage = data_coverage(df, time_column, filters)
    partial_first = partial_last = False
    if coverage is not None:
        covered_from, covered_until = coverage
        tolerance = _COMPLETENESS_TOLERANCE[grain]
        first_start = pd.Timestamp(frame["period"].iloc[0])
        last_start = pd.Timestamp(frame["period"].iloc[-1])
        partial_first = (covered_from - first_start) > tolerance
        partial_last = (period_end(last_start, grain) - covered_until) > tolerance
        frame.attrs["covered_from"] = covered_from.isoformat()
        frame.attrs["covered_until"] = covered_until.isoformat()

    frame.attrs["partial_first"] = bool(partial_first)
    frame.attrs["partial_last"] = bool(partial_last)
    frame.attrs["excluded_partial"] = []

    if complete_only and (partial_first or partial_last):
        keep = pd.Series(True, index=frame.index)
        excluded: list[str] = []
        if partial_last and len(frame) >= 3:
            keep.iloc[-1] = False
            excluded.append(str(frame["label"].iloc[-1]))
        if partial_first and int(keep.sum()) >= 3:
            keep.iloc[0] = False
            excluded.insert(0, str(frame["label"].iloc[0]))
        attrs = dict(frame.attrs)
        frame = frame[keep].reset_index(drop=True)
        frame.attrs.update(attrs)
        frame.attrs["excluded_partial"] = excluded

    return frame


def partial_note(frame: pd.DataFrame) -> str | None:
    """Human sentence explaining excluded partial periods, if any."""
    excluded = frame.attrs.get("excluded_partial") or []
    if not excluded:
        return None
    until = frame.attrs.get("covered_until")
    until_text = f" (data runs to {str(until)[:10]})" if until else ""
    return (
        f"{' and '.join(excluded)} {'is' if len(excluded) == 1 else 'are'} only "
        f"partly covered{until_text}, so {'it was' if len(excluded) == 1 else 'they were'} "
        "excluded to avoid comparing an incomplete period with complete ones."
    )


def _aggregate_window(
    df: pd.DataFrame,
    *,
    time_column: str,
    value_column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    agg: Aggregation,
    filters: list[FilterSpec] | None = None,
) -> float | None:
    frame, _ = apply_filters(df, filters or [])
    tcol = resolve_column(time_column, frame)
    vcol = resolve_column(value_column, frame)
    times = frame[tcol]
    if not pd.api.types.is_datetime64_any_dtype(times):
        times = pd.to_datetime(times, errors="coerce")
    values = frame.loc[(times >= start) & (times < end), vcol]
    if values.empty:
        return 0.0 if agg in _ADDITIVE_AGGS | {Aggregation.COUNT_DISTINCT} else None
    if agg == Aggregation.COUNT:
        return float(values.count())
    if agg == Aggregation.COUNT_DISTINCT:
        return float(values.nunique())
    numeric = pd.to_numeric(values, errors="coerce")
    if agg == Aggregation.MEAN:
        return safe_float(numeric.mean())
    if agg == Aggregation.MEDIAN:
        return safe_float(numeric.median())
    if agg == Aggregation.MIN:
        return safe_float(numeric.min())
    if agg == Aggregation.MAX:
        return safe_float(numeric.max())
    return safe_float(numeric.sum())


# ------------------------------------------------------------------ trend -----
def trend_analysis(
    df: pd.DataFrame,
    *,
    time_column: str,
    value_column: str,
    grain: TimeGrain,
    agg: Aggregation = Aggregation.SUM,
    filters: list[FilterSpec] | None = None,
) -> dict[str, Any]:
    """Direction, strength, growth and volatility of a measure over time."""
    frame = build_series(
        df,
        time_column=time_column,
        value_column=value_column,
        grain=grain,
        agg=agg,
        filters=filters,
        complete_only=True,
    )
    values = frame["value"].to_numpy(dtype="float64")
    n = values.size
    unit = guess_unit(value_column)
    caveats = [note] if (note := partial_note(frame)) else []

    out: dict[str, Any] = {
        "time_column": resolve_column(time_column, df),
        "value_column": resolve_column(value_column, df),
        "grain": grain.value,
        "agg": agg.value,
        "unit": unit,
        "points": n,
        "first_period": str(frame["label"].iloc[0]),
        "last_period": str(frame["label"].iloc[-1]),
        "first_value": safe_float(values[0]),
        "last_value": safe_float(values[-1]),
        "total": safe_float(values.sum()),
        "mean": safe_float(values.mean()),
        "peak": {
            "label": str(frame["label"].iloc[int(np.argmax(values))]),
            "value": safe_float(values.max()),
        },
        "trough": {
            "label": str(frame["label"].iloc[int(np.argmin(values))]),
            "value": safe_float(values.min()),
        },
        "excluded_partial": frame.attrs.get("excluded_partial", []),
        "caveats": caveats,
        "series": [
            {"period": str(p.date()), "label": str(lbl), "value": safe_float(v)}
            for p, lbl, v in zip(frame["period"], frame["label"], frame["value"])
        ],
    }

    if n < _MIN_TREND_POINTS:
        out["direction"] = "insufficient_data"
        out["summary"] = (
            f"Only {n} complete {grain.label} period(s) of data — at least "
            f"{_MIN_TREND_POINTS} are needed to describe a trend."
        )
        return out

    x = np.arange(n, dtype="float64")
    reg = scipy_stats.linregress(x, values)
    slope = float(reg.slope)
    r_squared = float(reg.rvalue) ** 2
    p_value = float(reg.pvalue)

    change_pct = pct_change(values[-1], values[0])
    mean_abs = float(np.mean(np.abs(values))) or 1.0
    relative_slope = slope / mean_abs

    if p_value >= 0.1 or abs(relative_slope) < 0.005:
        direction = "flat"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    volatility = safe_float(
        values.std(ddof=1) / abs(values.mean()) if values.mean() != 0 and n > 1 else None
    )

    deltas = np.diff(values)
    pop = [
        {
            "label": str(frame["label"].iloc[i + 1]),
            "value": safe_float(values[i + 1]),
            "change": safe_float(deltas[i]),
            "change_pct": pct_change(values[i + 1], values[i]),
        }
        for i in range(len(deltas))
    ]

    cagr = None
    if values[0] > 0 and values[-1] > 0 and n > 1:
        years = (n - 1) / grain.periods_per_year
        if years >= 1:  # annualising less than a year of data exaggerates wildly
            cagr = safe_float(((values[-1] / values[0]) ** (1 / years) - 1) * 100)

    out.update(
        {
            "direction": direction,
            "slope_per_period": safe_float(slope),
            "slope_pct_per_period": safe_float(relative_slope * 100),
            "r_squared": safe_float(round(r_squared, 4)),
            "p_value": safe_float(p_value),
            "significant": bool(p_value < 0.05),
            "change_pct": change_pct,
            "change_abs": safe_float(values[-1] - values[0]),
            "cagr_pct": cagr,
            "volatility": volatility,
            "period_over_period": pop[-24:],
            "best_period": max(pop, key=lambda d: d["change_pct"] or -1e18, default=None),
            "worst_period": min(pop, key=lambda d: d["change_pct"] or 1e18, default=None),
            "summary": _trend_sentence(
                value_column, direction, change_pct, r_squared, grain, unit,
                values, str(frame["label"].iloc[0]), str(frame["label"].iloc[-1]),
            ),
        }
    )
    return out


def _trend_sentence(
    column: str,
    direction: str,
    change_pct: float | None,
    r_squared: float,
    grain: TimeGrain,
    unit: str,
    values: np.ndarray,
    first_label: str,
    last_label: str,
) -> str:
    first = compact_number(values[0], unit=unit)
    last = compact_number(values[-1], unit=unit)
    if direction == "flat":
        return (
            f"{column} is broadly flat from {first_label} to {last_label} "
            f"({first} → {last}); the variation is not statistically "
            "distinguishable from noise."
        )
    word = "rose" if direction == "increasing" else "fell"
    pct = f"{abs(change_pct):.1f}%" if change_pct is not None else "an unmeasurable amount"
    fit = (
        "a consistent trend"
        if r_squared >= 0.6
        else ("a noisy trend" if r_squared >= 0.25 else "a weak, irregular trend")
    )
    return (
        f"{column} {word} {pct} from {first} in {first_label} to {last} in "
        f"{last_label} — {fit} across the {grain.label}ly series (R²={r_squared:.2f})."
    )


# -------------------------------------------------------- period matching -----
def _token_kind(token: str) -> str | None:
    t = token.strip().lower()
    if re.fullmatch(r"(?:fy\s*)?\d{4}", t):
        return "year"
    if re.fullmatch(r"q[1-4](?:\s*(?:of\s*)?\d{4})?|\d{4}\s*q[1-4]", t):
        return "quarter"
    words = re.findall(r"[a-z]+", t)
    if words and words[0].rstrip(".") in MONTH_TOKENS:
        return "month"
    return None


def grain_for_tokens(tokens: list[str]) -> TimeGrain | None:
    """If every period token is the same kind (years, quarters, months), the
    comparison must happen at that grain regardless of the dataset default."""
    kinds = {_token_kind(t) for t in tokens if t}
    if len(kinds) != 1:
        return None
    return {
        "year": TimeGrain.YEAR,
        "quarter": TimeGrain.QUARTER,
        "month": TimeGrain.MONTH,
    }.get(next(iter(kinds)) or "")


def match_period(token: str, frame: pd.DataFrame, grain: TimeGrain) -> int:
    """Index of the period a human token refers to.

    Accepts '2025', 'FY2025', 'Q3', 'Q3 2026', '2026 Q3', 'September',
    'Sept 2026', 'Sep', '2026-09', '2026-09-17' and exact labels. When the year
    is omitted, the most recent matching period is used ("September" means the
    latest September in the data).
    """
    raw = str(token).strip()
    t = re.sub(r"\s+", " ", raw.lower().replace(",", " ")).strip()
    labels = [str(x) for x in frame["label"]]
    periods = [pd.Timestamp(p) for p in frame["period"]]

    def latest(candidates: list[int]) -> int | None:
        return candidates[-1] if candidates else None

    for i, label in enumerate(labels):
        if label.lower() == t:
            return i

    found: int | None = None
    if m := re.fullmatch(r"(?:fy\s*)?(\d{4})", t):
        year = int(m.group(1))
        found = latest([i for i, p in enumerate(periods) if p.year == year]) if grain == TimeGrain.YEAR else None
        if found is None and grain != TimeGrain.YEAR:
            raise BadRequestError(
                f"'{raw}' is a year, but the comparison is at {grain.label} grain. "
                "Compare years at yearly grain."
            )
    elif m := (
        re.fullmatch(r"q([1-4])(?:\s*(?:of\s*)?(\d{4}))?", t)
        or re.fullmatch(r"(\d{4})\s*q([1-4])", t)
    ):
        groups = m.groups()
        if t.startswith("q"):
            quarter, year = int(groups[0]), (int(groups[1]) if groups[1] else None)
        else:
            year, quarter = int(groups[0]), int(groups[1])
        found = latest(
            [
                i
                for i, p in enumerate(periods)
                if (p.month - 1) // 3 + 1 == quarter and (year is None or p.year == year)
            ]
        )
    elif m := re.fullmatch(r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?", t):
        target = pd.Timestamp(
            year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3) or 1)
        )
        found = latest(
            [i for i, p in enumerate(periods) if p <= target < period_end(p, grain)]
        )
    else:
        words = re.findall(r"[a-z]+\.?|\d{4}", t)
        month = next(
            (MONTH_TOKENS[w.rstrip(".")] for w in words if w.rstrip(".") in MONTH_TOKENS),
            None,
        )
        year_match = re.search(r"\d{4}", t)
        year = int(year_match.group(0)) if year_match else None
        if month is not None:
            found = latest(
                [
                    i
                    for i, p in enumerate(periods)
                    if p.month == month and (year is None or p.year == year)
                ]
            )
        else:
            found = latest([i for i, label in enumerate(labels) if t in label.lower()])

    if found is None:
        raise BadRequestError(
            f"Period '{raw}' is not in the data. Available {grain.label}s: "
            + ", ".join(labels[-12:])
        )
    return found


# ------------------------------------------------------- period comparison ----
def compare_periods(
    df: pd.DataFrame,
    *,
    time_column: str,
    value_column: str,
    grain: TimeGrain,
    agg: Aggregation = Aggregation.SUM,
    period_a: str | None = None,
    period_b: str | None = None,
    filters: list[FilterSpec] | None = None,
) -> dict[str, Any]:
    """Compare two periods, like-for-like.

    Defaults to the latest *complete* period versus the one before it. When an
    explicitly requested period is only partly covered by the data, both
    periods are cut to the same span (e.g. Jan 1 – Sep 30 of each year).
    """
    tokens = [p for p in (period_a, period_b) if p]
    grain = grain_for_tokens(tokens) or grain

    frame = build_series(
        df,
        time_column=time_column,
        value_column=value_column,
        grain=grain,
        agg=agg,
        filters=filters,
    )
    if len(frame) < 2:
        raise AnalysisError(
            f"Only one {grain.label} period is present, so there is nothing to "
            "compare it against."
        )

    notes: list[str] = []
    last_complete = len(frame) - 1
    if frame.attrs.get("partial_last") and len(frame) >= 3:
        last_complete = len(frame) - 2

    if period_b:
        idx_b = match_period(period_b, frame, grain)
    else:
        idx_b = last_complete
        if last_complete != len(frame) - 1:
            notes.append(
                f"{frame['label'].iloc[-1]} is still in progress, so the latest "
                f"complete {grain.label} ({frame['label'].iloc[idx_b]}) was used."
            )
    idx_a = match_period(period_a, frame, grain) if period_a else idx_b - 1
    if idx_a < 0:
        raise AnalysisError(
            f"{frame['label'].iloc[idx_b]} is the first {grain.label} in the data, so "
            "there is no earlier period to compare it with."
        )
    if idx_a == idx_b:
        raise BadRequestError("The two periods compared must be different.")
    if idx_a > idx_b:  # baseline is always the earlier period
        idx_a, idx_b = idx_b, idx_a

    tcol = resolve_column(time_column, df)
    vcol = resolve_column(value_column, df)
    unit = guess_unit(vcol)

    a_start = pd.Timestamp(frame["period"].iloc[idx_a])
    b_start = pd.Timestamp(frame["period"].iloc[idx_b])
    a_end, b_end = period_end(a_start, grain), period_end(b_start, grain)
    a_full = safe_float(frame["value"].iloc[idx_a])
    b_full = safe_float(frame["value"].iloc[idx_b])

    # ---- like-for-like windows when either period is only partly covered ----
    coverage = data_coverage(df, tcol, filters)
    like_for_like = False
    a_window, b_window = (a_start, a_end), (b_start, b_end)
    a_value, b_value = a_full, b_full
    if coverage is not None:
        covered_from, covered_until = coverage
        tolerance = _COMPLETENESS_TOLERANCE[grain]
        a_off = (max(covered_from, a_start) - a_start, min(covered_until, a_end) - a_start)
        b_off = (max(covered_from, b_start) - b_start, min(covered_until, b_end) - b_start)
        a_partial = a_off[0] > tolerance or (a_end - a_start) - a_off[1] > tolerance
        b_partial = b_off[0] > tolerance or (b_end - b_start) - b_off[1] > tolerance
        if a_partial or b_partial:
            off_start = max(a_off[0], b_off[0])
            off_end = min(a_off[1], b_off[1])
            if off_end <= off_start:
                raise AnalysisError(
                    f"{frame['label'].iloc[idx_a]} and {frame['label'].iloc[idx_b]} "
                    "have no overlapping span of data to compare like-for-like."
                )
            like_for_like = True
            a_window = (a_start + off_start, a_start + off_end)
            b_window = (b_start + off_start, b_start + off_end)
            a_value = _aggregate_window(
                df, time_column=tcol, value_column=vcol, start=a_window[0],
                end=a_window[1], agg=agg, filters=filters,
            )
            b_value = _aggregate_window(
                df, time_column=tcol, value_column=vcol, start=b_window[0],
                end=b_window[1], agg=agg, filters=filters,
            )
            span = (
                f"{_short_date(a_window[0])}–{_short_date(a_window[1] - pd.Timedelta(seconds=1))}"
            )
            notes.append(
                f"{frame['label'].iloc[idx_b if b_partial else idx_a]} is only partly "
                f"covered by the data, so both periods are compared over the same "
                f"span ({span} of each). Full {frame['label'].iloc[idx_a]}: "
                f"{compact_number(a_full, unit=unit)}."
            )

    change = (b_value or 0.0) - (a_value or 0.0)
    change_pct = pct_change(b_value, a_value)
    a_label = str(frame["label"].iloc[idx_a])
    b_label = str(frame["label"].iloc[idx_b])
    qualifier = " (same span)" if like_for_like else ""

    return {
        "value_column": vcol,
        "time_column": tcol,
        "grain": grain.value,
        "agg": agg.value,
        "unit": unit,
        "like_for_like": like_for_like,
        "baseline": {
            "label": a_label,
            "period": str(a_start.date()),
            "window": [a_window[0].isoformat(), a_window[1].isoformat()],
            "value": a_value,
            "full_period_value": a_full,
            "formatted": compact_number(a_value, unit=unit),
        },
        "comparison": {
            "label": b_label,
            "period": str(b_start.date()),
            "window": [b_window[0].isoformat(), b_window[1].isoformat()],
            "value": b_value,
            "full_period_value": b_full,
            "formatted": compact_number(b_value, unit=unit),
        },
        "change_abs": safe_float(change),
        "change_pct": change_pct,
        "direction": "up" if change > 0 else ("down" if change < 0 else "flat"),
        "notes": notes,
        "summary": (
            f"{vcol} moved from {compact_number(a_value, unit=unit)} in {a_label} to "
            f"{compact_number(b_value, unit=unit)} in {b_label}{qualifier} — "
            + (
                f"{'up' if change > 0 else 'down'} {abs(change_pct):.1f}%."
                if change_pct is not None
                else "a change that cannot be expressed as a percentage (zero baseline)."
            )
        ),
    }


def _short_date(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%d %b").lstrip("0")


# ----------------------------------------------------- driver decomposition ---
def _decompose(
    df: pd.DataFrame,
    *,
    tcol: str,
    vcol: str,
    dcol: str,
    a_window: tuple[pd.Timestamp, pd.Timestamp],
    b_window: tuple[pd.Timestamp, pd.Timestamp],
    agg: Aggregation,
    total_change: float,
) -> list[dict[str, Any]]:
    times = df[tcol]
    if not pd.api.types.is_datetime64_any_dtype(times):
        times = pd.to_datetime(times, errors="coerce")
    values = pd.to_numeric(df[vcol], errors="coerce")
    levels = df[dcol].astype("object").where(df[dcol].notna(), "(missing)")

    def grouped(window: tuple[pd.Timestamp, pd.Timestamp]) -> pd.Series:
        mask = (times >= window[0]) & (times < window[1])
        if not mask.any():
            return pd.Series(dtype="float64")
        g = values[mask].groupby(levels[mask], observed=True)
        if agg == Aggregation.MEAN:
            return g.mean()
        if agg == Aggregation.MEDIAN:
            return g.median()
        if agg == Aggregation.COUNT:
            return g.count()
        if agg == Aggregation.COUNT_DISTINCT:
            return df.loc[mask, vcol].groupby(levels[mask], observed=True).nunique()
        return g.sum()

    before_s, after_s = grouped(a_window), grouped(b_window)
    baseline_total = float(before_s.sum()) if len(before_s) else 0.0
    overall_rate = (total_change / baseline_total) if baseline_total else None
    out: list[dict[str, Any]] = []
    for level in set(before_s.index) | set(after_s.index):
        before = safe_float(before_s.get(level, 0.0)) or 0.0
        after = safe_float(after_s.get(level, 0.0)) or 0.0
        if before == 0 and after == 0:
            continue
        delta = after - before
        # What this level "should" have moved if it had followed the overall
        # rate; the excess is what makes it a driver rather than just big.
        expected = before * overall_rate if overall_rate is not None else None
        out.append(
            {
                "value": str(level),
                "before": before,
                "after": after,
                "change_abs": safe_float(delta),
                "change_pct": pct_change(after, before),
                "contribution_pct": safe_float(delta / total_change * 100)
                if total_change
                else None,
                "expected_change": safe_float(expected),
                "excess_change": safe_float(delta - expected) if expected is not None else None,
                "baseline_share_pct": safe_float(before / baseline_total * 100)
                if baseline_total
                else None,
                "appeared": before == 0 and after != 0,
                "disappeared": after == 0 and before != 0,
            }
        )
    out.sort(key=lambda c: -abs(c["change_abs"] or 0.0))
    return out


def _driver_score(drivers: list[dict[str, Any]], total_change: float) -> float:
    """How strongly one value of this dimension explains the change.

    Uses the largest *excess* change — movement beyond what the value's
    baseline size predicts — relative to the headline change. A dimension
    dominated by one huge level (e.g. 86% of orders are "Delivered") scores
    low, because that level merely moved in proportion to its size.
    """
    if not drivers or not total_change:
        return 0.0
    excess = [abs(d["excess_change"] or 0.0) for d in drivers if d.get("excess_change") is not None]
    if not excess:
        return 0.0
    return max(excess) / abs(total_change)


def explain_change(
    df: pd.DataFrame,
    *,
    time_column: str,
    value_column: str,
    grain: TimeGrain,
    dimension: str | None = None,
    candidate_dimensions: list[str] | None = None,
    agg: Aggregation = Aggregation.SUM,
    period_a: str | None = None,
    period_b: str | None = None,
    top_n: int = 8,
) -> dict[str, Any]:
    """Attribute a period-over-period change to the values of a dimension.

    With SUM or COUNT the contributions sum to the headline change exactly.
    When `dimension` is omitted, each of `candidate_dimensions` is tried and
    the one where the change is most concentrated is reported, together with
    the scores of the others.
    """
    tcol = resolve_column(time_column, df)
    vcol = resolve_column(value_column, df)
    if agg not in {Aggregation.COUNT, Aggregation.COUNT_DISTINCT} and not pd.api.types.is_numeric_dtype(df[vcol]):
        raise BadRequestError(f"'{vcol}' must be numeric to decompose a change.")

    overall = compare_periods(
        df,
        time_column=tcol,
        value_column=vcol,
        grain=grain,
        agg=agg,
        period_a=period_a,
        period_b=period_b,
    )
    a_window = tuple(pd.Timestamp(x) for x in overall["baseline"]["window"])
    b_window = tuple(pd.Timestamp(x) for x in overall["comparison"]["window"])
    total_change = safe_float(overall["change_abs"]) or 0.0

    if dimension:
        candidates = [resolve_column(dimension, df)]
    else:
        candidates = [resolve_column(c, df) for c in (candidate_dimensions or [])][:6]
        if not candidates:
            raise BadRequestError(
                "No dimension is available to attribute the change across."
            )

    scored: list[tuple[float, str, list[dict[str, Any]]]] = []
    for dcol in candidates:
        if df[dcol].nunique(dropna=True) > 200:
            continue
        drivers = _decompose(
            df, tcol=tcol, vcol=vcol, dcol=dcol, a_window=a_window,  # type: ignore[arg-type]
            b_window=b_window, agg=agg, total_change=total_change,  # type: ignore[arg-type]
        )
        if drivers:
            scored.append((_driver_score(drivers, total_change), dcol, drivers))

    if not scored:
        raise AnalysisError(
            "No rows fall in the compared periods for any candidate dimension."
        )
    scored.sort(key=lambda s: -s[0])
    score, dcol, contributions = scored[0]
    # Within the chosen dimension, lead with the value that moved most beyond
    # its expected share; the evidence table stays sorted by absolute change.
    standout = max(
        contributions,
        key=lambda c: abs(c["excess_change"] or 0.0)
        if c.get("excess_change") is not None
        else abs(c["change_abs"] or 0.0),
    )
    unit = overall["unit"]

    positives = [c for c in contributions if (c["change_abs"] or 0) > 0]
    negatives = [c for c in contributions if (c["change_abs"] or 0) < 0]
    primary = standout
    falling = total_change < 0
    contribution_sum = sum(c["change_abs"] or 0.0 for c in contributions)
    additive = agg in _ADDITIVE_AGGS

    narrative = overall["summary"]
    if total_change:
        moved = "fell" if (primary["change_abs"] or 0) < 0 else "rose"
        narrative += (
            f" The standout driver is {dcol} = '{primary['value']}', which {moved} "
            f"{abs(primary['change_pct'] or 0):.1f}% "
            f"({compact_number(abs(primary['change_abs'] or 0), unit=unit)})"
        )
        overall_pct = overall.get("change_pct")
        if overall_pct is not None:
            narrative += f" against {abs(overall_pct):.1f}% overall."
        else:
            narrative += "."

    return {
        "overall": overall,
        "dimension": dcol,
        "dimension_selected_automatically": dimension is None,
        "dimension_scores": [
            {"dimension": d, "score": round(s, 4)} for s, d, _ in scored
        ],
        "drivers": contributions[:top_n],
        "standout": standout,
        "overall_change_pct": overall.get("change_pct"),
        "biggest_decline": negatives[0] if negatives else None,
        "biggest_increase": positives[0] if positives else None,
        "driver_count": len(contributions),
        "driver_score": round(score, 4),
        "decomposition_check": {
            "additive": additive,
            "sum_of_contributions": safe_float(contribution_sum),
            "reported_total_change": safe_float(total_change),
            "exact": bool(
                additive
                and abs(contribution_sum - total_change)
                < max(abs(total_change) * 1e-6, 1e-6)
            ),
        },
        "summary": narrative,
    }


def format_share(share: float | None) -> str:
    if share is None:
        return "—"
    # A single driver can exceed 100% of a net change when others offset it.
    return f"{abs(share):.0f}%"


# ---------------------------------------------------------- concentration -----
def top_contributors(
    df: pd.DataFrame,
    *,
    dimension: str,
    measure: str,
    agg: Aggregation = Aggregation.SUM,
    top_n: int = 10,
    ascending: bool = False,
) -> dict[str, Any]:
    """Ranked contribution of a dimension's values to a measure, with the
    Pareto point (how few values make up 80% of the total)."""
    dcol = resolve_column(dimension, df)
    mcol = resolve_column(measure, df)

    work = pd.DataFrame(
        {"d": df[dcol].astype("object"), "m": pd.to_numeric(df[mcol], errors="coerce")}
    ).dropna()
    if work.empty:
        raise AnalysisError(f"No rows have both '{dcol}' and a numeric '{mcol}'.")

    grouped = work.groupby("d", observed=True)["m"]
    if agg == Aggregation.MEAN:
        totals = grouped.mean()
    elif agg == Aggregation.COUNT:
        totals = grouped.count()
    elif agg == Aggregation.COUNT_DISTINCT:
        totals = grouped.nunique()
    elif agg == Aggregation.MEDIAN:
        totals = grouped.median()
    else:
        totals = grouped.sum()

    totals = totals.sort_values(ascending=False)
    # Shares are only meaningful for additive aggregations: the "share" of an
    # average is not a thing.
    additive = agg in _ADDITIVE_AGGS
    grand_total = float(totals.sum()) if additive else None
    unit = guess_unit(mcol)

    ranked: list[dict[str, Any]] = []
    cumulative = 0.0
    pareto_count: int | None = None
    for rank, (level, value) in enumerate(totals.items(), start=1):
        value_f = safe_float(value) or 0.0
        cumulative += value_f
        share = safe_div(value_f, grand_total) if additive else None
        cumulative_share = safe_div(cumulative, grand_total) if additive else None
        if pareto_count is None and cumulative_share is not None and cumulative_share >= 0.8:
            pareto_count = rank
        ranked.append(
            {
                "rank": rank,
                "value": str(level),
                "metric": value_f,
                "formatted": compact_number(value_f, unit=unit),
                "share_pct": safe_float(share * 100) if share is not None else None,
                "cumulative_share_pct": safe_float(cumulative_share * 100)
                if cumulative_share is not None
                else None,
                "rows": int(grouped.size().get(level, 0)),
            }
        )

    level_count = len(ranked)
    if ascending:
        ranked = list(reversed(ranked))
        for position, row in enumerate(ranked, start=1):
            row["rank"] = position
    rows = ranked[:top_n]

    summary = ""
    if rows:
        leader = rows[0]
        position = "smallest" if ascending else "largest"
        agg_word = {"mean": "average ", "median": "median "}.get(agg.value, "")
        summary = (
            f"{dcol} = '{leader['value']}' has the {position} {agg_word}{mcol} at "
            f"{leader['formatted']}"
            + (
                f" ({leader['share_pct']:.1f}% of the total)."
                if leader["share_pct"] is not None
                else "."
            )
        )
        if pareto_count and level_count > 1 and not ascending:
            summary += (
                f" {pareto_count} of {level_count} {dcol} values account for 80% "
                "of the total."
            )

    return {
        "dimension": dcol,
        "measure": mcol,
        "agg": agg.value,
        "unit": unit,
        "additive": additive,
        "ascending": ascending,
        "total": safe_float(grand_total) if grand_total is not None else None,
        "level_count": level_count,
        "rows": rows,
        "pareto_80_count": pareto_count,
        "concentration": safe_float(
            sum(r["share_pct"] or 0 for r in ranked[: min(3, len(ranked))])
        )
        if additive and not ascending
        else None,
        "summary": summary,
    }
