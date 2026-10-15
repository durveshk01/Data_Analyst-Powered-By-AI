"""Anomaly detection with automatic method selection.

Five methods, each appropriate for a different shape of data:

* ``iqr``        — Tukey fences. Robust, assumption-free, good for skewed data.
* ``zscore``     — modified z-score on the median absolute deviation. Robust
                   variant chosen deliberately: a classic mean/σ z-score is
                   itself distorted by the outliers it is trying to find.
* ``rolling``    — rolling median ± k·MAD on a time series. Catches a point
                   that is wrong *for its moment* even if it is normal overall.
* ``seasonal``   — residuals after removing trend and a repeating seasonal
                   pattern. Needed so "sales dropped on Sunday" is not reported
                   as an anomaly every week.
* ``isolation``  — Isolation Forest over all numeric columns. Finds rows that
                   are unusual in combination while every single value is
                   normal.

:func:`choose_method` picks one from the data's own properties, and the chosen
method is always reported alongside the findings.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.engine.query import resolve_column
from app.engine.types import (
    AnomalyModel,
    Aggregation,
    DatasetProfile,
    SemanticModel,
    Severity,
    TimeGrain,
)
from app.engine.utils import compact_number, guess_unit, safe_float
from app.errors import AnalysisError, BadRequestError
from app.obs.logging import get_logger

log = get_logger(__name__)

try:  # statsmodels is optional — the engine degrades, it does not fail.
    from statsmodels.tsa.seasonal import seasonal_decompose

    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover - exercised only when the wheel is absent
    seasonal_decompose = None  # type: ignore[assignment]
    _HAS_STATSMODELS = False

Method = Literal["auto", "iqr", "zscore", "rolling", "seasonal", "isolation"]

_MAD_TO_SIGMA = 1.4826  # makes the MAD a consistent estimator of σ for normals
_MIN_POINTS_ROLLING = 10
_MIN_SEASONS = 2
_MAX_ANOMALIES = 50
_ISOLATION_SAMPLE = 50_000


def _anomaly_id() -> str:
    return uuid.uuid4().hex[:12]


def _severity_from_score(score: float) -> Severity:
    if score >= 6:
        return Severity.CRITICAL
    if score >= 3.5:
        return Severity.WARNING
    return Severity.INFO


def _seasonal_period(grain: TimeGrain) -> int:
    return {
        TimeGrain.HOUR: 24,
        TimeGrain.DAY: 7,
        TimeGrain.WEEK: 52,
        TimeGrain.MONTH: 12,
        TimeGrain.QUARTER: 4,
        TimeGrain.YEAR: 1,
    }[grain]


# ------------------------------------------------------- method selection -----
def anomaly_grain(span_days: int | None, native: TimeGrain | None) -> TimeGrain:
    """The grain to hunt for anomalies at — finer than the reporting grain.

    Charts read best monthly, but a one-day outage vanishes inside a monthly
    total. Daily is the right resolution whenever the span allows a few hundred
    to a couple of thousand points; never finer than the data itself.
    """
    if span_days is None:
        return native or TimeGrain.DAY
    if span_days <= 3:
        chosen = TimeGrain.HOUR
    elif span_days <= 1500:
        chosen = TimeGrain.DAY
    elif span_days <= 365 * 15:
        chosen = TimeGrain.WEEK
    else:
        chosen = TimeGrain.MONTH
    order = list(TimeGrain)
    if native is not None and order.index(native) > order.index(chosen):
        return native
    return chosen


def choose_method(
    *,
    is_time_series: bool,
    points: int,
    grain: TimeGrain | None,
    skewness: float | None,
    numeric_column_count: int,
) -> tuple[Method, str]:
    """Pick a detection method and explain the choice."""
    if is_time_series and grain is not None:
        season = _seasonal_period(grain)
        if _HAS_STATSMODELS and season > 1 and points >= season * _MIN_SEASONS + 1:
            return (
                "seasonal",
                f"{points} {grain.label} periods cover at least {_MIN_SEASONS} full "
                f"seasonal cycles ({season} periods each), so trend and seasonality "
                "are removed before looking for outliers.",
            )
        if points >= _MIN_POINTS_ROLLING:
            return (
                "rolling",
                f"{points} {grain.label} periods is enough for a local baseline but "
                "too short to estimate seasonality, so a rolling median is used.",
            )
        return (
            "iqr",
            f"Only {points} periods available — a distribution-free IQR test is the "
            "only defensible choice at this length.",
        )

    if numeric_column_count >= 3:
        return (
            "isolation",
            f"{numeric_column_count} numeric columns are available, so rows are "
            "scored jointly with an Isolation Forest to catch combinations that are "
            "unusual even when each value looks normal.",
        )
    if skewness is not None and abs(skewness) > 1.0:
        return (
            "iqr",
            f"The distribution is strongly skewed (skewness {skewness:.2f}), where "
            "quartile fences are more reliable than a z-score.",
        )
    return (
        "zscore",
        "The distribution is roughly symmetric, so a modified z-score on the median "
        "absolute deviation is used.",
    )


# ---------------------------------------------------------------- methods ------
def detect_iqr(
    values: pd.Series, *, labels: list[str] | None = None, k: float = 1.5
) -> list[AnomalyModel]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 8:
        return []
    q1, q3 = np.percentile(numeric, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return []
    lower, upper = q1 - k * iqr, q3 + k * iqr
    median = float(np.median(numeric))

    out: list[AnomalyModel] = []
    for position, (idx, value) in enumerate(numeric.items()):
        v = float(value)
        if lower <= v <= upper:
            continue
        distance = (lower - v) if v < lower else (v - upper)
        score = safe_float(distance / iqr) or 0.0
        out.append(
            AnomalyModel(
                id=_anomaly_id(),
                method="iqr",
                column=str(values.name or "value"),
                label=_label_for(labels, idx, position),
                observed=safe_float(v),
                expected=safe_float(median),
                deviation_pct=safe_float((v - median) / abs(median) * 100)
                if median
                else None,
                score=round(score, 3),
                severity=_severity_from_score(score * 2),
                row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                explanation=(
                    f"{compact_number(v)} lies outside the Tukey fence "
                    f"[{compact_number(lower)}, {compact_number(upper)}] "
                    f"(Q1={compact_number(q1)}, Q3={compact_number(q3)}, "
                    f"IQR={compact_number(iqr)})."
                ),
                context={
                    "q1": safe_float(q1),
                    "q3": safe_float(q3),
                    "iqr": safe_float(iqr),
                    "lower_fence": safe_float(lower),
                    "upper_fence": safe_float(upper),
                },
            )
        )
    return out


def detect_zscore(
    values: pd.Series, *, labels: list[str] | None = None, threshold: float = 3.5
) -> list[AnomalyModel]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 8:
        return []
    median = float(np.median(numeric))
    mad = float(np.median(np.abs(numeric - median)))
    if mad == 0:
        # Degenerate MAD (more than half the values identical): fall back to σ.
        sigma = float(numeric.std(ddof=1))
        if sigma == 0:
            return []
        scores = (numeric - numeric.mean()).abs() / sigma
        centre, scale, basis = float(numeric.mean()), sigma, "mean/σ"
    else:
        scores = (numeric - median).abs() / (mad * _MAD_TO_SIGMA)
        centre, scale, basis = median, mad * _MAD_TO_SIGMA, "median/MAD"

    out: list[AnomalyModel] = []
    for position, (idx, score) in enumerate(scores.items()):
        s = float(score)
        if s < threshold:
            continue
        v = float(numeric.loc[idx])
        out.append(
            AnomalyModel(
                id=_anomaly_id(),
                method="zscore",
                column=str(values.name or "value"),
                label=_label_for(labels, idx, position),
                observed=safe_float(v),
                expected=safe_float(centre),
                deviation_pct=safe_float((v - centre) / abs(centre) * 100)
                if centre
                else None,
                score=round(s, 3),
                severity=_severity_from_score(s),
                row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                explanation=(
                    f"{compact_number(v)} is {s:.1f} robust standard deviations from "
                    f"the centre {compact_number(centre)} ({basis}, scale "
                    f"{compact_number(scale)})."
                ),
                context={"centre": safe_float(centre), "scale": safe_float(scale), "basis": basis},
            )
        )
    return out


def detect_rolling(
    series: pd.DataFrame,
    *,
    window: int | None = None,
    threshold: float = 3.0,
    column_name: str = "value",
) -> list[AnomalyModel]:
    """Rolling median ± k·MAD over a `period`/`label`/`value` frame."""
    values = pd.to_numeric(series["value"], errors="coerce")
    n = len(values)
    if n < _MIN_POINTS_ROLLING:
        return []
    window = window or max(5, min(13, n // 3))

    centre = values.rolling(window, center=True, min_periods=max(3, window // 2)).median()
    deviation = (values - centre).abs()
    scale = deviation.rolling(
        window, center=True, min_periods=max(3, window // 2)
    ).median() * _MAD_TO_SIGMA
    # A flat window gives scale 0; fall back to the global dispersion.
    global_scale = float(deviation.median() * _MAD_TO_SIGMA) or float(values.std(ddof=1) or 0)
    scale = scale.replace(0, np.nan).fillna(global_scale if global_scale else np.nan)

    out: list[AnomalyModel] = []
    unit = guess_unit(column_name)
    for i in range(n):
        expected = safe_float(centre.iloc[i])
        observed = safe_float(values.iloc[i])
        s = safe_float(scale.iloc[i])
        if expected is None or observed is None or not s:
            continue
        score = abs(observed - expected) / s
        if score < threshold:
            continue
        deviation_pct = (
            safe_float((observed - expected) / abs(expected) * 100) if expected else None
        )
        label = str(series["label"].iloc[i]) if "label" in series else str(i)
        direction = "below" if observed < expected else "above"
        out.append(
            AnomalyModel(
                id=_anomaly_id(),
                method="rolling",
                column=column_name,
                label=label,
                observed=observed,
                expected=expected,
                deviation_pct=deviation_pct,
                score=round(score, 3),
                severity=_severity_from_score(score),
                row_index=i,
                explanation=(
                    f"{label}: {column_name} was {compact_number(observed, unit=unit)}, "
                    f"{abs(deviation_pct):.0f}% {direction} the local expectation of "
                    f"{compact_number(expected, unit=unit)} "
                    f"(rolling median over {window} periods)."
                    if deviation_pct is not None
                    else (
                        f"{label}: {compact_number(observed, unit=unit)} vs expected "
                        f"{compact_number(expected, unit=unit)}."
                    )
                ),
                context={
                    "window": window,
                    "period": str(pd.Timestamp(series["period"].iloc[i]).date())
                    if "period" in series
                    else None,
                    "scale": s,
                },
            )
        )
    return out


def detect_seasonal(
    series: pd.DataFrame,
    *,
    grain: TimeGrain,
    threshold: float = 3.0,
    column_name: str = "value",
) -> list[AnomalyModel]:
    """Decompose into trend + seasonal + residual, then test the residual."""
    if not _HAS_STATSMODELS or seasonal_decompose is None:
        return detect_rolling(series, threshold=threshold, column_name=column_name)

    values = pd.to_numeric(series["value"], errors="coerce").astype("float64")
    period = _seasonal_period(grain)
    if period < 2 or len(values) < period * _MIN_SEASONS + 1:
        return detect_rolling(series, threshold=threshold, column_name=column_name)

    indexed = pd.Series(
        values.to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(series["period"]))
    ).interpolate(limit_direction="both")

    try:
        decomposition = seasonal_decompose(
            indexed, model="additive", period=period, extrapolate_trend="freq"
        )
    except Exception as exc:  # pragma: no cover - statsmodels edge cases
        log.info("seasonal_decompose_failed", extra={"error": str(exc)[:160]})
        return detect_rolling(series, threshold=threshold, column_name=column_name)

    residual = pd.Series(decomposition.resid).dropna()
    if residual.empty:
        return detect_rolling(series, threshold=threshold, column_name=column_name)

    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    scale = mad * _MAD_TO_SIGMA if mad > 0 else float(residual.std(ddof=1) or 0)
    if not scale:
        return []

    expected_series = (
        pd.Series(decomposition.trend) + pd.Series(decomposition.seasonal)
    )
    unit = guess_unit(column_name)

    out: list[AnomalyModel] = []
    for i, timestamp in enumerate(indexed.index):
        if timestamp not in residual.index:
            continue
        r = float(residual.loc[timestamp])
        score = abs(r - median) / scale
        if score < threshold:
            continue
        observed = safe_float(indexed.loc[timestamp])
        expected = safe_float(expected_series.get(timestamp))
        if observed is None or expected is None:
            continue
        deviation_pct = (
            safe_float((observed - expected) / abs(expected) * 100) if expected else None
        )
        label = str(series["label"].iloc[i]) if "label" in series else str(timestamp.date())
        direction = "below" if observed < expected else "above"
        out.append(
            AnomalyModel(
                id=_anomaly_id(),
                method="seasonal",
                column=column_name,
                label=label,
                observed=observed,
                expected=expected,
                deviation_pct=deviation_pct,
                score=round(score, 3),
                severity=_severity_from_score(score),
                row_index=i,
                explanation=(
                    f"{label}: {column_name} was {compact_number(observed, unit=unit)}, "
                    f"{abs(deviation_pct):.0f}% {direction} the "
                    f"{compact_number(expected, unit=unit)} expected once the trend "
                    f"and the repeating {grain.label}-of-cycle pattern are removed."
                    if deviation_pct is not None
                    else f"{label}: residual {score:.1f}σ from the seasonal baseline."
                ),
                context={
                    "period": str(timestamp.date()),
                    "seasonal_period": period,
                    "residual": safe_float(r),
                    "trend": safe_float(pd.Series(decomposition.trend).get(timestamp)),
                    "seasonal": safe_float(
                        pd.Series(decomposition.seasonal).get(timestamp)
                    ),
                },
            )
        )
    return out


def detect_isolation(
    df: pd.DataFrame,
    columns: list[str],
    *,
    contamination: float = 0.01,
    max_results: int = 25,
) -> list[AnomalyModel]:
    """Multivariate row-level anomalies via Isolation Forest."""
    usable = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(usable) < 2:
        return []

    frame = df[usable].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 50:
        return []

    sample = (
        frame.sample(n=_ISOLATION_SAMPLE, random_state=42)
        if len(frame) > _ISOLATION_SAMPLE
        else frame
    )
    contamination = float(min(max(contamination, 0.0005), 0.1))

    model = IsolationForest(
        n_estimators=120,
        contamination=contamination,
        random_state=42,
        n_jobs=1,          # single core: the free tier has little CPU to spare
        max_samples=min(256, len(sample)),
    )
    model.fit(sample)
    scores = model.score_samples(frame)   # higher = more normal
    predictions = model.predict(frame)

    flagged = np.where(predictions == -1)[0]
    if flagged.size == 0:
        return []
    # Most anomalous first.
    flagged = flagged[np.argsort(scores[flagged])][:max_results]

    medians = frame.median()
    out: list[AnomalyModel] = []
    for position in flagged:
        row_label = frame.index[position]
        row = frame.iloc[position]
        # Name the columns that deviate most, so the finding is explainable.
        deviations: list[tuple[str, float]] = []
        for col in usable:
            centre = float(medians[col])
            spread = float(frame[col].std(ddof=1) or 0)
            if spread == 0:
                continue
            deviations.append((col, abs(float(row[col]) - centre) / spread))
        deviations.sort(key=lambda d: -d[1])
        drivers = deviations[:3]

        detail = ", ".join(
            f"{col}={compact_number(row[col], unit=guess_unit(col))} "
            f"({z:.1f}σ from typical)"
            for col, z in drivers
        )
        normalised_score = float(-scores[position]) * 10
        out.append(
            AnomalyModel(
                id=_anomaly_id(),
                method="isolation_forest",
                column=", ".join(c for c, _ in drivers) or usable[0],
                label=f"Row {int(row_label) + 1}",
                observed=safe_float(row[drivers[0][0]]) if drivers else None,
                expected=safe_float(medians[drivers[0][0]]) if drivers else None,
                score=round(normalised_score, 3),
                severity=_severity_from_score(normalised_score),
                row_index=int(row_label) if isinstance(row_label, (int, np.integer)) else None,
                explanation=(
                    f"Row {int(row_label) + 1} is unusual across {len(usable)} numeric "
                    f"columns considered together: {detail}."
                ),
                context={
                    "columns_considered": usable,
                    "isolation_score": safe_float(float(scores[position])),
                    "drivers": [
                        {"column": c, "sigma": round(z, 3)} for c, z in drivers
                    ],
                },
            )
        )
    return out


def _label_for(labels: list[str] | None, idx: Any, position: int) -> str:
    if labels and position < len(labels):
        return str(labels[position])
    if isinstance(idx, (int, np.integer)):
        return f"Row {int(idx) + 1}"
    return str(idx)


# ------------------------------------------------------------- orchestrator ---
def detect_anomalies(
    df: pd.DataFrame,
    *,
    profile: DatasetProfile,
    semantics: SemanticModel,
    column: str | None = None,
    method: Method = "auto",
    sensitivity: float = 1.0,
    max_results: int = _MAX_ANOMALIES,
) -> dict[str, Any]:
    """Run anomaly detection, choosing a method when `method='auto'`."""
    from app.engine.trends import build_series  # local import avoids a cycle

    target = column or semantics.primary_measure
    if not target:
        raise AnalysisError(
            "No numeric measure was detected in this dataset, so there is nothing "
            "to test for anomalies."
        )
    target = resolve_column(target, df)
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise BadRequestError(
            f"'{target}' is {df[target].dtype}; anomaly detection needs a numeric column."
        )

    col_profile = profile.column(target)
    time_column = semantics.primary_time_column
    grain: TimeGrain | None = None
    if time_column:
        time_profile = profile.column(time_column)
        grain = anomaly_grain(
            time_profile.stats.span_days if time_profile else None,
            time_profile.stats.inferred_grain if time_profile else None,
        )

    series: pd.DataFrame | None = None
    if time_column and grain:
        try:
            series = build_series(
                df,
                time_column=time_column,
                value_column=target,
                grain=grain,
                agg=Aggregation.SUM,
                # An in-progress final period always looks like a collapse.
                complete_only=True,
            )
        except (AnalysisError, BadRequestError):
            series = None

    numeric_columns = [
        c.name for c in profile.columns if c.semantic_type.is_numeric and c.stats.count > 0
    ]

    if method == "auto":
        chosen, reason = choose_method(
            is_time_series=series is not None and len(series) >= 6,
            points=len(series) if series is not None else 0,
            grain=grain if series is not None else None,
            skewness=col_profile.stats.skewness if col_profile else None,
            numeric_column_count=len(numeric_columns),
        )
    else:
        chosen, reason = method, f"Method '{method}' was requested explicitly."

    # Sensitivity scales the detection threshold: >1 is stricter (fewer, more
    # extreme findings); <1 surfaces more.
    sensitivity = float(min(max(sensitivity, 0.25), 4.0))
    anomalies: list[AnomalyModel] = []

    if chosen == "seasonal" and series is not None and grain:
        anomalies = detect_seasonal(
            series, grain=grain, threshold=3.0 * sensitivity, column_name=target
        )
    elif chosen == "rolling" and series is not None:
        anomalies = detect_rolling(
            series, threshold=3.0 * sensitivity, column_name=target
        )
    elif chosen == "isolation":
        anomalies = detect_isolation(
            df,
            numeric_columns,
            contamination=0.01 / sensitivity,
            max_results=min(max_results, 25),
        )
    elif chosen == "zscore":
        anomalies = detect_zscore(df[target], threshold=3.5 * sensitivity)
    else:
        chosen = "iqr"
        anomalies = detect_iqr(df[target], k=1.5 * sensitivity)

    # If the chosen method found nothing, say so rather than silently switching
    # to a looser method and presenting the result as the same analysis.
    anomalies.sort(key=lambda a: -(a.score or 0))
    truncated = len(anomalies) > max_results
    anomalies = anomalies[:max_results]

    unit = guess_unit(target)
    return {
        "column": target,
        "unit": unit,
        "method": chosen,
        "method_reason": reason,
        "sensitivity": sensitivity,
        "time_column": time_column if chosen in {"rolling", "seasonal"} else None,
        "grain": grain.value if grain and chosen in {"rolling", "seasonal"} else None,
        "count": len(anomalies),
        "truncated": truncated,
        "statsmodels_available": _HAS_STATSMODELS,
        "anomalies": [a.model_dump() for a in anomalies],
        "series": (
            [
                {
                    "period": str(pd.Timestamp(p).date()),
                    "label": str(lbl),
                    "value": safe_float(v),
                }
                for p, lbl, v in zip(series["period"], series["label"], series["value"])
            ]
            if series is not None
            else []
        ),
        "summary": (
            f"No anomalies detected in '{target}' using the {chosen} method."
            if not anomalies
            else (
                f"{len(anomalies)} anomal{'y' if len(anomalies) == 1 else 'ies'} "
                f"detected in '{target}' using the {chosen} method. "
                f"The most extreme is {anomalies[0].label}: "
                f"{compact_number(anomalies[0].observed, unit=unit)}"
                + (
                    f" versus an expected {compact_number(anomalies[0].expected, unit=unit)}."
                    if anomalies[0].expected is not None
                    else "."
                )
            )
        ),
    }
