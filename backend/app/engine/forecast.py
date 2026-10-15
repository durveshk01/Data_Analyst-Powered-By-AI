"""Forecasting with honest model selection.

Candidate models are fitted on a training window, scored on a held-out
validation window, and the winner is chosen by sMAPE (with RMSE as a
tie-break). The scoreboard of every candidate is returned alongside the
forecast, so the choice is auditable rather than asserted.

A naive baseline is always included. If no model beats "assume next period
equals this period", the forecast says so — a model that cannot beat the naive
baseline is not a model worth trusting.

All models are free and local: NumPy/sklearn for the regression and smoothing
variants, statsmodels for ETS/ARIMA when the wheel is present. Nothing is
sent anywhere.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from app.engine.types import (
    Aggregation,
    ChartAnnotation,
    ChartSeries,
    ChartSpec,
    ChartType,
    DatasetProfile,
    ForecastModel,
    ForecastPoint,
    SemanticModel,
    TimeGrain,
)
from app.engine.utils import guess_unit, safe_float
from app.errors import AnalysisError
from app.obs.logging import get_logger

log = get_logger(__name__)

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    ARIMA = None  # type: ignore[assignment]
    ExponentialSmoothing = None  # type: ignore[assignment]
    SimpleExpSmoothing = None  # type: ignore[assignment]
    _HAS_STATSMODELS = False

_MIN_POINTS = 6
_MAX_HORIZON = 36
_Z95 = 1.959964


@dataclass(slots=True)
class Candidate:
    """A fitted-and-scored forecasting model."""

    name: str
    description: str
    # Produces `h` future values given the full history.
    predict: Callable[[np.ndarray, int], np.ndarray]
    metrics: dict[str, float | None] = field(default_factory=dict)
    residual_std: float | None = None
    interval: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]] | None = None
    failed: str | None = None


# ----------------------------------------------------------------- metrics ----
def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    mask = actual != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric MAPE — bounded, and defined when an actual value is zero."""
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    mask = denominator != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100)


def _score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    predicted = np.nan_to_num(predicted, nan=float(np.mean(actual)))
    return {
        "mae": round(_mae(actual, predicted), 4),
        "rmse": round(_rmse(actual, predicted), 4),
        "mape": (lambda m: round(m, 3) if m is not None else None)(
            _mape(actual, predicted)
        ),
        "smape": round(_smape(actual, predicted), 3),
    }


# ------------------------------------------------------------- the models -----
def _naive(history: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(history[-1], h)


def _mean_model(history: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(float(np.mean(history)), h)


def _make_seasonal_naive(season: int) -> Callable[[np.ndarray, int], np.ndarray]:
    def predict(history: np.ndarray, h: int) -> np.ndarray:
        if len(history) < season:
            return _naive(history, h)
        last_cycle = history[-season:]
        return np.array([last_cycle[i % season] for i in range(h)], dtype="float64")

    return predict


def _make_moving_average(window: int) -> Callable[[np.ndarray, int], np.ndarray]:
    def predict(history: np.ndarray, h: int) -> np.ndarray:
        w = min(window, len(history))
        series = list(history.astype("float64"))
        out: list[float] = []
        for _ in range(h):
            value = float(np.mean(series[-w:]))
            out.append(value)
            series.append(value)
        return np.array(out, dtype="float64")

    return predict


def _make_ses(alpha: float) -> Callable[[np.ndarray, int], np.ndarray]:
    """Simple exponential smoothing, implemented directly so it works without
    statsmodels installed."""

    def predict(history: np.ndarray, h: int) -> np.ndarray:
        level = float(history[0])
        for value in history[1:]:
            level = alpha * float(value) + (1 - alpha) * level
        return np.repeat(level, h)

    return predict


def _make_holt(alpha: float, beta: float) -> Callable[[np.ndarray, int], np.ndarray]:
    """Holt's linear (double exponential) smoothing — level plus trend."""

    def predict(history: np.ndarray, h: int) -> np.ndarray:
        if len(history) < 2:
            return _naive(history, h)
        level = float(history[0])
        trend = float(history[1] - history[0])
        for value in history[1:]:
            previous_level = level
            level = alpha * float(value) + (1 - alpha) * (level + trend)
            trend = beta * (level - previous_level) + (1 - beta) * trend
        return np.array([level + (i + 1) * trend for i in range(h)], dtype="float64")

    return predict


def _make_linear_trend(season: int | None) -> Callable[[np.ndarray, int], np.ndarray]:
    """Ridge regression on the time index plus (optional) seasonal dummies."""

    def design(indices: np.ndarray) -> np.ndarray:
        features = [indices.astype("float64")]
        if season and season > 1:
            phase = indices % season
            for k in range(1, min(3, season // 2) + 1):
                features.append(np.sin(2 * np.pi * k * phase / season))
                features.append(np.cos(2 * np.pi * k * phase / season))
        return np.column_stack(features)

    def predict(history: np.ndarray, h: int) -> np.ndarray:
        n = len(history)
        x = np.arange(n)
        model = Ridge(alpha=1.0)
        model.fit(design(x), history.astype("float64"))
        future = np.arange(n, n + h)
        return np.asarray(model.predict(design(future)), dtype="float64")

    return predict


def _make_statsmodels_ets(
    season: int | None, trend: str | None, seasonal: str | None
) -> Callable[[np.ndarray, int], np.ndarray]:
    def predict(history: np.ndarray, h: int) -> np.ndarray:
        if ExponentialSmoothing is None:
            return _naive(history, h)
        kwargs: dict[str, Any] = {"trend": trend, "initialization_method": "estimated"}
        if seasonal and season and len(history) >= season * 2:
            kwargs["seasonal"] = seasonal
            kwargs["seasonal_periods"] = season
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ExponentialSmoothing(history.astype("float64"), **kwargs).fit(
                optimized=True
            )
        return np.asarray(fitted.forecast(h), dtype="float64")

    return predict


def _fit_arima(history: np.ndarray, order: tuple[int, int, int]) -> Any:
    # Convergence is judged by hold-out error on the scoreboard, not by the
    # optimiser's warnings, which on short business series are routine noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(history.astype("float64"), order=order).fit()


def _make_arima(order: tuple[int, int, int]) -> Callable[[np.ndarray, int], np.ndarray]:
    def predict(history: np.ndarray, h: int) -> np.ndarray:
        if ARIMA is None:
            return _naive(history, h)
        fitted = _fit_arima(history, order)
        return np.asarray(fitted.forecast(h), dtype="float64")

    return predict


def _arima_interval(
    order: tuple[int, int, int],
) -> Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]:
    def interval(history: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
        fitted = _fit_arima(history, order)
        prediction = fitted.get_forecast(h)
        frame = prediction.conf_int(alpha=0.05)
        values = np.asarray(frame, dtype="float64")
        return values[:, 0], values[:, 1]

    return interval


def build_candidates(points: int, season: int | None) -> list[Candidate]:
    """Assemble the candidate set appropriate for this series length."""
    candidates: list[Candidate] = [
        Candidate(
            "naive",
            "Assume the next period repeats the most recent value. The baseline "
            "every other model must beat.",
            _naive,
        ),
        Candidate(
            "mean",
            "Assume every future period equals the historical average.",
            _mean_model,
        ),
    ]

    if points >= 8:
        window = 3 if points < 14 else min(6, points // 3)
        candidates.append(
            Candidate(
                f"moving_average_{window}",
                f"Average of the last {window} periods, rolled forward.",
                _make_moving_average(window),
            )
        )
        candidates.append(
            Candidate(
                "exponential_smoothing",
                "Simple exponential smoothing (α=0.3): recent periods weighted "
                "more heavily, no trend assumed.",
                _make_ses(0.3),
            )
        )

    if points >= 10:
        candidates.append(
            Candidate(
                "holt_linear",
                "Holt's linear smoothing: separate level and trend components.",
                _make_holt(0.4, 0.2),
            )
        )
        candidates.append(
            Candidate(
                "linear_trend",
                "Ridge regression on the time index"
                + (
                    f" with {season}-period Fourier seasonality."
                    if season and season > 1
                    else " (straight-line trend)."
                ),
                _make_linear_trend(season if points >= (season or 0) * 2 else None),
            )
        )

    if season and season > 1 and points >= season * 2:
        candidates.append(
            Candidate(
                "seasonal_naive",
                f"Repeat the last complete {season}-period cycle.",
                _make_seasonal_naive(season),
            )
        )

    if _HAS_STATSMODELS and points >= 12:
        candidates.append(
            Candidate(
                "ets_additive_trend",
                "Exponential smoothing state-space model with an additive trend "
                "(statsmodels, parameters optimised by maximum likelihood).",
                _make_statsmodels_ets(None, "add", None),
            )
        )
        if season and season > 1 and points >= season * 2 + 2:
            candidates.append(
                Candidate(
                    "holt_winters_seasonal",
                    f"Holt-Winters with additive trend and {season}-period additive "
                    "seasonality.",
                    _make_statsmodels_ets(season, "add", "add"),
                )
            )
        for order in ((1, 1, 1), (2, 1, 1), (1, 1, 0)):
            candidates.append(
                Candidate(
                    f"arima{order}".replace(" ", ""),
                    f"ARIMA{order}: autoregressive integrated moving average.",
                    _make_arima(order),
                    interval=_arima_interval(order),
                )
            )

    return candidates


# -------------------------------------------------------------- evaluation ----
def _validation_split(n: int, horizon: int) -> int:
    """Size of the hold-out window."""
    return max(2, min(horizon, max(2, n // 4), n - _MIN_POINTS + 2, 12))


def _future_periods(last: pd.Timestamp, grain: TimeGrain, h: int) -> list[pd.Timestamp]:
    offset = {
        TimeGrain.HOUR: pd.DateOffset(hours=1),
        TimeGrain.DAY: pd.DateOffset(days=1),
        TimeGrain.WEEK: pd.DateOffset(weeks=1),
        TimeGrain.MONTH: pd.DateOffset(months=1),
        TimeGrain.QUARTER: pd.DateOffset(months=3),
        TimeGrain.YEAR: pd.DateOffset(years=1),
    }[grain]
    out: list[pd.Timestamp] = []
    cursor = last
    for _ in range(h):
        cursor = cursor + offset
        out.append(cursor)
    return out


def _label_period(timestamp: pd.Timestamp, grain: TimeGrain) -> str:
    if grain == TimeGrain.HOUR:
        return timestamp.strftime("%Y-%m-%d %H:00")
    if grain == TimeGrain.DAY:
        return timestamp.strftime("%Y-%m-%d")
    if grain == TimeGrain.WEEK:
        return f"W{timestamp.isocalendar().week:02d} {timestamp.year}"
    if grain == TimeGrain.MONTH:
        return timestamp.strftime("%b %Y")
    if grain == TimeGrain.QUARTER:
        return f"Q{((timestamp.month - 1) // 3) + 1} {timestamp.year}"
    return str(timestamp.year)


def _seasonal_period_for(grain: TimeGrain) -> int | None:
    return {
        TimeGrain.HOUR: 24,
        TimeGrain.DAY: 7,
        TimeGrain.WEEK: 52,
        TimeGrain.MONTH: 12,
        TimeGrain.QUARTER: 4,
        TimeGrain.YEAR: None,
    }[grain]


def generate_forecast(
    df: pd.DataFrame,
    *,
    profile: DatasetProfile,
    semantics: SemanticModel,
    target_column: str | None = None,
    time_column: str | None = None,
    grain: TimeGrain | None = None,
    horizon: int = 3,
    agg: Aggregation = Aggregation.SUM,
) -> ForecastModel:
    """Fit, score and select a forecasting model, then project `horizon` ahead."""
    from app.engine.semantics import choose_reporting_grain
    from app.engine.trends import build_series

    target = target_column or semantics.primary_measure
    tcol = time_column or semantics.primary_time_column
    if not target:
        raise AnalysisError("No numeric measure was detected, so nothing can be forecast.")
    if not tcol:
        raise AnalysisError(
            "No date column was detected in this dataset. Forecasting needs a time axis."
        )

    horizon = int(max(1, min(horizon, _MAX_HORIZON)))

    if grain is None:
        time_profile = profile.column(tcol)
        grain = choose_reporting_grain(
            time_profile.stats.span_days if time_profile else None,
            time_profile.stats.inferred_grain if time_profile else None,
        )

    from app.engine.trends import partial_note

    # Training on a period that is still in progress would teach every model
    # that the series just collapsed.
    series = build_series(
        df, time_column=tcol, value_column=target, grain=grain, agg=agg,
        complete_only=True,
    )
    values = series["value"].to_numpy(dtype="float64")
    n = values.size
    unit = guess_unit(target)
    caveats: list[str] = []
    if note := partial_note(series):
        caveats.append(
            note + " The forecast therefore starts with that period, projected in full."
        )

    if n < _MIN_POINTS:
        raise AnalysisError(
            f"Only {n} {grain.label} period(s) of history are available; at least "
            f"{_MIN_POINTS} are needed to validate a forecast. Try a coarser grain "
            "or a longer date range."
        )

    season = _seasonal_period_for(grain)
    if season and n < season * 2:
        caveats.append(
            f"History covers {n} {grain.label} periods, less than two full "
            f"{season}-period cycles, so seasonality cannot be estimated reliably."
        )
        season_for_models = season if n >= season + 2 else None
    else:
        season_for_models = season

    holdout = _validation_split(n, horizon)
    train, validation = values[:-holdout], values[-holdout:]

    candidates = build_candidates(n, season_for_models)
    scoreboard: list[dict[str, Any]] = []
    scored: list[Candidate] = []

    for candidate in candidates:
        try:
            predicted = np.asarray(
                candidate.predict(train, holdout), dtype="float64"
            )[:holdout]
            if predicted.size != holdout or not np.all(np.isfinite(predicted)):
                raise ValueError("model produced non-finite predictions")
            candidate.metrics = _score(validation, predicted)
            candidate.residual_std = float(np.std(validation - predicted, ddof=1)) if holdout > 1 else None
            scored.append(candidate)
            scoreboard.append(
                {
                    "model": candidate.name,
                    "description": candidate.description,
                    **candidate.metrics,
                    "selected": False,
                }
            )
        except Exception as exc:
            candidate.failed = str(exc)[:160]
            scoreboard.append(
                {
                    "model": candidate.name,
                    "description": candidate.description,
                    "error": candidate.failed,
                    "selected": False,
                }
            )
            log.info(
                "forecast_candidate_failed",
                extra={"model": candidate.name, "error": candidate.failed},
            )

    if not scored:
        raise AnalysisError(
            "No forecasting model could be fitted to this series. The history may be "
            "too short or contain too many gaps."
        )

    def rank(candidate: Candidate) -> tuple[float, float]:
        smape = candidate.metrics.get("smape")
        rmse = candidate.metrics.get("rmse")
        return (
            smape if smape is not None else 1e18,
            rmse if rmse is not None else 1e18,
        )

    scored.sort(key=rank)
    winner = scored[0]
    naive = next((c for c in scored if c.name == "naive"), None)

    for entry in scoreboard:
        if entry["model"] == winner.name:
            entry["selected"] = True

    beats_naive = True
    if naive is not None and winner.name != "naive":
        w, b = rank(winner)[0], rank(naive)[0]
        beats_naive = w < b
        if not beats_naive:
            caveats.append(
                "No model outperformed the naive baseline on the validation window, "
                "so treat this projection as a rough continuation rather than a "
                "prediction."
            )
    elif winner.name == "naive":
        caveats.append(
            "The naive baseline scored best: this series has no trend or seasonal "
            "structure that the candidate models could exploit."
        )

    # --- refit on the full history and project forward ---------------------
    try:
        forecast_values = np.asarray(winner.predict(values, horizon), dtype="float64")
    except Exception as exc:
        raise AnalysisError(
            f"The selected model ({winner.name}) could not be refitted on the full "
            f"history: {str(exc)[:160]}"
        ) from exc

    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    interval_basis = "empirical"
    if winner.interval is not None:
        try:
            lower, upper = winner.interval(values, horizon)
            interval_basis = "model"
        except Exception:
            lower = upper = None

    if lower is None or upper is None:
        # Empirical interval from validation residuals, widened with the square
        # root of the horizon (uncertainty compounds the further out you go).
        sigma = winner.residual_std
        if sigma is None or not np.isfinite(sigma) or sigma == 0:
            sigma = float(np.std(np.diff(values), ddof=1)) if n > 2 else 0.0
        steps = np.sqrt(np.arange(1, horizon + 1, dtype="float64"))
        margin = _Z95 * float(sigma) * steps
        lower = forecast_values - margin
        upper = forecast_values + margin

    # A non-negative measure cannot have a negative lower bound.
    non_negative = bool(np.all(values >= 0))
    if non_negative:
        lower = np.maximum(lower, 0.0)
        forecast_values = np.maximum(forecast_values, 0.0)

    last_period = pd.Timestamp(series["period"].iloc[-1])
    future = _future_periods(last_period, grain, horizon)

    points = [
        ForecastPoint(
            period=_label_period(ts, grain),
            value=round(float(forecast_values[i]), 4),
            lower=round(float(lower[i]), 4),
            upper=round(float(upper[i]), 4),
        )
        for i, ts in enumerate(future)
    ]

    history = [
        {
            "period": str(pd.Timestamp(p).date()),
            "label": str(lbl),
            "value": safe_float(v),
        }
        for p, lbl, v in zip(series["period"], series["label"], series["value"])
    ]

    if interval_basis == "empirical":
        caveats.append(
            "The confidence band is derived from validation residuals and widens "
            "with the forecast horizon; it reflects past error, not future shocks."
        )

    chart = _forecast_chart(
        history=history,
        points=points,
        target=target,
        unit=unit,
        grain=grain,
        model_name=winner.name,
    )

    model = ForecastModel(
        target_column=target,
        time_column=tcol,
        grain=grain,
        horizon=horizon,
        model_name=winner.name,
        model_reason=(
            f"{winner.description} Selected from {len(scored)} candidates: lowest "
            f"sMAPE ({winner.metrics.get('smape')}%) on a {holdout}-period hold-out."
            + ("" if beats_naive else " It did not beat the naive baseline.")
        ),
        metrics={
            **winner.metrics,
            "validation_periods": float(holdout),
            "training_periods": float(n - holdout),
            "beats_naive": 1.0 if beats_naive else 0.0,
        },
        candidates=scoreboard,
        history=history,
        points=points,
        chart=chart,
        caveats=caveats,
    )

    log.info(
        "forecast_generated",
        extra={
            "target": target,
            "model": winner.name,
            "grain": grain.value,
            "horizon": horizon,
            "smape": winner.metrics.get("smape"),
        },
    )
    return model


def _forecast_chart(
    *,
    history: list[dict[str, Any]],
    points: list[ForecastPoint],
    target: str,
    unit: str,
    grain: TimeGrain,
    model_name: str,
) -> ChartSpec:
    data: list[dict[str, Any]] = [
        {
            "label": row["label"],
            "actual": row["value"],
            "forecast": None,
            "lower": None,
            "upper": None,
        }
        for row in history
    ]
    # Join the lines: the last actual point is also the forecast's anchor.
    if data:
        data[-1]["forecast"] = data[-1]["actual"]
        data[-1]["lower"] = data[-1]["actual"]
        data[-1]["upper"] = data[-1]["actual"]
    for point in points:
        data.append(
            {
                "label": point.period,
                "actual": None,
                "forecast": point.value,
                "lower": point.lower,
                "upper": point.upper,
            }
        )

    return ChartSpec(
        id="forecast",
        type=ChartType.FORECAST,
        title=f"{target} — {len(points)} {grain.label} forecast",
        subtitle=f"Model: {model_name} · 95% confidence band",
        x_key="label",
        x_label=grain.label.capitalize(),
        y_label=target,
        series=[
            ChartSeries(key="actual", label="Actual", unit=unit),  # type: ignore[arg-type]
            ChartSeries(key="forecast", label="Forecast", unit=unit),  # type: ignore[arg-type]
            ChartSeries(key="lower", label="Lower 95%", unit=unit),  # type: ignore[arg-type]
            ChartSeries(key="upper", label="Upper 95%", unit=unit),  # type: ignore[arg-type]
        ],
        data=data,
        annotations=(
            [
                ChartAnnotation(
                    x=points[0].period,
                    label="Forecast starts",
                    severity="info",  # type: ignore[arg-type]
                )
            ]
            if points
            else []
        ),
        reason="A forecast chart shows actuals, projection and uncertainty on one axis.",
        meta={"model": model_name, "horizon": len(points), "grain": grain.value},
    )
