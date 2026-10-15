"""Safe query execution over a pandas DataFrame.

This is the single chokepoint for *all* data access in InsightOS — the HTTP
API, the automatic insight engine, and the AI tool layer all go through
:func:`run_query`. It accepts only a validated :class:`QuerySpec`; there is no
code path that evaluates a user- or model-supplied expression.

What makes it safe:

* every column name is checked against the real schema before use, so an
  unknown name is a 4xx and never an attribute lookup,
* `DataFrame.query` / `eval` are never used (they execute Python),
* aggregations come from a fixed map of pandas callables,
* string filters use `str.contains(..., regex=False)` so a filter value can
  never become a regular expression (and so cannot cause catastrophic
  backtracking),
* results are bounded by `limit`, and the spec caps `limit` itself.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.engine.types import (
    Aggregation,
    DatasetProfile,
    DimensionSpec,
    FilterOp,
    FilterSpec,
    MetricSpec,
    QueryResult,
    QuerySpec,
    SemanticType,
    TimeGrain,
)
from app.engine.utils import dataframe_to_records, safe_float, to_jsonable
from app.errors import AnalysisError, BadRequestError
from app.obs.logging import get_logger
from app.obs.metrics import metrics

log = get_logger(__name__)

# Fixed aggregation vocabulary. Nothing outside this map is callable.
_AGG_FUNCS: dict[Aggregation, str | Callable[[pd.Series], Any]] = {
    Aggregation.SUM: "sum",
    Aggregation.MEAN: "mean",
    Aggregation.MEDIAN: "median",
    Aggregation.MIN: "min",
    Aggregation.MAX: "max",
    Aggregation.COUNT: "count",
    Aggregation.COUNT_DISTINCT: "nunique",
    Aggregation.STD: "std",
    Aggregation.VAR: "var",
    Aggregation.P25: lambda s: s.quantile(0.25),
    Aggregation.P75: lambda s: s.quantile(0.75),
    Aggregation.P90: lambda s: s.quantile(0.90),
    Aggregation.P95: lambda s: s.quantile(0.95),
}

_NUMERIC_ONLY_AGGS = {
    Aggregation.SUM,
    Aggregation.MEAN,
    Aggregation.MEDIAN,
    Aggregation.STD,
    Aggregation.VAR,
    Aggregation.P25,
    Aggregation.P75,
    Aggregation.P90,
    Aggregation.P95,
}


# ------------------------------------------------------------- validation ----
def resolve_column(name: str, df: pd.DataFrame) -> str:
    """Resolve a column name, tolerating case and whitespace differences."""
    if name in df.columns:
        return name
    lowered = {str(c).strip().lower(): str(c) for c in df.columns}
    key = str(name).strip().lower()
    if key in lowered:
        return lowered[key]
    underscored = {str(c).strip().lower().replace(" ", "_"): str(c) for c in df.columns}
    if key.replace(" ", "_") in underscored:
        return underscored[key.replace(" ", "_")]
    raise BadRequestError(
        f"Column '{name}' does not exist in this dataset. "
        f"Available: {', '.join(str(c) for c in list(df.columns)[:20])}"
        + ("…" if df.shape[1] > 20 else "")
    )


def _coerce_filter_value(series: pd.Series, value: Any) -> Any:
    """Coerce a JSON filter value to the column's dtype."""
    if value is None:
        return None
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(value, errors="coerce")
        if isinstance(parsed, pd.Series):
            return parsed
        if pd.isna(parsed):
            raise BadRequestError(f"'{value}' is not a valid date.")
        return parsed
    if pd.api.types.is_numeric_dtype(series) and not isinstance(value, bool):
        coerced = pd.to_numeric(value, errors="coerce")
        if coerced is None or (np.isscalar(coerced) and pd.isna(coerced)):
            raise BadRequestError(f"'{value}' is not a valid number for this column.")
        return coerced
    if pd.api.types.is_bool_dtype(series) or isinstance(series.dtype, pd.BooleanDtype):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "yes", "1", "y"}
    return value


# ----------------------------------------------------------------- filters ----
def apply_filters(
    df: pd.DataFrame, filters: list[FilterSpec]
) -> tuple[pd.DataFrame, list[str]]:
    """Apply filters with a boolean mask. Returns (filtered, descriptions)."""
    if not filters:
        return df, []

    mask = pd.Series(True, index=df.index)
    descriptions: list[str] = []

    for spec in filters:
        column = resolve_column(spec.column, df)
        series = df[column]
        op = spec.op

        if op == FilterOp.IS_NULL:
            condition = series.isna()
            descriptions.append(f"{column} is empty")
        elif op == FilterOp.NOT_NULL:
            condition = series.notna()
            descriptions.append(f"{column} is not empty")
        elif op in {FilterOp.CONTAINS, FilterOp.STARTS_WITH, FilterOp.ENDS_WITH}:
            text = series.astype("string")
            needle = str(spec.value)
            if op == FilterOp.CONTAINS:
                # regex=False is load-bearing: it stops a filter value from
                # being interpreted as a pattern.
                condition = text.str.contains(needle, case=False, na=False, regex=False)
                descriptions.append(f"{column} contains '{needle}'")
            elif op == FilterOp.STARTS_WITH:
                condition = text.str.lower().str.startswith(needle.lower(), na=False)
                descriptions.append(f"{column} starts with '{needle}'")
            else:
                condition = text.str.lower().str.endswith(needle.lower(), na=False)
                descriptions.append(f"{column} ends with '{needle}'")
        elif op == FilterOp.BETWEEN:
            low = _coerce_filter_value(series, spec.value[0])
            high = _coerce_filter_value(series, spec.value[1])
            if low is not None and high is not None and low > high:
                low, high = high, low
            condition = series.between(low, high, inclusive="both")
            descriptions.append(f"{column} between {low} and {high}")
        elif op in {FilterOp.IN, FilterOp.NOT_IN}:
            values = [_coerce_filter_value(series, v) for v in spec.value]
            inside = series.isin(values)
            condition = inside if op == FilterOp.IN else ~inside
            verb = "is one of" if op == FilterOp.IN else "is not one of"
            descriptions.append(f"{column} {verb} {values[:5]}")
        else:
            value = _coerce_filter_value(series, spec.value)
            if op == FilterOp.EQ:
                if isinstance(value, str):
                    condition = series.astype("string").str.lower() == value.lower()
                else:
                    condition = series == value
                descriptions.append(f"{column} = {value}")
            elif op == FilterOp.NE:
                if isinstance(value, str):
                    condition = series.astype("string").str.lower() != value.lower()
                else:
                    condition = series != value
                descriptions.append(f"{column} != {value}")
            elif op == FilterOp.GT:
                condition = series > value
                descriptions.append(f"{column} > {value}")
            elif op == FilterOp.GTE:
                condition = series >= value
                descriptions.append(f"{column} >= {value}")
            elif op == FilterOp.LT:
                condition = series < value
                descriptions.append(f"{column} < {value}")
            elif op == FilterOp.LTE:
                condition = series <= value
                descriptions.append(f"{column} <= {value}")
            else:  # pragma: no cover - FilterOp is exhaustive
                raise BadRequestError(f"Unsupported filter operator '{op}'.")

        mask &= condition.fillna(False).astype(bool)

    return df[mask], descriptions


# -------------------------------------------------------------- dimensions ----
def build_dimension_series(
    df: pd.DataFrame, spec: DimensionSpec
) -> tuple[pd.Series, str]:
    """Materialise a grouping key, applying a time grain when requested."""
    column = resolve_column(spec.column, df)
    series = df[column]

    if spec.time_grain is None:
        return series, spec.alias or column

    grain = spec.time_grain
    if not pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().sum() == 0:
            raise BadRequestError(
                f"Column '{column}' cannot be grouped by {grain.value} because it "
                "holds no valid dates."
            )
        series = parsed

    if grain == TimeGrain.HOUR:
        keys = series.dt.floor("h")
    elif grain == TimeGrain.DAY:
        keys = series.dt.normalize()
    elif grain == TimeGrain.WEEK:
        # A "W-SUN" period ends on Sunday, so its start_time is the Monday that
        # opens the ISO week — the same bucket time_series() produces.
        keys = series.dt.to_period("W-SUN").dt.start_time
    elif grain == TimeGrain.MONTH:
        keys = series.dt.to_period("M").dt.start_time
    elif grain == TimeGrain.QUARTER:
        keys = series.dt.to_period("Q").dt.start_time
    else:
        keys = series.dt.to_period("Y").dt.start_time

    return keys, spec.output_name


# ------------------------------------------------------------- aggregation ----
def _validate_metric(df: pd.DataFrame, spec: MetricSpec) -> str:
    column = resolve_column(spec.column, df)
    series = df[column]
    if spec.agg in _NUMERIC_ONLY_AGGS and not pd.api.types.is_numeric_dtype(series):
        raise BadRequestError(
            f"'{spec.agg.value}' needs a numeric column, but '{column}' is "
            f"{series.dtype}. Use count or count_distinct instead."
        )
    return column


def _aggregate_series(series: pd.Series, agg: Aggregation) -> Any:
    func = _AGG_FUNCS[agg]
    if callable(func):
        return func(series)
    if agg == Aggregation.COUNT:
        return series.count()
    if agg == Aggregation.COUNT_DISTINCT:
        return series.nunique()
    return getattr(series, str(func))()


def _grand_totals(df: pd.DataFrame, metrics_spec: list[MetricSpec]) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for spec in metrics_spec:
        column = _validate_metric(df, spec)
        totals[spec.output_name] = to_jsonable(
            _aggregate_series(df[column], spec.agg)
        )
    return totals


def run_query(
    df: pd.DataFrame,
    spec: QuerySpec,
    *,
    profile: DatasetProfile | None = None,
) -> QueryResult:
    """Execute a validated query spec. The only data-access path in InsightOS."""
    started = time.perf_counter()

    if df.empty:
        raise AnalysisError("Dataset has no rows to analyse.")

    with metrics.timer("query.execute"):
        filtered, filter_descriptions = apply_filters(df, spec.filters)

        if filtered.empty:
            return QueryResult(
                columns=[d.output_name for d in spec.dimensions]
                + [m.output_name for m in spec.metrics],
                rows=[],
                row_count=0,
                totals=None,
                duration_ms=int((time.perf_counter() - started) * 1000),
                spec=spec,
                applied_filters=filter_descriptions,
            )

        # ---- no dimensions: scalar aggregate --------------------------------
        if not spec.dimensions:
            row = _grand_totals(filtered, spec.metrics)
            row["_row_count"] = int(len(filtered))
            return QueryResult(
                columns=list(row.keys()),
                rows=[row],
                row_count=1,
                totals=None,
                duration_ms=int((time.perf_counter() - started) * 1000),
                spec=spec,
                applied_filters=filter_descriptions,
            )

        # ---- grouped aggregate ----------------------------------------------
        group_frame = pd.DataFrame(index=filtered.index)
        dimension_names: list[str] = []
        for dim in spec.dimensions:
            keys, name = build_dimension_series(filtered, dim)
            if name in group_frame.columns:
                raise BadRequestError(f"Duplicate dimension output name '{name}'.")
            group_frame[name] = keys.values
            dimension_names.append(name)

        metric_columns: dict[str, tuple[str, Aggregation]] = {}
        work = group_frame.copy()
        for metric in spec.metrics:
            column = _validate_metric(filtered, metric)
            temp_name = f"__m_{metric.output_name}"
            work[temp_name] = filtered[column].values
            metric_columns[metric.output_name] = (temp_name, metric.agg)

        if not metric_columns:
            # Dimension-only query: distinct values with row counts.
            counted = (
                work.groupby(dimension_names, dropna=False, observed=True)
                .size()
                .reset_index(name="row_count")
            )
            grouped = counted
            output_metric_names = ["row_count"]
        else:
            agg_map = {
                temp: (_AGG_FUNCS[agg] if not callable(_AGG_FUNCS[agg]) else _AGG_FUNCS[agg])
                for temp, agg in metric_columns.values()
            }
            # pandas needs the aggregation keyed by the temp column name.
            grouped = (
                work.groupby(dimension_names, dropna=False, observed=True)
                .agg(agg_map)
                .reset_index()
            )
            rename = {temp: out for out, (temp, _) in metric_columns.items()}
            grouped = grouped.rename(columns=rename)
            output_metric_names = list(metric_columns.keys())

        # ---- sort ------------------------------------------------------------
        if spec.sort:
            by: list[str] = []
            ascending: list[bool] = []
            for sort_spec in spec.sort:
                target = sort_spec.by
                if target not in grouped.columns:
                    # Allow sorting by the raw column name of a metric.
                    match = next(
                        (
                            name
                            for name in output_metric_names
                            if name.startswith(f"{target}_") or name == target
                        ),
                        None,
                    )
                    if match is None and target in dimension_names:
                        match = target
                    if match is None:
                        raise BadRequestError(
                            f"Cannot sort by '{target}'. Available: "
                            f"{', '.join(list(grouped.columns))}"
                        )
                    target = match
                by.append(target)
                ascending.append(not sort_spec.desc)
            grouped = grouped.sort_values(by=by, ascending=ascending, kind="mergesort")
        elif output_metric_names and output_metric_names[0] in grouped.columns:
            first = output_metric_names[0]
            if pd.api.types.is_numeric_dtype(grouped[first]):
                grouped = grouped.sort_values(by=first, ascending=False, kind="mergesort")
        elif dimension_names:
            grouped = grouped.sort_values(by=dimension_names, kind="mergesort")

        total_groups = int(len(grouped))
        window = grouped.iloc[spec.offset : spec.offset + spec.limit]
        truncated = total_groups > spec.offset + spec.limit

        totals = _grand_totals(filtered, spec.metrics) if spec.include_totals else None
        if totals is not None:
            totals["_group_count"] = total_groups
            totals["_row_count"] = int(len(filtered))

        rows = dataframe_to_records(window)
        duration_ms = int((time.perf_counter() - started) * 1000)

    return QueryResult(
        columns=list(window.columns.astype(str)),
        rows=rows,
        row_count=total_groups,
        truncated=truncated,
        totals=totals,
        duration_ms=duration_ms,
        spec=spec,
        applied_filters=filter_descriptions,
    )


# ---------------------------------------------------------------- helpers -----
def scalar_aggregate(
    df: pd.DataFrame,
    column: str,
    agg: Aggregation,
    *,
    filters: list[FilterSpec] | None = None,
) -> float | None:
    """Convenience wrapper used throughout the engine for a single number."""
    frame, _ = apply_filters(df, filters or [])
    if frame.empty:
        return None
    resolved = resolve_column(column, frame)
    return safe_float(_aggregate_series(frame[resolved], agg))


def time_series(
    df: pd.DataFrame,
    *,
    time_column: str,
    value_column: str,
    grain: TimeGrain,
    agg: Aggregation = Aggregation.SUM,
    filters: list[FilterSpec] | None = None,
    fill_gaps: bool = True,
) -> pd.DataFrame:
    """Aggregate to a regular time series with a `period`/`value` shape.

    Gap filling matters: a month with no rows is a zero, not a missing point,
    and leaving it out would make a trend line lie.
    """
    frame, _ = apply_filters(df, filters or [])
    tcol = resolve_column(time_column, frame)
    vcol = resolve_column(value_column, frame)

    times = frame[tcol]
    if not pd.api.types.is_datetime64_any_dtype(times):
        times = pd.to_datetime(times, errors="coerce")
    work = pd.DataFrame({"__t": times, "__v": frame[vcol].values}).dropna(subset=["__t"])
    if work.empty:
        return pd.DataFrame(columns=["period", "value"])

    work = work.set_index("__t").sort_index()
    if grain == TimeGrain.WEEK:
        # Default weekly resampling closes bins on the anchor day and labels them
        # by their end; ISO weeks start on Monday, so close and label left.
        resampled = work["__v"].resample("W-MON", label="left", closed="left")
    else:
        resampled = work["__v"].resample(grain.pandas_freq)
    series = _resample_agg(resampled, agg)

    if not fill_gaps:
        series = series.dropna()
    else:
        series = series.fillna(0.0) if agg in {Aggregation.SUM, Aggregation.COUNT, Aggregation.COUNT_DISTINCT} else series.dropna()

    out = series.reset_index()
    out.columns = ["period", "value"]
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"]).reset_index(drop=True)


def _resample_agg(resampled: Any, agg: Aggregation) -> pd.Series:
    if agg == Aggregation.SUM:
        return resampled.sum(min_count=0)
    if agg == Aggregation.MEAN:
        return resampled.mean()
    if agg == Aggregation.MEDIAN:
        return resampled.median()
    if agg == Aggregation.MIN:
        return resampled.min()
    if agg == Aggregation.MAX:
        return resampled.max()
    if agg == Aggregation.COUNT:
        return resampled.count()
    if agg == Aggregation.COUNT_DISTINCT:
        return resampled.nunique()
    if agg == Aggregation.STD:
        return resampled.std()
    if agg == Aggregation.VAR:
        return resampled.var()
    quantiles = {
        Aggregation.P25: 0.25,
        Aggregation.P75: 0.75,
        Aggregation.P90: 0.90,
        Aggregation.P95: 0.95,
    }
    if agg in quantiles:
        return resampled.quantile(quantiles[agg])
    return resampled.sum(min_count=0)  # pragma: no cover


def sample_rows(df: pd.DataFrame, n: int = 5) -> list[dict[str, Any]]:
    """A small, bounded sample for display. Never used to compute a number."""
    n = max(1, min(int(n), 50))
    return dataframe_to_records(df.head(n))


def column_values(
    df: pd.DataFrame, column: str, *, limit: int = 50
) -> list[dict[str, Any]]:
    """Distinct values of a dimension with counts — used for filter pickers."""
    resolved = resolve_column(column, df)
    counts = df[resolved].value_counts(dropna=True).head(max(1, min(limit, 500)))
    total = max(int(df[resolved].notna().sum()), 1)
    return [
        {
            "value": to_jsonable(value),
            "count": int(count),
            "share": round(float(count) / total * 100, 2),
        }
        for value, count in counts.items()
    ]
