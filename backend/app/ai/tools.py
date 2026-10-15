"""The safe tool registry.

This is the capability boundary of the AI layer. An LLM can *request* a tool by
name with JSON arguments; it cannot execute code, write SQL that bypasses the
guard, reach the filesystem, or touch another tenant's data. The list below is
the complete set of things any AI provider can cause to happen.

Three properties make this safe regardless of how thoroughly a prompt is
injected:

1. **Fixed vocabulary.** Tool names are looked up in a dict. An unknown name is
   an error, not a dynamic dispatch.
2. **Typed arguments.** Every tool has a Pydantic model with `extra="forbid"`.
   Malformed or extra arguments are rejected before the handler runs.
3. **Scoped context.** The handler receives a :class:`ToolContext` containing
   exactly one already-authorised DataFrame. There is no connection, no
   credential and no path in scope.

The deterministic NL planner emits calls into this same registry, so the
non-AI path and the AI path are the same code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError

from app.engine import charts as charts_mod
from app.engine.types import (
    Aggregation,
    DatasetProfile,
    FilterSpec,
    QualityReport,
    QuerySpec,
    SemanticModel,
    TimeGrain,
)
from app.engine.utils import compact_number, to_jsonable
from app.errors import BadRequestError, InsightOSError, ValidationError
from app.obs.logging import get_logger
from app.obs.metrics import metrics

log = get_logger(__name__)


@dataclass(slots=True)
class ToolContext:
    """Everything a tool is allowed to see."""

    df: pd.DataFrame
    profile: DatasetProfile
    semantics: SemanticModel
    quality: QualityReport | None = None
    dataset_name: str = "dataset"
    dataset_id: str | None = None

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.df.columns]


# =============================================================== arg models ==
class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InspectSchemaArgs(_Args):
    pass


class ColumnStatisticsArgs(_Args):
    column: str = Field(description="Name of the column to describe.")
    bins: int = Field(default=20, ge=5, le=60)


class QueryDataArgs(_Args):
    metrics: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=8,
        description="Each item: {column, agg, alias?}. agg is one of "
        "sum|mean|median|min|max|count|count_distinct|std|var|p25|p75|p90|p95.",
    )
    dimensions: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=3,
        description="Each item: {column, time_grain?}. time_grain is one of "
        "hour|day|week|month|quarter|year.",
    )
    filters: list[dict[str, Any]] = Field(default_factory=list, max_length=10)
    sort: list[dict[str, Any]] = Field(default_factory=list, max_length=3)
    limit: int = Field(default=50, ge=1, le=1000)
    include_totals: bool = True


class CalculateKpiArgs(_Args):
    column: str | None = Field(
        default=None, description="Measure column. Omitted means the primary measure."
    )
    agg: Literal[
        "sum", "mean", "median", "min", "max", "count", "count_distinct"
    ] = "sum"
    per: str | None = Field(
        default=None,
        description="Optional entity column to divide by its distinct count, e.g. "
        "column='revenue', agg='sum', per='order_id' gives average order value.",
    )
    filters: list[dict[str, Any]] = Field(default_factory=list, max_length=10)


class ComparePeriodsArgs(_Args):
    measure: str | None = None
    period_a: str | None = Field(
        default=None, description="Baseline period label, e.g. 'Aug 2026' or '2025'."
    )
    period_b: str | None = Field(default=None, description="Comparison period label.")
    grain: Literal["hour", "day", "week", "month", "quarter", "year"] | None = None
    agg: Literal["sum", "mean", "median", "count", "count_distinct"] = "sum"


class ExplainChangeArgs(_Args):
    measure: str | None = None
    dimension: str | None = Field(
        default=None, description="Dimension to attribute the change across."
    )
    period_a: str | None = None
    period_b: str | None = None
    grain: Literal["hour", "day", "week", "month", "quarter", "year"] | None = None
    top_n: int = Field(default=8, ge=1, le=25)


class CorrelationArgs(_Args):
    x: str | None = None
    y: str | None = None
    method: Literal["pearson", "spearman"] = "pearson"
    columns: list[str] | None = Field(default=None, max_length=25)


class DetectAnomaliesArgs(_Args):
    column: str | None = None
    method: Literal["auto", "iqr", "zscore", "rolling", "seasonal", "isolation"] = "auto"
    sensitivity: float = Field(default=1.0, ge=0.25, le=4.0)
    max_results: int = Field(default=25, ge=1, le=50)


class ForecastArgs(_Args):
    measure: str | None = None
    horizon: int = Field(default=3, ge=1, le=36)
    grain: Literal["hour", "day", "week", "month", "quarter", "year"] | None = None
    agg: Literal["sum", "mean", "count", "count_distinct"] = "sum"


class TrendArgs(_Args):
    measure: str | None = None
    grain: Literal["hour", "day", "week", "month", "quarter", "year"] | None = None
    agg: Literal["sum", "mean", "median", "count", "count_distinct"] = "sum"


class TopContributorsArgs(_Args):
    dimension: str | None = None
    measure: str | None = None
    agg: Literal["sum", "mean", "median", "count", "count_distinct"] = "sum"
    top_n: int = Field(default=10, ge=1, le=50)
    ascending: bool = False


class GenerateChartArgs(_Args):
    chart_type: Literal[
        "line", "bar", "hbar", "histogram", "scatter", "donut", "auto"
    ] = "auto"
    measure: str | None = None
    dimension: str | None = None
    grain: Literal["hour", "day", "week", "month", "quarter", "year"] | None = None
    second_measure: str | None = None
    top_n: int = Field(default=12, ge=2, le=30)


class DistributionArgs(_Args):
    column: str | None = None
    bins: int = Field(default=24, ge=5, le=60)


class QualityArgs(_Args):
    max_issues: int = Field(default=20, ge=1, le=60)


class SampleRowsArgs(_Args):
    n: int = Field(default=5, ge=1, le=25)


class RunSqlArgs(_Args):
    sql: str = Field(
        description="A single read-only SELECT over the table named 'dataset'.",
        max_length=4000,
    )
    row_limit: int = Field(default=200, ge=1, le=2000)


class ColumnValuesArgs(_Args):
    column: str
    limit: int = Field(default=25, ge=1, le=200)


# ================================================================== helpers ==
def _agg(value: str | None, default: Aggregation = Aggregation.SUM) -> Aggregation:
    if not value:
        return default
    try:
        return Aggregation(value)
    except ValueError as exc:
        raise BadRequestError(
            f"Unknown aggregation '{value}'. Allowed: "
            + ", ".join(a.value for a in Aggregation)
        ) from exc


def _grain(value: str | None) -> TimeGrain | None:
    if not value:
        return None
    try:
        return TimeGrain(value)
    except ValueError as exc:
        raise BadRequestError(
            f"Unknown time grain '{value}'. Allowed: "
            + ", ".join(g.value for g in TimeGrain)
        ) from exc


def _resolved_grain(ctx: ToolContext, requested: str | None) -> TimeGrain:
    from app.engine.semantics import choose_reporting_grain

    explicit = _grain(requested)
    if explicit:
        return explicit
    time_column = ctx.semantics.primary_time_column
    time_profile = ctx.profile.column(time_column) if time_column else None
    return choose_reporting_grain(
        time_profile.stats.span_days if time_profile else None,
        time_profile.stats.inferred_grain if time_profile else None,
    )


def _require_measure(ctx: ToolContext, requested: str | None) -> str:
    measure = requested or ctx.semantics.primary_measure
    if not measure:
        raise BadRequestError(
            "This dataset has no numeric measure, so there is nothing to aggregate."
        )
    from app.engine.query import resolve_column

    return resolve_column(measure, ctx.df)


def _require_time_column(ctx: ToolContext) -> str:
    time_column = ctx.semantics.primary_time_column
    if not time_column:
        raise BadRequestError(
            "No date column was detected in this dataset, so time-based analysis is "
            "not available. Columns recognised as dates: none."
        )
    return time_column


def _require_dimension(ctx: ToolContext, requested: str | None) -> str:
    if requested:
        from app.engine.query import resolve_column

        return resolve_column(requested, ctx.df)
    ranked = charts_mod._rank_dimensions(ctx.profile, ctx.semantics)
    if not ranked:
        raise BadRequestError(
            "This dataset has no usable grouping column (every candidate is "
            "constant, empty, or has too many distinct values)."
        )
    return ranked[0]


# ================================================================== handlers =
def _inspect_schema(ctx: ToolContext, args: InspectSchemaArgs) -> dict[str, Any]:
    from app.engine.sqlrun import schema_ddl

    return {
        "dataset": ctx.dataset_name,
        "row_count": ctx.profile.row_count,
        "column_count": ctx.profile.column_count,
        "duplicate_rows": ctx.profile.duplicate_rows,
        "missing_ratio": round(ctx.profile.missing_ratio, 4),
        "columns": [
            {
                "name": c.name,
                "dtype": c.dtype,
                "semantic_type": c.semantic_type.value,
                "role": c.role.value,
                "unit": c.unit,
                "missing_pct": round(c.stats.null_ratio * 100, 2),
                "distinct": c.stats.unique_count,
            }
            for c in ctx.profile.columns
        ],
        "measures": ctx.semantics.measures,
        "dimensions": ctx.semantics.dimensions,
        "time_columns": ctx.semantics.time_columns,
        "identifiers": ctx.semantics.identifiers,
        "primary_measure": ctx.semantics.primary_measure,
        "primary_time_column": ctx.semantics.primary_time_column,
        "detected_concepts": ctx.semantics.detected_concepts,
        "domain_guess": ctx.semantics.domain_guess,
        "sql_schema": schema_ddl(ctx.df),
    }


def _get_column_statistics(
    ctx: ToolContext, args: ColumnStatisticsArgs
) -> dict[str, Any]:
    from app.engine.stats import describe_column

    return describe_column(ctx.df, args.column, bins=args.bins)


def _query_data(ctx: ToolContext, args: QueryDataArgs) -> dict[str, Any]:
    from app.engine.query import run_query

    try:
        spec = QuerySpec(
            metrics=args.metrics,  # type: ignore[arg-type]
            dimensions=args.dimensions,  # type: ignore[arg-type]
            filters=args.filters,  # type: ignore[arg-type]
            sort=args.sort,  # type: ignore[arg-type]
            limit=args.limit,
            include_totals=args.include_totals,
        )
    except PydanticValidationError as exc:
        raise ValidationError(
            "Query specification is invalid.", details=_pydantic_details(exc)
        ) from exc

    result = run_query(ctx.df, spec, profile=ctx.profile)
    return result.model_dump(mode="json")


def _calculate_kpi(ctx: ToolContext, args: CalculateKpiArgs) -> dict[str, Any]:
    from app.engine.query import apply_filters, resolve_column, scalar_aggregate
    from app.engine.utils import guess_unit

    agg = _agg(args.agg)
    if agg in {Aggregation.COUNT, Aggregation.COUNT_DISTINCT} and args.column:
        # Counting works on any column, not only numeric measures.
        column = resolve_column(args.column, ctx.df)
    else:
        column = _require_measure(ctx, args.column)
    filters = [FilterSpec(**f) for f in args.filters] if args.filters else []
    frame, descriptions = apply_filters(ctx.df, filters)
    rows_considered = int(frame.shape[0])

    numerator = scalar_aggregate(frame, column, agg) if rows_considered else None
    unit = "count" if agg in {Aggregation.COUNT, Aggregation.COUNT_DISTINCT} else guess_unit(column)
    value = numerator
    per_column: str | None = None
    per_count: int | None = None

    if args.per:
        # Ratio KPI: e.g. sum(revenue) / distinct(order_id) = average order value.
        # Computed as a ratio of totals, which is what the business term means;
        # the per-row mean only coincides when every row is a whole order.
        per_column = resolve_column(args.per, ctx.df)
        per_count = int(frame[per_column].nunique(dropna=True))
        value = (numerator / per_count) if (numerator is not None and per_count) else None
        unit = guess_unit(column)

    agg_text = f"{agg.value} of '{column}'"
    basis = (
        f"{agg_text} ({compact_number(numerator, unit=guess_unit(column))}) ÷ "
        f"{per_count:,} distinct '{per_column}'"
        if per_column
        else agg_text
    )
    return {
        "column": column,
        "agg": agg.value,
        "per": per_column,
        "per_count": per_count,
        "numerator": numerator,
        "value": value,
        "formatted": compact_number(value, unit=unit),
        "unit": unit,
        "rows_considered": rows_considered,
        "total_rows": int(ctx.df.shape[0]),
        "filters_applied": descriptions,
        "basis": (
            f"{basis} over {rows_considered:,} row(s)"
            + (f" matching {', '.join(descriptions)}" if descriptions else "")
            + "."
        ),
    }


def _compare_periods(ctx: ToolContext, args: ComparePeriodsArgs) -> dict[str, Any]:
    from app.engine.trends import compare_periods

    measure = _require_measure(ctx, args.measure)
    time_column = _require_time_column(ctx)
    return compare_periods(
        ctx.df,
        time_column=time_column,
        value_column=measure,
        grain=_resolved_grain(ctx, args.grain),
        agg=_agg(args.agg),
        period_a=args.period_a,
        period_b=args.period_b,
    )


def _explain_change(ctx: ToolContext, args: ExplainChangeArgs) -> dict[str, Any]:
    from app.engine.trends import explain_change

    measure = _require_measure(ctx, args.measure)
    time_column = _require_time_column(ctx)
    candidates = None
    if args.dimension:
        from app.engine.query import resolve_column

        dimension: str | None = resolve_column(args.dimension, ctx.df)
    else:
        # Let the engine pick the dimension where the change is concentrated.
        dimension = None
        candidates = charts_mod._rank_dimensions(ctx.profile, ctx.semantics)[:6]
        if not candidates:
            raise BadRequestError(
                "This dataset has no usable grouping column to attribute the change to."
            )
    result = explain_change(
        ctx.df,
        time_column=time_column,
        value_column=measure,
        dimension=dimension,
        candidate_dimensions=candidates,
        grain=_resolved_grain(ctx, args.grain),
        period_a=args.period_a,
        period_b=args.period_b,
        top_n=args.top_n,
    )
    dimension = result["dimension"]
    result["chart"] = charts_mod.category_chart(
        [
            {
                "value": d["value"],
                "metric": d["change_abs"],
                "share_pct": d["contribution_pct"],
            }
            for d in result["drivers"]
        ],
        dimension=dimension,
        measure=f"Change in {measure}",
        total_levels=result["driver_count"],
        chart_id="explain_change",
        title=(
            f"Change in {measure} by {dimension}: "
            f"{result['overall']['baseline']['label']} → "
            f"{result['overall']['comparison']['label']}"
        ),
    ).model_dump(mode="json")
    return result


def _calculate_correlation(ctx: ToolContext, args: CorrelationArgs) -> dict[str, Any]:
    from app.engine.stats import correlate_pair, correlation_matrix

    if args.x and args.y:
        result = correlate_pair(ctx.df, args.x, args.y, method=args.method)
        result["chart"] = charts_mod.scatter_chart(
            ctx.df, args.x, args.y, correlation=result
        ).model_dump(mode="json")
        return result

    matrix = correlation_matrix(
        ctx.df, columns=args.columns, method=args.method, profile=ctx.profile
    )
    matrix["chart"] = charts_mod.correlation_heatmap_chart(matrix).model_dump(mode="json")
    return matrix


def _detect_anomalies(ctx: ToolContext, args: DetectAnomaliesArgs) -> dict[str, Any]:
    from app.engine.anomaly import detect_anomalies

    result = detect_anomalies(
        ctx.df,
        profile=ctx.profile,
        semantics=ctx.semantics,
        column=args.column,
        method=args.method,  # type: ignore[arg-type]
        sensitivity=args.sensitivity,
        max_results=args.max_results,
    )
    if result["series"]:
        from app.engine.types import ChartAnnotation, Severity

        series_frame = pd.DataFrame(result["series"])
        result["chart"] = charts_mod.time_series_chart(
            series_frame,
            value_column=result["column"],
            grain=TimeGrain(result["grain"]) if result.get("grain") else TimeGrain.DAY,
            chart_id="anomalies",
            title=f"{result['column']} with detected anomalies",
            annotations=[
                ChartAnnotation(
                    x=a["label"],
                    label=a["label"],
                    severity=Severity(a["severity"]),
                    value=a.get("observed"),
                )
                for a in result["anomalies"][:10]
            ],
        ).model_dump(mode="json")
    return result


def _generate_forecast(ctx: ToolContext, args: ForecastArgs) -> dict[str, Any]:
    from app.engine.forecast import generate_forecast

    model = generate_forecast(
        ctx.df,
        profile=ctx.profile,
        semantics=ctx.semantics,
        target_column=args.measure,
        time_column=ctx.semantics.primary_time_column,
        grain=_grain(args.grain),
        horizon=args.horizon,
        agg=_agg(args.agg),
    )
    return model.model_dump(mode="json")


def _analyse_trend(ctx: ToolContext, args: TrendArgs) -> dict[str, Any]:
    from app.engine.trends import trend_analysis

    measure = _require_measure(ctx, args.measure)
    time_column = _require_time_column(ctx)
    grain = _resolved_grain(ctx, args.grain)
    result = trend_analysis(
        ctx.df,
        time_column=time_column,
        value_column=measure,
        grain=grain,
        agg=_agg(args.agg),
    )
    result["chart"] = charts_mod.time_series_chart(
        pd.DataFrame(result["series"]), value_column=measure, grain=grain
    ).model_dump(mode="json")
    return result


def _top_contributors(ctx: ToolContext, args: TopContributorsArgs) -> dict[str, Any]:
    from app.engine.trends import top_contributors

    measure = _require_measure(ctx, args.measure)
    dimension = _require_dimension(ctx, args.dimension)
    result = top_contributors(
        ctx.df,
        dimension=dimension,
        measure=measure,
        agg=_agg(args.agg),
        top_n=args.top_n,
        ascending=args.ascending,
    )

    result["chart"] = charts_mod.category_chart(
        result["rows"],
        dimension=dimension,
        measure=measure,
        total_levels=result["level_count"],
    ).model_dump(mode="json")
    return result


def _generate_chart(ctx: ToolContext, args: GenerateChartArgs) -> dict[str, Any]:
    from app.engine.trends import build_series, top_contributors

    if args.chart_type == "scatter" or (args.measure and args.second_measure):
        x = _require_measure(ctx, args.measure)
        y = _require_measure(ctx, args.second_measure)
        return {
            "chart": charts_mod.scatter_chart(ctx.df, x, y).model_dump(mode="json")
        }

    if args.chart_type == "histogram":
        column = _require_measure(ctx, args.measure)
        return {
            "chart": charts_mod.distribution_chart(ctx.df, column).model_dump(mode="json")
        }

    if args.chart_type in {"bar", "hbar", "donut"} or (
        args.dimension and not args.grain
    ):
        measure = _require_measure(ctx, args.measure)
        dimension = _require_dimension(ctx, args.dimension)
        contributions = top_contributors(
            ctx.df, dimension=dimension, measure=measure, top_n=args.top_n
        )
        builder = (
            charts_mod.composition_chart
            if args.chart_type == "donut"
            else charts_mod.category_chart
        )
        return {
            "chart": builder(
                contributions["rows"],
                dimension=dimension,
                measure=measure,
                total_levels=contributions["level_count"],
            ).model_dump(mode="json"),
            "rows": contributions["rows"],
        }

    if args.chart_type in {"line", "auto"} and ctx.semantics.primary_time_column:
        measure = _require_measure(ctx, args.measure)
        grain = _resolved_grain(ctx, args.grain)
        series = build_series(
            ctx.df,
            time_column=_require_time_column(ctx),
            value_column=measure,
            grain=grain,
        )
        return {
            "chart": charts_mod.time_series_chart(
                series, value_column=measure, grain=grain
            ).model_dump(mode="json")
        }

    auto = charts_mod.auto_charts(
        ctx.df, profile=ctx.profile, semantics=ctx.semantics, limit=1
    )
    if not auto:
        raise BadRequestError(
            "No chart could be built from this dataset — it has no measure that can "
            "be plotted against a dimension or a date."
        )
    return {"chart": auto[0].model_dump(mode="json")}


def _analyse_distribution(ctx: ToolContext, args: DistributionArgs) -> dict[str, Any]:
    from app.engine.stats import describe_column

    column = args.column or _require_measure(ctx, None)
    description = describe_column(ctx.df, column, bins=args.bins)
    if description.get("kind") == "numeric":
        description["chart"] = charts_mod.distribution_chart(
            ctx.df, column, bins=args.bins
        ).model_dump(mode="json")
    return description


def _data_quality(ctx: ToolContext, args: QualityArgs) -> dict[str, Any]:
    if ctx.quality is None:
        raise BadRequestError("No data-quality report is available for this dataset.")
    payload = ctx.quality.model_dump(mode="json")
    payload["issues"] = payload["issues"][: args.max_issues]
    return payload


def _sample_rows(ctx: ToolContext, args: SampleRowsArgs) -> dict[str, Any]:
    from app.engine.query import sample_rows

    return {
        "columns": ctx.columns,
        "rows": sample_rows(ctx.df, args.n),
        "total_rows": int(ctx.df.shape[0]),
        "note": (
            "Sample rows are for orientation only. Every number InsightOS reports is "
            "computed over the full dataset, not this sample."
        ),
    }


def _column_values(ctx: ToolContext, args: ColumnValuesArgs) -> dict[str, Any]:
    from app.engine.query import column_values, resolve_column

    column = resolve_column(args.column, ctx.df)
    return {
        "column": column,
        "values": column_values(ctx.df, column, limit=args.limit),
        "distinct_total": int(ctx.df[column].nunique(dropna=True)),
    }


def _run_sql(ctx: ToolContext, args: RunSqlArgs) -> dict[str, Any]:
    from app.engine.sqlrun import run_sql

    return run_sql(ctx.df, args.sql, row_limit=args.row_limit)


# ================================================================ registry ===
@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    args_model: type[_Args]
    handler: Callable[[ToolContext, Any], dict[str, Any]]
    # Tools that need a date column, surfaced so the planner can avoid them.
    requires_time: bool = False
    requires_measure: bool = False

    def json_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": schema,
        }


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in (
        Tool(
            "inspect_schema",
            "List every column with its inferred type, analytic role, missing-value "
            "rate and distinct count, plus the detected business concepts. Call this "
            "first when unsure which columns exist.",
            InspectSchemaArgs,
            _inspect_schema,
        ),
        Tool(
            "get_column_statistics",
            "Full descriptive statistics for one column: count, missing, distinct, "
            "min/max/mean/median/std, percentiles, skewness and a histogram (numeric) "
            "or top values and entropy (categorical).",
            ColumnStatisticsArgs,
            _get_column_statistics,
        ),
        Tool(
            "query_data",
            "Aggregate, group, filter and sort the dataset. The general-purpose tool: "
            "supply metrics (column + aggregation), optional dimensions (with an "
            "optional time grain), filters and sort order.",
            QueryDataArgs,
            _query_data,
        ),
        Tool(
            "calculate_kpi",
            "Compute a single headline number for one measure, optionally filtered. "
            "Returns the value, a formatted label and the row count it was computed "
            "over.",
            CalculateKpiArgs,
            _calculate_kpi,
            requires_measure=True,
        ),
        Tool(
            "compare_periods",
            "Compare one measure between two time periods (defaults to the latest "
            "period versus the one before it). Returns both values, the absolute and "
            "percentage change.",
            ComparePeriodsArgs,
            _compare_periods,
            requires_time=True,
            requires_measure=True,
        ),
        Tool(
            "explain_change",
            "Explain WHY a measure changed between two periods by decomposing the "
            "change exactly across the values of a dimension, ranked by contribution. "
            "Use this for any 'why did X go up/down' question.",
            ExplainChangeArgs,
            _explain_change,
            requires_time=True,
            requires_measure=True,
        ),
        Tool(
            "analyse_trend",
            "Fit a trend to a measure over time: direction, slope, R², significance, "
            "CAGR, volatility, best and worst periods, and the full series.",
            TrendArgs,
            _analyse_trend,
            requires_time=True,
            requires_measure=True,
        ),
        Tool(
            "top_contributors",
            "Rank the values of a dimension by a measure, with each one's share of the "
            "total and the Pareto point. Set ascending=true for worst performers.",
            TopContributorsArgs,
            _top_contributors,
            requires_measure=True,
        ),
        Tool(
            "calculate_correlation",
            "Correlation between two numeric columns (with p-value, confidence "
            "interval and sample size), or the full pairwise matrix when x and y are "
            "omitted.",
            CorrelationArgs,
            _calculate_correlation,
        ),
        Tool(
            "detect_anomalies",
            "Find unusual values or periods. The method (IQR, modified z-score, "
            "rolling median, seasonal decomposition, Isolation Forest) is chosen from "
            "the data unless specified.",
            DetectAnomaliesArgs,
            _detect_anomalies,
            requires_measure=True,
        ),
        Tool(
            "generate_forecast",
            "Forecast a measure forward. Fits several models, scores them on a "
            "hold-out window, selects the best by sMAPE and returns the projection "
            "with a 95% confidence band and the full model scoreboard.",
            ForecastArgs,
            _generate_forecast,
            requires_time=True,
            requires_measure=True,
        ),
        Tool(
            "generate_chart",
            "Build a chart specification. With chart_type='auto' the type is chosen "
            "from the data's structure.",
            GenerateChartArgs,
            _generate_chart,
        ),
        Tool(
            "analyse_distribution",
            "Describe how one column's values are distributed, with a histogram and "
            "shape diagnostics.",
            DistributionArgs,
            _analyse_distribution,
        ),
        Tool(
            "data_quality_report",
            "Return the data-quality score, per-dimension sub-scores and the list of "
            "detected issues with suggested fixes.",
            QualityArgs,
            _data_quality,
        ),
        Tool(
            "sample_rows",
            "Return a few raw rows for orientation. Never use these to compute a "
            "number — use query_data for that.",
            SampleRowsArgs,
            _sample_rows,
        ),
        Tool(
            "list_column_values",
            "List the distinct values of a dimension with their row counts and share, "
            "so filters can be built with real values.",
            ColumnValuesArgs,
            _column_values,
        ),
        Tool(
            "run_sql",
            "Run a single read-only SELECT against the table 'dataset'. The statement "
            "is parsed and proved read-only before execution; anything other than a "
            "SELECT, or any filesystem/network function, is rejected.",
            RunSqlArgs,
            _run_sql,
        ),
    )
}


def tool_names() -> list[str]:
    return sorted(TOOLS)


def tool_catalogue(ctx: ToolContext | None = None) -> list[dict[str, Any]]:
    """Tool definitions for a prompt or for `GET /tools`.

    When a context is supplied, tools that cannot work on this dataset (no date
    column, no measure) are filtered out — telling a model about a tool that is
    guaranteed to fail wastes a turn and invites a hallucinated answer.
    """
    out: list[dict[str, Any]] = []
    for tool in TOOLS.values():
        if ctx is not None:
            if tool.requires_time and not ctx.semantics.primary_time_column:
                continue
            if tool.requires_measure and not (
                ctx.semantics.primary_measure or ctx.semantics.measures
            ):
                continue
        out.append(tool.json_schema())
    return out


def _pydantic_details(exc: PydanticValidationError) -> list[dict[str, Any]]:
    return [
        {
            "field": ".".join(str(p) for p in error["loc"]) or "(root)",
            "problem": error["msg"],
        }
        for error in exc.errors()[:8]
    ]


@dataclass(slots=True)
class ToolResult:
    name: str
    arguments: dict[str, Any]
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "arguments": to_jsonable(self.arguments),
            "ok": self.ok,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


def execute_tool(
    ctx: ToolContext, name: str, arguments: dict[str, Any] | None = None
) -> ToolResult:
    """Validate and run one tool call. Never raises for expected failures.

    Returning a failed :class:`ToolResult` rather than raising lets the AI loop
    show the model its own mistake and retry, and lets the deterministic path
    report a clean error to the user.
    """
    started = time.perf_counter()
    arguments = dict(arguments or {})

    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult(
            name=name,
            arguments=arguments,
            ok=False,
            error=(
                f"Unknown tool '{name}'. Available tools: {', '.join(tool_names())}."
            ),
            duration_ms=0,
        )

    try:
        parsed = tool.args_model(**arguments)
    except PydanticValidationError as exc:
        details = _pydantic_details(exc)
        return ToolResult(
            name=name,
            arguments=arguments,
            ok=False,
            error=(
                "Invalid arguments: "
                + "; ".join(f"{d['field']}: {d['problem']}" for d in details)
            ),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    try:
        with metrics.timer("tool.execute", tool=name):
            payload = tool.handler(ctx, parsed)
        duration_ms = int((time.perf_counter() - started) * 1000)
        metrics.incr("tool.ok", tool=name)
        log.info("tool_executed", extra={"tool": name, "ms": duration_ms})
        return ToolResult(
            name=name,
            arguments=arguments,
            ok=True,
            result=to_jsonable(payload),
            duration_ms=duration_ms,
        )
    except InsightOSError as exc:
        metrics.incr("tool.error", tool=name)
        log.info("tool_failed", extra={"tool": name, "error": exc.message})
        return ToolResult(
            name=name,
            arguments=arguments,
            ok=False,
            error=exc.message,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:
        metrics.incr("tool.exception", tool=name)
        log.exception("tool_crashed", extra={"tool": name})
        return ToolResult(
            name=name,
            arguments=arguments,
            ok=False,
            error=f"The {name} analysis failed unexpectedly: {str(exc)[:200]}",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
