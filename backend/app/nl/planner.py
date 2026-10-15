"""Intent → tool call planning.

Maps a :class:`~app.nl.intent.ParsedIntent` onto a call into the shared tool
registry. Keeping this translation in one small module is what lets the
deterministic path and the LLM path converge: both ultimately produce
``(tool_name, arguments)`` pairs that :func:`app.ai.tools.execute_tool`
validates and runs.

The planner also degrades gracefully: if an intent needs a time column or a
measure the dataset does not have, it substitutes the closest analysis that
*can* be answered and records why, instead of failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.ai.tools import ToolContext
from app.engine.types import Aggregation
from app.nl.intent import ParsedIntent
from app.obs.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class ToolPlan:
    tool: str
    arguments: dict[str, Any]
    rationale: str
    fallback_notes: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


def _period_pair(intent: ParsedIntent) -> tuple[str | None, str | None]:
    """Order two mentioned periods as (baseline, comparison)."""
    if len(intent.periods) >= 2:
        a, b = intent.periods[0], intent.periods[1]
        # "compare 2026 and 2025" should still baseline on the earlier one.
        if a.isdigit() and b.isdigit() and int(a) > int(b):
            a, b = b, a
        return a, b
    if len(intent.periods) == 1:
        return None, intent.periods[0]
    return None, None


def plan(intent: ParsedIntent, ctx: ToolContext) -> ToolPlan:
    """Choose the tool that answers `intent` for this dataset."""
    notes: list[str] = []
    has_time = bool(ctx.semantics.primary_time_column)
    has_measure = bool(ctx.semantics.primary_measure or ctx.semantics.measures)
    name = intent.intent

    # ---- intents that need a time axis -----------------------------------
    if name in {"time_series", "growth", "forecast", "compare_periods", "why_change"} and not has_time:
        notes.append(
            "No date column was detected, so a time-based analysis is not possible. "
            + (
                "Showing the breakdown by the most informative dimension instead."
                if has_measure
                else "Showing the dataset structure instead."
            )
        )
        name = "top_n" if has_measure else "schema"

    if name in {"aggregate", "count", "top_n", "bottom_n", "breakdown", "distribution"} and not has_measure:
        if name != "count":
            notes.append(
                "No numeric measure was detected in this dataset, so there is nothing "
                "to aggregate. Showing the dataset structure instead."
            )
            name = "schema"

    # ---- dispatch ---------------------------------------------------------
    if name == "schema":
        return ToolPlan(
            "inspect_schema",
            {},
            "The question is about the dataset's structure rather than its values.",
            notes,
        )

    if name == "sample":
        return ToolPlan(
            "sample_rows",
            {"n": min(intent.limit, 25)},
            "The question asks to see raw rows.",
            notes,
        )

    if name == "quality":
        return ToolPlan(
            "data_quality_report",
            {"max_issues": 20},
            "The question is about data quality rather than a metric.",
            notes,
        )

    if name == "anomaly":
        return ToolPlan(
            "detect_anomalies",
            {
                "column": intent.measure,
                "method": "auto",
                "sensitivity": 1.0,
                "max_results": 25,
            },
            "The question asks for unusual values, so anomaly detection runs with the "
            "method chosen from the data.",
            notes,
        )

    if name == "forecast":
        return ToolPlan(
            "generate_forecast",
            {
                "measure": intent.measure,
                "horizon": intent.horizon,
                "grain": intent.time_grain.value if intent.time_grain else None,
                "agg": _forecast_agg(intent.aggregation),
            },
            f"The question asks for a projection {intent.horizon} period(s) ahead.",
            notes,
        )

    if name == "why_change":
        baseline, comparison = _period_pair(intent)
        return ToolPlan(
            "explain_change",
            {
                "measure": intent.measure,
                "dimension": intent.dimension,
                "period_a": baseline,
                "period_b": comparison,
                "grain": intent.time_grain.value if intent.time_grain else None,
                "top_n": 8,
            },
            "A 'why did it change' question is answered by decomposing the change "
            "across a dimension, so the attribution is arithmetic rather than a guess.",
            notes,
        )

    if name == "compare_periods":
        baseline, comparison = _period_pair(intent)
        return ToolPlan(
            "compare_periods",
            {
                "measure": intent.measure,
                "period_a": baseline,
                "period_b": comparison,
                "grain": intent.time_grain.value if intent.time_grain else None,
                "agg": _compare_agg(intent.aggregation),
            },
            "The question names two periods to compare.",
            notes,
        )

    if name == "correlation":
        return ToolPlan(
            "calculate_correlation",
            {
                "x": intent.measure,
                "y": intent.second_measure,
                "method": "pearson",
            }
            if intent.measure and intent.second_measure
            else {"method": "pearson"},
            "The question asks about a relationship between measures.",
            notes,
        )

    if name == "distribution":
        return ToolPlan(
            "analyse_distribution",
            {"column": intent.measure, "bins": 24},
            "The question asks how values are spread, which a distribution answers.",
            notes,
        )

    if name in {"time_series", "growth"}:
        return ToolPlan(
            "analyse_trend",
            {
                "measure": intent.measure,
                "grain": intent.time_grain.value if intent.time_grain else None,
                "agg": _compare_agg(intent.aggregation),
            },
            f"The question asks for {intent.measure or 'the primary measure'} over "
            "time, so a trend analysis is run at the requested grain.",
            notes,
        )

    if name in {"top_n", "bottom_n", "breakdown"}:
        return ToolPlan(
            "top_contributors",
            {
                "dimension": intent.dimension,
                "measure": intent.measure,
                "agg": _compare_agg(intent.aggregation),
                "top_n": intent.limit,
                "ascending": name == "bottom_n",
            },
            (
                "The question asks for a ranking, so values of the dimension are "
                "ranked by the measure with each one's share of the total."
            ),
            notes,
        )

    # ---- default: a single aggregate -------------------------------------
    agg = intent.aggregation
    if name == "count":
        if intent.column:
            column: str | None = intent.column
        else:
            # "How many rows / records?" — count a column with no gaps.
            column = next(
                (c.name for c in ctx.profile.columns if c.stats.null_count == 0),
                ctx.columns[0] if ctx.columns else None,
            )
            agg = Aggregation.COUNT
    else:
        column = intent.measure

    arguments: dict[str, Any] = {
        "column": column,
        "agg": _kpi_agg(agg),
        "filters": intent.filters,
    }
    rationale = (
        "The question asks for a single number, so one aggregate is computed over "
        "the whole dataset (with any filters the question specified)."
    )
    if intent.per_entity:
        arguments["per"] = intent.per_entity
        rationale = (
            f"The question asks for {column} per {intent.per_entity}, so the total is "
            f"divided by the number of distinct {intent.per_entity} values — a ratio "
            "of totals, not an average of rows."
        )

    return ToolPlan(
        "calculate_kpi",
        arguments,
        rationale,
        notes,
        suggestions=_suggestions(ctx) if not intent.is_confident else [],
    )


def _kpi_agg(agg: Aggregation) -> str:
    allowed = {"sum", "mean", "median", "min", "max", "count", "count_distinct"}
    return agg.value if agg.value in allowed else "sum"


def _compare_agg(agg: Aggregation) -> str:
    allowed = {"sum", "mean", "median", "count", "count_distinct"}
    return agg.value if agg.value in allowed else "sum"


def _forecast_agg(agg: Aggregation) -> str:
    allowed = {"sum", "mean", "count", "count_distinct"}
    return agg.value if agg.value in allowed else "sum"


def _suggestions(ctx: ToolContext) -> list[str]:
    """Concrete questions this dataset can actually answer."""
    measure = ctx.semantics.primary_measure
    dimension = next(
        iter(
            [
                d
                for d in ctx.semantics.dimensions
                if (ctx.profile.column(d) and 2 <= ctx.profile.column(d).stats.unique_count <= 50)  # type: ignore[union-attr]
            ]
        ),
        None,
    )
    out: list[str] = []
    if measure:
        out.append(f"What was total {measure}?")
        if dimension:
            out.append(f"Top 10 {dimension} by {measure}")
            out.append(f"{measure} by {dimension}")
        if ctx.semantics.primary_time_column:
            out.append(f"Show monthly {measure}")
            out.append(f"Forecast {measure} for the next 3 months")
            out.append(f"Find anomalies in {measure}")
    out.append("What columns are in this dataset?")
    return out[:6]
