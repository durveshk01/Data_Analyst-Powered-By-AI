"""Template-based answer composition.

Turns a tool result into the structured answer format InsightOS always uses:

    ANSWER · EVIDENCE · CHART · WHY · RECOMMENDATION

This runs with no LLM. When an AI provider *is* configured it rewrites the
prose of the ANSWER and WHY sections, but the numbers, the evidence table and
the chart always come from here — from the tool result — so an AI answer and a
deterministic answer can never disagree about a figure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.ai.tools import ToolResult
from app.engine.utils import compact_number, format_delta, safe_float, truncate
from app.nl.intent import SUPPORTED_QUESTION_SHAPES, ParsedIntent
from app.obs.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class AnalystAnswer:
    """The canonical shape of an answer, whatever produced it."""

    answer: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    evidence_caption: str = ""
    why: str | None = None
    recommendation: str | None = None
    chart: dict[str, Any] | None = None
    charts: list[dict[str, Any]] = field(default_factory=list)
    method: str = ""
    confidence: str = "high"
    engine: str = "deterministic"
    tool: str = ""
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    ok: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence": self.evidence,
            "evidence_caption": self.evidence_caption,
            "why": self.why,
            "recommendation": self.recommendation,
            "chart": self.chart,
            "charts": self.charts,
            "method": self.method,
            "confidence": self.confidence,
            "engine": self.engine,
            "tool": self.tool,
            "tool_arguments": self.tool_arguments,
            "notes": self.notes,
            "suggestions": self.suggestions,
            "ok": self.ok,
        }


def _unit_of(result: dict[str, Any], default: str = "number") -> str:
    return str(result.get("unit") or default)


# ============================================================== per-tool ====
def _narrate_kpi(result: dict[str, Any], intent: ParsedIntent) -> AnalystAnswer:
    value = result.get("value")
    column = result.get("column")
    agg = str(result.get("agg"))
    filters = result.get("filters_applied") or []
    scope = f" where {', '.join(filters)}" if filters else ""
    per = result.get("per")

    if per:
        unit = _unit_of(result)
        answer = (
            f"Average {column} per {per}{scope} is {result.get('formatted')} — "
            f"{compact_number(result.get('numerator'), unit=unit)} total {column} "
            f"across {result.get('per_count', 0):,} distinct {per} values."
            if value is not None
            else f"{column} per {per} cannot be computed{scope}."
        )
        evidence = [
            {"metric": f"Total {column}", "value": compact_number(result.get("numerator"), unit=unit)},
            {"metric": f"Distinct {per}", "value": f"{result.get('per_count', 0):,}"},
            {"metric": f"{column} per {per}", "value": result.get("formatted")},
        ]
        why = (
            f"A ratio of totals: all {column} divided by the number of distinct {per} "
            "values. This differs from the average of individual rows whenever one "
            f"{per} spans several rows."
        )
    elif agg in {"count", "count_distinct"}:
        exact = f"{int(value):,}" if value is not None else "—"
        answer = (
            f"There are {exact} distinct {column} values{scope}."
            if agg == "count_distinct"
            else f"There are {exact} rows with a {column} value{scope}."
        )
        evidence = [
            {"metric": "Distinct values" if agg == "count_distinct" else "Non-empty rows", "value": exact},
            {"metric": "Rows considered", "value": f"{result.get('rows_considered', 0):,}"},
        ]
        why = (
            "Each value is counted once, however many rows it appears on."
            if agg == "count_distinct"
            else "Rows where the column is empty are not counted."
        )
    else:
        agg_word = {
            "sum": "Total",
            "mean": "Average",
            "median": "Median",
            "min": "Minimum",
            "max": "Maximum",
        }.get(agg, "Value of")
        answer = (
            f"{agg_word} {column}{scope} is {result.get('formatted')}."
            if value is not None
            else f"{column} has no values to aggregate{scope}."
        )
        evidence = [
            {"metric": f"{agg_word} {column}", "value": result.get("formatted")},
            {"metric": "Rows included", "value": f"{result.get('rows_considered', 0):,}"},
            {"metric": "Rows in dataset", "value": f"{result.get('total_rows', 0):,}"},
        ]
        why = (
            f"Computed directly over {result.get('rows_considered', 0):,} rows of the "
            f"uploaded data. Rows with a missing {column} are excluded."
        )

    return AnalystAnswer(
        answer=answer,
        evidence=evidence,
        evidence_caption=str(result.get("basis", "")),
        method=str(result.get("basis", "")),
        why=why,
        confidence="high",
    )


def _narrate_top_contributors(
    result: dict[str, Any], intent: ParsedIntent
) -> AnalystAnswer:
    rows = result.get("rows") or []
    unit = _unit_of(result)
    dimension = result.get("dimension")
    measure = result.get("measure")
    ascending = bool(result.get("ascending"))
    additive = bool(result.get("additive", True))

    if not rows:
        return AnalystAnswer(
            answer=f"No {dimension} values have a {measure} to rank.",
            ok=False,
            confidence="low",
        )

    leader = rows[0]
    direction = "lowest" if ascending else "highest"
    agg_word = {"mean": "average ", "median": "median "}.get(str(result.get("agg")), "")
    answer = f"{dimension} '{leader['value']}' has the {direction} {agg_word}{measure} at {leader['formatted']}"
    if leader.get("share_pct") is not None and result.get("total") is not None:
        answer += (
            f" ({leader['share_pct']:.1f}% of the "
            f"{compact_number(result.get('total'), unit=unit)} total)."
        )
    else:
        answer += "."

    why_parts = [
        f"{measure} was aggregated ({result.get('agg')}) for each of "
        f"{result.get('level_count', 0):,} distinct {dimension} values and ranked."
    ]
    pareto = result.get("pareto_80_count")
    recommendation = None
    if pareto and result.get("level_count") and not ascending:
        level_count = int(result["level_count"])
        why_parts.append(
            f"{pareto} of {level_count} values account for 80% of the total."
        )
        if level_count >= 5 and pareto <= max(1, level_count // 5):
            recommendation = (
                f"{measure} is concentrated in a few {dimension} values — track them "
                "individually, since a change in any one moves the total."
            )
    if ascending:
        recommendation = (
            f"Compare '{leader['value']}' with the top {dimension} values on volume "
            "and price before deciding whether it is under-performing or simply small."
        )

    def row_view(row: dict[str, Any]) -> dict[str, Any]:
        view = {
            "rank": row["rank"],
            str(dimension): row["value"],
            str(measure): row["formatted"],
        }
        if additive and row.get("share_pct") is not None:
            view["share"] = f"{row['share_pct']:.1f}%"
        view["rows"] = f"{row.get('rows', 0):,}"
        return view

    return AnalystAnswer(
        answer=answer,
        evidence=[row_view(row) for row in rows[:10]],
        evidence_caption=(
            f"{'Bottom' if ascending else 'Top'} {min(len(rows), 10)} of "
            f"{result.get('level_count', len(rows)):,} {dimension} values by {measure}"
        ),
        method=f"{result.get('agg')} of {measure} grouped by {dimension}",
        why=" ".join(why_parts),
        recommendation=recommendation,
        chart=result.get("chart"),
        confidence="high",
    )


def _narrate_trend(result: dict[str, Any], intent: ParsedIntent) -> AnalystAnswer:
    unit = _unit_of(result)
    measure = result.get("value_column")
    direction = str(result.get("direction"))

    if direction == "insufficient_data":
        return AnalystAnswer(
            answer=str(result.get("summary")),
            method=f"Time series of {measure} at {result.get('grain')} grain",
            confidence="low",
            ok=False,
        )

    change = safe_float(result.get("change_pct"))
    answer = (
        f"{measure} {'increased' if direction == 'increasing' else 'decreased' if direction == 'decreasing' else 'stayed broadly flat'}"
        + (f" {abs(change):.1f}%" if change is not None and direction != "flat" else "")
        + f" from {result.get('first_period')} to {result.get('last_period')}, "
        f"ending at {compact_number(result.get('last_value'), unit=unit)}."
    )

    series = result.get("series") or []
    tail = series[-6:]

    evidence = [
        {"period": row["label"], str(measure): compact_number(row["value"], unit=unit)}
        for row in tail
    ]

    significant = bool(result.get("significant"))
    r2 = safe_float(result.get("r_squared"))
    why = (
        f"An ordinary-least-squares line was fitted to {result.get('points')} "
        f"{result.get('grain')} periods. Slope is "
        f"{compact_number(result.get('slope_per_period'), unit=unit)} per period "
        f"(R²={r2:.2f}, p={result.get('p_value'):.4f}). "
        + (
            "The trend is statistically significant."
            if significant
            else "The trend is NOT statistically significant — treat it as noise until "
            "more periods are available."
        )
    )
    if result.get("volatility") is not None:
        why += f" Period-to-period volatility is {result['volatility']:.2f} (σ/mean)."

    recommendation = None
    worst, best = result.get("worst_period"), result.get("best_period")
    if direction == "decreasing" and worst:
        recommendation = (
            f"Investigate {worst['label']} ({format_delta(worst['change_pct'])}), the "
            "sharpest single-period decline."
        )
    elif direction == "increasing" and best:
        recommendation = (
            f"Identify what drove {best['label']} ({format_delta(best['change_pct'])}) "
            "and whether it is repeatable."
        )

    return AnalystAnswer(
        answer=answer,
        evidence=evidence,
        evidence_caption=f"Last {len(tail)} of {result.get('points')} periods",
        method=f"OLS trend over a {result.get('grain')} series of {measure}",
        why=why,
        recommendation=recommendation,
        chart=result.get("chart"),
        confidence="high" if significant and (r2 or 0) > 0.4 else "moderate",
        notes=list(result.get("caveats") or []),
    )


def _narrate_compare(result: dict[str, Any], intent: ParsedIntent) -> AnalystAnswer:
    unit = _unit_of(result)
    baseline, comparison = result["baseline"], result["comparison"]
    change_pct = safe_float(result.get("change_pct"))
    measure = result.get("value_column")
    like_for_like = bool(result.get("like_for_like"))
    span = " over the same span" if like_for_like else ""

    answer = (
        f"{measure} was {comparison['formatted']} in {comparison['label']} versus "
        f"{baseline['formatted']} in {baseline['label']}{span} — "
        + (
            f"{'up' if (change_pct or 0) > 0 else 'down'} {abs(change_pct):.1f}%."
            if change_pct is not None
            else "a change that cannot be expressed as a percentage (the baseline is zero)."
        )
    )

    evidence = [
        {"period": baseline["label"] + (" (same span)" if like_for_like else ""), str(measure): baseline["formatted"]},
        {"period": comparison["label"] + (" (same span)" if like_for_like else ""), str(measure): comparison["formatted"]},
        {
            "period": "Change",
            str(measure): (
                f"{compact_number(result.get('change_abs'), unit=unit)} "
                f"({format_delta(change_pct)})"
            ),
        },
    ]
    if like_for_like:
        evidence.append(
            {
                "period": f"{baseline['label']} (full period)",
                str(measure): compact_number(baseline.get("full_period_value"), unit=unit),
            }
        )

    why = (
        "Both periods were cut to the same span because one of them is only partly "
        "covered by the data; comparing a part-period with a whole one would show a "
        "change that is really just missing days."
        if like_for_like
        else "Both figures come from the same aggregation over the same column, so "
        "the difference is a like-for-like comparison."
    )

    return AnalystAnswer(
        answer=answer,
        evidence=evidence,
        evidence_caption=f"{result.get('agg')} of {measure} by {result.get('grain')}",
        method=(
            f"Both periods aggregated with {result.get('agg')} at "
            f"{result.get('grain')} grain from column '{result.get('time_column')}'"
        ),
        why=why,
        confidence="high",
        notes=list(result.get("notes") or []),
    )


def _narrate_explain_change(
    result: dict[str, Any], intent: ParsedIntent
) -> AnalystAnswer:
    overall = result["overall"]
    unit = _unit_of(overall)
    measure = overall.get("value_column")
    dimension = result.get("dimension")
    drivers = result.get("drivers") or []
    change_pct = safe_float(overall.get("change_pct"))

    if not drivers:
        return _narrate_compare(overall, intent)

    primary = drivers[0]
    falling = (change_pct or 0) < 0

    # Check the question's premise against the data.
    asked = (intent.original or "").lower()
    asked_fall = bool(re.search(r"\b(fall|fell|drop|declin|decreas|down|dip|lower)", asked))
    asked_rise = bool(re.search(r"\b(rise|rose|grow|grew|increas|up|jump|spike|higher)", asked))
    premise_note = None
    if (asked_fall and not falling) or (asked_rise and falling and not asked_fall):
        premise_note = (
            f"The question assumes {measure} {'fell' if asked_fall else 'rose'}, but the "
            f"data shows it {'rose' if not falling else 'fell'} between these periods."
        )

    driver_word = "decline" if falling else "increase"
    answer = (
        f"{measure} {'fell' if falling else 'rose'} "
        + (f"{abs(change_pct):.1f}% " if change_pct is not None else "")
        + f"from {overall['baseline']['formatted']} in {overall['baseline']['label']} "
        f"to {overall['comparison']['formatted']} in {overall['comparison']['label']}. "
        f"Primary driver: {dimension} '{primary['value']}' "
        f"({compact_number(primary['change_abs'], unit=unit)}, "
        f"{format_delta(primary.get('change_pct'))})."
    )
    if premise_note:
        answer = premise_note + " " + answer

    contribution = safe_float(primary.get("contribution_pct"))
    check = result.get("decomposition_check", {})
    why = (
        f"The change of {compact_number(overall.get('change_abs'), unit=unit)} was "
        f"split across {result.get('driver_count', 0)} '{dimension}' values. "
        f"'{primary['value']}' moved {compact_number(primary['change_abs'], unit=unit)}"
        + (
            f", {abs(contribution):.0f}% of the net {driver_word}."
            if contribution is not None and check.get("additive", True)
            else "."
        )
    )
    if abs(contribution or 0) > 100:
        why += (
            " A share above 100% means other values moved the opposite way and "
            "partly offset it."
        )
    if check.get("exact"):
        why += " The contributions sum exactly to the headline change."
    elif not check.get("additive", True):
        why += (
            f" Because the measure is a {overall.get('agg')}, contributions are "
            "indicative and do not sum to the headline change."
        )

    scores = result.get("dimension_scores") or []
    if result.get("dimension_selected_automatically") and len(scores) > 1:
        others = ", ".join(s["dimension"] for s in scores[1:4])
        why += (
            f" '{dimension}' was chosen because the change is more concentrated there "
            f"than in {others}."
        )

    evidence = [
        {
            str(dimension): d["value"],
            overall["baseline"]["label"]: compact_number(d["before"], unit=unit),
            overall["comparison"]["label"]: compact_number(d["after"], unit=unit),
            "change": compact_number(d["change_abs"], unit=unit),
            "share of change": (
                f"{d['contribution_pct']:.0f}%"
                if d.get("contribution_pct") is not None
                else "—"
            ),
        }
        for d in drivers[:8]
    ]

    biggest_decline = result.get("biggest_decline")
    recommendation = None
    if falling and biggest_decline:
        recommendation = (
            f"Start with {dimension} = '{biggest_decline['value']}': it lost "
            f"{compact_number(abs(biggest_decline['change_abs'] or 0), unit=unit)} "
            f"({format_delta(biggest_decline.get('change_pct'))}) between the two "
            "periods, the single largest contributor to the decline."
        )
    elif drivers:
        recommendation = (
            f"'{primary['value']}' drove most of the movement — check whether the "
            "cause is repeatable before extrapolating."
        )

    return AnalystAnswer(
        answer=answer,
        evidence=evidence,
        evidence_caption=(
            f"Change in {measure} by {dimension}: {overall['baseline']['label']} → "
            f"{overall['comparison']['label']}"
        ),
        method=(
            f"Exact period-over-period decomposition of {measure} across {dimension}"
        ),
        why=why,
        recommendation=recommendation,
        chart=result.get("chart"),
        confidence="high",
        notes=list(overall.get("notes") or []),
    )


def _narrate_anomalies(result: dict[str, Any], intent: ParsedIntent) -> AnalystAnswer:
    anomalies = result.get("anomalies") or []
    unit = _unit_of(result)
    column = result.get("column")

    if not anomalies:
        return AnalystAnswer(
            answer=(
                f"No anomalies were detected in {column} using the "
                f"{result.get('method')} method."
            ),
            evidence=[
                {"metric": "Method", "value": str(result.get("method"))},
                {"metric": "Periods / rows tested", "value": f"{len(result.get('series') or []) or '—'}"},
            ],
            method=str(result.get("method_reason", "")),
            why=str(result.get("method_reason", "")),
            chart=result.get("chart"),
            confidence="high",
        )

    worst = anomalies[0]
    answer = (
        f"{len(anomalies)} anomal{'y' if len(anomalies) == 1 else 'ies'} detected in "
        f"{column}. The most extreme is {worst['label']}: "
        f"{compact_number(worst.get('observed'), unit=unit)}"
        + (
            f" against an expected {compact_number(worst.get('expected'), unit=unit)} "
            f"({format_delta(worst.get('deviation_pct'))})."
            if worst.get("expected") is not None
            else f" (anomaly score {worst.get('score')})."
        )
    )

    return AnalystAnswer(
        answer=answer,
        evidence=[
            {
                "when": a["label"],
                "observed": compact_number(a.get("observed"), unit=unit),
                "expected": compact_number(a.get("expected"), unit=unit),
                "deviation": format_delta(a.get("deviation_pct")),
                "score": a.get("score"),
            }
            for a in anomalies[:10]
        ],
        evidence_caption=(
            f"{min(len(anomalies), 10)} of {result.get('count')} flagged points "
            f"({result.get('method')} method)"
        ),
        method=str(result.get("method_reason", "")),
        why=(
            str(result.get("method_reason", ""))
            + " A statistical anomaly is unusual relative to its own baseline — it may "
            "be an error, or a real event such as a promotion or an outage."
        ),
        recommendation=(
            f"Check {worst['label']} against operational records before treating it as "
            "a data error."
        ),
        chart=result.get("chart"),
        confidence="moderate",
    )


def _narrate_forecast(result: dict[str, Any], intent: ParsedIntent) -> AnalystAnswer:
    points = result.get("points") or []
    unit = "currency" if "revenue" in str(result.get("target_column", "")).lower() else "number"
    from app.engine.utils import guess_unit

    unit = guess_unit(str(result.get("target_column", "")))
    target = result.get("target_column")
    grain = result.get("grain")

    if not points:
        return AnalystAnswer(
            answer=f"No forecast could be produced for {target}.",
            ok=False,
            confidence="low",
        )

    first, last = points[0], points[-1]
    total = sum(p["value"] for p in points)
    answer = (
        f"{target} is projected at {compact_number(first['value'], unit=unit)} in "
        f"{first['period']}, reaching {compact_number(last['value'], unit=unit)} by "
        f"{last['period']} — {compact_number(total, unit=unit)} over the next "
        f"{len(points)} {grain}(s)."
    )

    metrics_payload = result.get("metrics") or {}
    beats_naive = bool(metrics_payload.get("beats_naive"))
    why = (
        f"{result.get('model_reason')} Validation error on held-out periods: "
        f"sMAPE {metrics_payload.get('smape')}%, MAE "
        f"{compact_number(metrics_payload.get('mae'), unit=unit)}."
    )
    if not beats_naive:
        why += (
            " This model did not beat the naive 'next period equals this period' "
            "baseline, so confidence is low."
        )

    caveats = result.get("caveats") or []
    scoreboard = [
        c for c in (result.get("candidates") or []) if c.get("smape") is not None
    ][:6]

    return AnalystAnswer(
        answer=answer,
        evidence=[
            {
                "period": p["period"],
                "forecast": compact_number(p["value"], unit=unit),
                "range (95%)": (
                    f"{compact_number(p.get('lower'), unit=unit)} – "
                    f"{compact_number(p.get('upper'), unit=unit)}"
                ),
            }
            for p in points
        ],
        evidence_caption=(
            f"Model: {result.get('model_name')} · selected from "
            f"{len(result.get('candidates') or [])} candidates"
        ),
        method=f"{result.get('model_name')} fitted on {metrics_payload.get('training_periods')} periods",
        why=why,
        recommendation=(
            "Treat the 95% band, not the central line, as the planning range."
            if beats_naive
            else "Collect more history before planning against this projection."
        ),
        chart=result.get("chart"),
        confidence="moderate" if beats_naive else "low",
        notes=[*caveats]
        + (
            [
                "Model scoreboard: "
                + "; ".join(
                    f"{c['model']} sMAPE {c['smape']}%" for c in scoreboard
                )
            ]
            if scoreboard
            else []
        ),
    )


def _narrate_correlation(result: dict[str, Any], intent: ParsedIntent) -> AnalystAnswer:
    if "top_pairs" in result:
        pairs = result.get("top_pairs") or []
        if not pairs:
            return AnalystAnswer(
                answer=(
                    "No meaningful correlations were found between the numeric columns "
                    "in this dataset."
                ),
                method=f"{result.get('method')} correlation matrix",
                confidence="high",
            )
        top = pairs[0]
        return AnalystAnswer(
            answer=(
                f"The strongest relationship is between '{top['x']}' and '{top['y']}' "
                f"(r = {top['r']:.2f}, {top['strength']} {top['direction']})."
            ),
            evidence=[
                {
                    "pair": f"{p['x']} ↔ {p['y']}",
                    "r": f"{p['r']:.3f}",
                    "strength": p["strength"],
                    "rows": f"{p['n']:,}",
                }
                for p in pairs[:8]
            ],
            evidence_caption=(
                f"{len(pairs)} column pairs ranked by |r| "
                f"({result.get('method')} coefficient)"
            ),
            method=f"{result.get('method')} correlation over complete pairs",
            why=str(result.get("note", "")),
            chart=result.get("chart"),
            confidence="high",
        )

    r = safe_float(result.get("r"))
    return AnalystAnswer(
        answer=(
            f"'{result.get('x')}' and '{result.get('y')}' have a "
            f"{result.get('strength')} {result.get('direction')} correlation "
            f"(r = {r:.3f})."
            if r is not None
            else "The correlation could not be computed."
        ),
        evidence=[
            {"metric": "Coefficient (r)", "value": f"{r:.4f}" if r is not None else "—"},
            {"metric": "R² (variance explained)", "value": f"{(result.get('r_squared') or 0) * 100:.1f}%"},
            {"metric": "p-value", "value": f"{result.get('p_value'):.5f}" if result.get("p_value") is not None else "—"},
            {"metric": "Rows with both values", "value": f"{result.get('n', 0):,}"},
            {"metric": "95% CI for r", "value": str(result.get("ci_95", "—"))},
        ],
        evidence_caption=f"{result.get('method')} correlation",
        method=f"{result.get('method')} correlation on {result.get('n', 0):,} complete pairs",
        why=str(result.get("interpretation", "")),
        recommendation=(
            "Association is not causation. Before acting, test whether changing one "
            "variable actually moves the other."
        ),
        chart=result.get("chart"),
        confidence="high" if result.get("significant") else "low",
    )


def _narrate_distribution(result: dict[str, Any], intent: ParsedIntent) -> AnalystAnswer:
    kind = result.get("kind")
    column = result.get("column")

    if kind == "numeric":
        from app.engine.utils import guess_unit

        unit = guess_unit(str(column))
        return AnalystAnswer(
            answer=(
                f"{column} ranges from {compact_number(result.get('min'), unit=unit)} to "
                f"{compact_number(result.get('max'), unit=unit)}, with a median of "
                f"{compact_number(result.get('median'), unit=unit)} and a mean of "
                f"{compact_number(result.get('mean'), unit=unit)}."
            ),
            evidence=[
                {"statistic": "Rows with a value", "value": f"{result.get('count', 0):,}"},
                {"statistic": "Minimum", "value": compact_number(result.get("min"), unit=unit)},
                {"statistic": "25th percentile", "value": compact_number((result.get("percentiles") or {}).get("p25"), unit=unit)},
                {"statistic": "Median", "value": compact_number(result.get("median"), unit=unit)},
                {"statistic": "Mean", "value": compact_number(result.get("mean"), unit=unit)},
                {"statistic": "75th percentile", "value": compact_number((result.get("percentiles") or {}).get("p75"), unit=unit)},
                {"statistic": "95th percentile", "value": compact_number((result.get("percentiles") or {}).get("p95"), unit=unit)},
                {"statistic": "Maximum", "value": compact_number(result.get("max"), unit=unit)},
                {"statistic": "Std deviation", "value": compact_number(result.get("std"), unit=unit)},
            ],
            evidence_caption=f"Descriptive statistics for {column}",
            method="Percentile and moment statistics over all non-missing values",
            why=(
                f"The distribution is {result.get('distribution_shape', 'not characterised')}."
                + (
                    f" Because the mean and median differ by "
                    f"{abs((result.get('mean') or 0) - (result.get('median') or 0)) / max(abs(result.get('median') or 1), 1e-9) * 100:.0f}%, "
                    "the median is the better description of a typical value."
                    if result.get("median")
                    and result.get("mean")
                    and abs((result["mean"] - result["median"]) / max(abs(result["median"]), 1e-9)) > 0.15
                    else ""
                )
            ),
            chart=result.get("chart"),
            confidence="high",
        )

    if kind == "categorical":
        top = result.get("top_values") or []
        return AnalystAnswer(
            answer=(
                f"{column} has {result.get('unique', 0):,} distinct values; the most "
                f"common is '{result.get('mode')}' ({result.get('mode_share_pct')}% of rows)."
            ),
            evidence=[
                {"value": v["value"], "rows": f"{v['count']:,}", "share": f"{v['share_pct']}%"}
                for v in top[:10]
            ],
            evidence_caption=f"Top {min(len(top), 10)} values of {column}",
            method="Value counts over all non-missing values",
            confidence="high",
        )

    return AnalystAnswer(
        answer=(
            f"{column} spans {result.get('min')} to {result.get('max')} "
            f"({result.get('span_days')} days)."
            if kind == "datetime"
            else f"{column}: {result.get('note', 'no distribution available')}."
        ),
        method="Column profile",
        confidence="high",
    )


def _narrate_schema(result: dict[str, Any], intent: ParsedIntent) -> AnalystAnswer:
    columns = result.get("columns") or []
    concepts = result.get("detected_concepts") or {}
    answer = (
        f"This dataset has {result.get('row_count', 0):,} rows and "
        f"{result.get('column_count', 0)} columns"
        + (
            f", and looks like {result['domain_guess']} data."
            if result.get("domain_guess")
            else "."
        )
    )
    return AnalystAnswer(
        answer=answer,
        evidence=[
            {
                "column": c["name"],
                "type": c["semantic_type"],
                "role": c["role"],
                "missing": f"{c['missing_pct']}%",
                "distinct": f"{c['distinct']:,}",
            }
            for c in columns[:40]
        ],
        evidence_caption=f"All {len(columns)} columns with inferred type and role",
        method="Type inference plus name and value heuristics",
        why=(
            "Measures are columns you aggregate, dimensions are columns you slice by, "
            "time columns provide the date axis and identifiers label rows. "
            + (
                "Recognised business concepts: "
                + ", ".join(f"{k} → {v}" for k, v in list(concepts.items())[:8])
                + "."
                if concepts
                else ""
            )
        ),
        confidence="high",
    )


def _narrate_quality(result: dict[str, Any], intent: ParsedIntent) -> AnalystAnswer:
    issues = result.get("issues") or []
    return AnalystAnswer(
        answer=(
            f"Data quality is {result.get('score')}/100 (grade "
            f"{result.get('grade')}) with {len(issues)} issue(s) detected."
        ),
        evidence=[
            {
                "issue": i["title"],
                "severity": i["severity"],
                "rows affected": f"{i['affected_rows']:,}",
                "fixable": "yes" if i.get("fix") else "review",
            }
            for i in issues[:12]
        ],
        evidence_caption=f"{min(len(issues), 12)} of {len(issues)} detected issues",
        method="Nine rule families across completeness, uniqueness, validity, consistency and accuracy",
        why=(
            "Sub-scores: "
            + ", ".join(
                f"{k} {v}" for k, v in (result.get("dimension_scores") or {}).items()
            )
            + "."
        ),
        recommendation=(
            issues[0].get("suggestion")
            if issues and issues[0].get("suggestion")
            else "Review the flagged issues before relying on aggregate figures."
        ),
        confidence="high",
    )


def _narrate_sample(result: dict[str, Any], intent: ParsedIntent) -> AnalystAnswer:
    rows = result.get("rows") or []
    return AnalystAnswer(
        answer=f"Here are {len(rows)} of {result.get('total_rows', 0):,} rows.",
        evidence=rows,
        evidence_caption=f"{len(rows)} sample rows",
        method="Raw rows from the head of the dataset",
        why=str(result.get("note", "")),
        confidence="high",
    )


def _narrate_query(result: dict[str, Any], intent: ParsedIntent) -> AnalystAnswer:
    rows = result.get("rows") or []
    columns = result.get("columns") or []
    totals = result.get("totals") or {}
    answer = (
        f"The query returned {result.get('row_count', 0):,} group(s) across "
        f"{len(columns)} column(s)."
        if rows
        else "The query matched no rows."
    )
    if totals:
        parts = [
            f"{k} = {compact_number(v)}"
            for k, v in totals.items()
            if not k.startswith("_") and isinstance(v, (int, float))
        ]
        if parts:
            answer += " Totals: " + ", ".join(parts[:4]) + "."
    return AnalystAnswer(
        answer=answer,
        evidence=rows[:20],
        evidence_caption=f"{min(len(rows), 20)} of {result.get('row_count', 0):,} rows",
        method="Validated query specification executed over the full dataset",
        why=(
            "Filters applied: " + "; ".join(result.get("applied_filters") or ["none"]) + "."
        ),
        confidence="high",
    )


def _narrate_sql(result: dict[str, Any], intent: ParsedIntent) -> AnalystAnswer:
    rows = result.get("rows") or []
    return AnalystAnswer(
        answer=f"The query returned {result.get('row_count', 0):,} row(s).",
        evidence=rows[:20],
        evidence_caption=f"{min(len(rows), 20)} of {result.get('row_count', 0):,} rows",
        method="Read-only SELECT validated by the SQL guard and executed in DuckDB",
        why=(
            f"Executed: {truncate(str(result.get('executed_sql')), 300)}. "
            f"Row limit {result.get('row_limit')} enforced."
        ),
        confidence="high",
        notes=(
            ["Results were truncated by the row limit."]
            if result.get("truncated")
            else []
        ),
    )


def _narrate_chart(result: dict[str, Any], intent: ParsedIntent) -> AnalystAnswer:
    chart = result.get("chart") or {}
    return AnalystAnswer(
        answer=str(chart.get("title") or "Chart generated."),
        evidence=(result.get("rows") or [])[:10],
        evidence_caption=str(chart.get("subtitle") or ""),
        method=str(chart.get("reason") or "Chart type chosen from the data's structure"),
        why=str(chart.get("reason") or ""),
        chart=chart or None,
        confidence="high",
    )


def _narrate_column_values(
    result: dict[str, Any], intent: ParsedIntent
) -> AnalystAnswer:
    values = result.get("values") or []
    return AnalystAnswer(
        answer=(
            f"'{result.get('column')}' has {result.get('distinct_total', 0):,} distinct "
            f"values."
        ),
        evidence=[
            {"value": v["value"], "rows": f"{v['count']:,}", "share": f"{v['share']}%"}
            for v in values[:25]
        ],
        evidence_caption=f"Top {min(len(values), 25)} values by frequency",
        method="Value counts over all non-missing values",
        confidence="high",
    )


_NARRATORS: dict[str, Any] = {
    "calculate_kpi": _narrate_kpi,
    "top_contributors": _narrate_top_contributors,
    "analyse_trend": _narrate_trend,
    "compare_periods": _narrate_compare,
    "explain_change": _narrate_explain_change,
    "detect_anomalies": _narrate_anomalies,
    "generate_forecast": _narrate_forecast,
    "calculate_correlation": _narrate_correlation,
    "analyse_distribution": _narrate_distribution,
    "get_column_statistics": _narrate_distribution,
    "inspect_schema": _narrate_schema,
    "data_quality_report": _narrate_quality,
    "sample_rows": _narrate_sample,
    "query_data": _narrate_query,
    "run_sql": _narrate_sql,
    "generate_chart": _narrate_chart,
    "list_column_values": _narrate_column_values,
}


def narrate(
    tool_result: ToolResult,
    intent: ParsedIntent | None = None,
    *,
    notes: list[str] | None = None,
    suggestions: list[str] | None = None,
) -> AnalystAnswer:
    """Compose the structured answer for a completed tool call."""
    parsed = intent or ParsedIntent(intent="unknown", confidence=0.0)

    if not tool_result.ok or tool_result.result is None:
        return AnalystAnswer(
            answer=(
                tool_result.error
                or "That analysis could not be completed on this dataset."
            ),
            ok=False,
            confidence="low",
            engine="deterministic",
            tool=tool_result.name,
            tool_arguments=tool_result.arguments,
            method="",
            notes=list(notes or []),
            suggestions=list(suggestions or list(SUPPORTED_QUESTION_SHAPES[:6])),
        )

    narrator = _NARRATORS.get(tool_result.name)
    if narrator is None:
        answer = AnalystAnswer(
            answer=f"{tool_result.name} completed.",
            evidence=[],
            method=tool_result.name,
        )
    else:
        answer = narrator(tool_result.result, parsed)

    answer.engine = "deterministic"
    answer.tool = tool_result.name
    answer.tool_arguments = tool_result.arguments
    answer.notes = [*(notes or []), *answer.notes]
    answer.suggestions = list(suggestions or answer.suggestions)

    # Carry the intent's unresolved words forward so a misread question is
    # visible to the user rather than silently producing a confident wrong answer.
    if parsed.unresolved_terms and parsed.confidence < 0.85:
        answer.notes.append(
            "These words were not matched to anything in the dataset: "
            + ", ".join(parsed.unresolved_terms)
            + "."
        )
    if parsed.intent != "unknown" and not parsed.is_confident:
        answer.confidence = "low"
        answer.notes.append(
            "The question was only partially understood, so check the method below "
            "matches what you asked."
        )
    return answer
