"""The automatic insight engine.

Every insight is produced by a generator that (a) runs a real calculation,
(b) attaches the numbers it used as `Evidence`, and (c) only then writes a
sentence. There is no path by which a sentence can exist without the
arithmetic behind it, and none of this uses an LLM — the optional AI layer can
*rephrase* an insight, never invent one.

Each generator is independently guarded: if its preconditions are not met (no
time column, too few periods, no numeric measure) it returns nothing rather
than lowering its standard of evidence.

Insights are ranked by `importance`, a 0-100 score combining effect size,
statistical confidence and business relevance, so the five shown first are the
five worth reading.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from app.engine import charts as charts_mod
from app.engine.semantics import choose_reporting_grain
from app.engine.types import (
    Aggregation,
    ChartAnnotation,
    DatasetProfile,
    Evidence,
    InsightModel,
    QualityReport,
    SemanticModel,
    Severity,
    TimeGrain,
)
from app.engine.utils import compact_number, format_delta, guess_unit, safe_float
from app.obs.logging import get_logger

log = get_logger(__name__)

Generator = Callable[["InsightContext"], list[InsightModel]]


class InsightContext:
    """Everything a generator needs, computed once and shared."""

    def __init__(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        semantics: SemanticModel,
        quality: QualityReport | None = None,
    ) -> None:
        self.df = df
        self.profile = profile
        self.semantics = semantics
        self.quality = quality
        self.measure = semantics.primary_measure
        self.time_column = semantics.primary_time_column
        self.unit = guess_unit(self.measure) if self.measure else "number"

        self.grain: TimeGrain | None = None
        if self.time_column:
            time_profile = profile.column(self.time_column)
            self.grain = choose_reporting_grain(
                time_profile.stats.span_days if time_profile else None,
                time_profile.stats.inferred_grain if time_profile else None,
            )

        self._series: pd.DataFrame | None = None
        self._series_loaded = False

    @property
    def series(self) -> pd.DataFrame | None:
        """The primary measure as a time series (computed at most once)."""
        if self._series_loaded:
            return self._series
        self._series_loaded = True
        if not (self.measure and self.time_column and self.grain):
            return None
        try:
            from app.engine.trends import build_series

            self._series = build_series(
                self.df,
                time_column=self.time_column,
                value_column=self.measure,
                grain=self.grain,
                agg=Aggregation.SUM,
            )
        except Exception as exc:
            log.info("insight_series_unavailable", extra={"error": str(exc)[:160]})
            self._series = None
        return self._series

    def ranked_dimensions(self) -> list[str]:
        return charts_mod._rank_dimensions(self.profile, self.semantics)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _importance(
    *,
    effect: float,
    confidence: float = 0.6,
    relevance: float = 0.6,
) -> float:
    """0-100 score. `effect` is a 0-1 normalised magnitude."""
    effect = max(0.0, min(effect, 1.0))
    confidence = max(0.0, min(confidence, 1.0))
    relevance = max(0.0, min(relevance, 1.0))
    return round((effect * 0.5 + confidence * 0.25 + relevance * 0.25) * 100, 1)


# ============================================================= generators ====
def _trend_insight(ctx: InsightContext) -> list[InsightModel]:
    series = ctx.series
    if series is None or len(series) < 4 or not (ctx.measure and ctx.time_column and ctx.grain):
        return []

    from app.engine.trends import trend_analysis

    try:
        trend = trend_analysis(
            ctx.df,
            time_column=ctx.time_column,
            value_column=ctx.measure,
            grain=ctx.grain,
            agg=Aggregation.SUM,
        )
    except Exception:
        return []

    direction = trend["direction"]
    if direction == "insufficient_data":
        return []

    change_pct = trend.get("change_pct")
    if change_pct is None:
        return []

    if direction == "flat":
        severity = Severity.INFO
        title = f"{ctx.measure} is broadly flat over the period"
    elif direction == "increasing":
        severity = Severity.POSITIVE
        title = f"{ctx.measure} increased {abs(change_pct):.1f}%"
    else:
        severity = Severity.WARNING
        title = f"{ctx.measure} decreased {abs(change_pct):.1f}%"

    chart = charts_mod.time_series_chart(
        series, value_column=ctx.measure, grain=ctx.grain
    )

    return [
        InsightModel(
            id=_new_id(),
            kind="trend",
            title=title,
            explanation=trend["summary"],
            severity=severity,
            importance=_importance(
                effect=min(abs(change_pct) / 40.0, 1.0),
                confidence=float(trend.get("r_squared") or 0.0),
                relevance=0.95,
            ),
            metric={
                "column": ctx.measure,
                "unit": ctx.unit,
                "change_pct": change_pct,
                "first_value": trend["first_value"],
                "last_value": trend["last_value"],
                "total": trend["total"],
            },
            evidence=Evidence(
                method=(
                    f"Ordinary least squares on {trend['points']} {ctx.grain.label} "
                    "periods of summed values"
                ),
                rows=[
                    {
                        "period": trend["first_period"],
                        ctx.measure: trend["first_value"],
                    },
                    {
                        "period": trend["last_period"],
                        ctx.measure: trend["last_value"],
                    },
                ],
                values={
                    "slope_per_period": trend["slope_per_period"],
                    "r_squared": trend["r_squared"],
                    "p_value": trend["p_value"],
                    "cagr_pct": trend.get("cagr_pct"),
                    "volatility": trend.get("volatility"),
                },
                sample_size=trend["points"],
                confidence=(
                    "high"
                    if (trend.get("r_squared") or 0) >= 0.6 and trend.get("significant")
                    else "moderate"
                    if trend.get("significant")
                    else "low"
                ),
                caveats=(
                    []
                    if trend.get("significant")
                    else [
                        "The trend is not statistically significant (p ≥ 0.05); it may "
                        "be short-run noise."
                    ]
                ),
            ),
            chart=chart,
            recommendation=(
                f"Review what changed around {trend['worst_period']['label']}, the "
                f"weakest period ({format_delta(trend['worst_period']['change_pct'])})."
                if direction == "decreasing" and trend.get("worst_period")
                else (
                    f"Identify what drove {trend['best_period']['label']} "
                    f"({format_delta(trend['best_period']['change_pct'])}) and whether "
                    "it is repeatable."
                    if direction == "increasing" and trend.get("best_period")
                    else None
                )
            ),
            created_at=_now(),
        )
    ]


def _period_change_insight(ctx: InsightContext) -> list[InsightModel]:
    """Latest period vs the previous one, decomposed by dimension."""
    series = ctx.series
    if series is None or len(series) < 3 or not (ctx.measure and ctx.time_column and ctx.grain):
        return []

    from app.engine.trends import explain_change

    dimensions = ctx.ranked_dimensions()
    if not dimensions:
        return []

    try:
        decomposition = explain_change(
            ctx.df,
            time_column=ctx.time_column,
            value_column=ctx.measure,
            dimension=None,
            candidate_dimensions=dimensions[:6],
            grain=ctx.grain,
            agg=Aggregation.SUM,
        )
    except Exception:
        return []

    overall = decomposition["overall"]
    change_pct = overall.get("change_pct")
    if change_pct is None or abs(change_pct) < 2.0:
        return []

    drivers = decomposition["drivers"]
    if not drivers:
        return []

    falling = change_pct < 0
    primary = drivers[0]
    dimension = decomposition["dimension"]

    chart = charts_mod.category_chart(
        [
            {
                "value": d["value"],
                "metric": d["change_abs"],
                "share_pct": d["contribution_pct"],
            }
            for d in drivers
        ],
        dimension=dimension,
        measure=f"Change in {ctx.measure}",
        total_levels=decomposition["driver_count"],
        chart_id="period_change_drivers",
        title=(
            f"Change in {ctx.measure} by {dimension}: "
            f"{overall['baseline']['label']} → {overall['comparison']['label']}"
        ),
    )

    return [
        InsightModel(
            id=_new_id(),
            kind="period_change",
            title=(
                f"{ctx.measure} {'fell' if falling else 'rose'} "
                f"{abs(change_pct):.1f}% in {overall['comparison']['label']}"
            ),
            explanation=decomposition["summary"],
            severity=Severity.WARNING if falling else Severity.POSITIVE,
            importance=_importance(
                # A 15% move in the latest period is as big as news gets.
                effect=min(abs(change_pct) / 15.0, 1.0),
                confidence=0.95,  # exact arithmetic, not an estimate
                relevance=1.0,
            ),
            metric={
                "column": ctx.measure,
                "unit": ctx.unit,
                "baseline": overall["baseline"],
                "comparison": overall["comparison"],
                "change_abs": overall["change_abs"],
                "change_pct": change_pct,
            },
            evidence=Evidence(
                method=(
                    f"Exact decomposition of the {ctx.grain.label}-over-{ctx.grain.label} "
                    f"change across {decomposition['driver_count']} '{dimension}' values"
                ),
                rows=[
                    {
                        dimension: d["value"],
                        overall["baseline"]["label"]: d["before"],
                        overall["comparison"]["label"]: d["after"],
                        "change": d["change_abs"],
                        "contribution_%": d["contribution_pct"],
                    }
                    for d in drivers[:6]
                ],
                values={
                    "total_change": overall["change_abs"],
                    "sum_of_contributions": decomposition["decomposition_check"][
                        "sum_of_contributions"
                    ],
                    "decomposition_is_exact": decomposition["decomposition_check"]["exact"],
                },
                sample_size=decomposition["driver_count"],
                confidence="high",
                caveats=(
                    []
                    if decomposition["decomposition_check"]["exact"]
                    else [
                        "Contributions do not sum exactly to the headline change; "
                        "some rows have a missing dimension value."
                    ]
                ),
            ),
            chart=chart,
            recommendation=(
                f"Start with {dimension} = '{primary['value']}' "
                f"({compact_number(primary['change_abs'], unit=ctx.unit)}, "
                f"{format_delta(primary['change_pct'])}) — it is the largest single "
                f"driver of the {'decline' if falling else 'increase'}."
            ),
            created_at=_now(),
        )
    ]


def _concentration_insight(ctx: InsightContext) -> list[InsightModel]:
    if not ctx.measure:
        return []
    from app.engine.trends import top_contributors

    out: list[InsightModel] = []
    for dimension in ctx.ranked_dimensions()[:3]:
        try:
            contributions = top_contributors(
                ctx.df, dimension=dimension, measure=ctx.measure, top_n=10
            )
        except Exception:
            continue

        rows = contributions["rows"]
        if not rows or contributions["level_count"] < 3:
            continue

        leader = rows[0]
        share = safe_float(leader["share_pct"]) or 0.0
        # Only interesting when concentration is genuinely uneven.
        even_share = 100.0 / contributions["level_count"]
        if share < max(2 * even_share, 20.0):
            continue

        pareto = contributions["pareto_80_count"]
        explanation = contributions["summary"]
        if pareto and pareto <= max(2, contributions["level_count"] // 5):
            explanation += (
                f" That is heavy concentration: losing the top {pareto} would remove "
                "most of the total."
            )

        out.append(
            InsightModel(
                id=_new_id(),
                kind="concentration",
                title=(
                    f"{dimension} '{leader['value']}' accounts for "
                    f"{share:.1f}% of {ctx.measure}"
                ),
                explanation=explanation,
                severity=Severity.WARNING if share >= 50 else Severity.INFO,
                importance=_importance(
                    effect=min(share / 80.0, 1.0),
                    confidence=0.95,
                    relevance=0.65,
                ),
                metric={
                    "column": ctx.measure,
                    "unit": contributions["unit"],
                    "dimension": dimension,
                    "top_value": leader["value"],
                    "share_pct": share,
                    "total": contributions["total"],
                    "pareto_80_count": pareto,
                },
                evidence=Evidence(
                    method=(
                        f"Sum of '{ctx.measure}' grouped by '{dimension}' over "
                        f"{contributions['level_count']} values"
                    ),
                    rows=[
                        {
                            dimension: r["value"],
                            ctx.measure: r["metric"],
                            "share_%": r["share_pct"],
                        }
                        for r in rows[:6]
                    ],
                    values={
                        "grand_total": contributions["total"],
                        "top_3_share_pct": contributions["concentration"],
                        "level_count": contributions["level_count"],
                    },
                    sample_size=contributions["level_count"],
                    confidence="high",
                ),
                chart=charts_mod.category_chart(
                    rows,
                    dimension=dimension,
                    measure=ctx.measure,
                    total_levels=contributions["level_count"],
                ),
                recommendation=(
                    f"Revenue concentration in '{leader['value']}' is a dependency "
                    "risk — check whether it is diversifying over time."
                    if share >= 50
                    else f"Treat '{leader['value']}' as the reference case when "
                    f"comparing other {dimension} values."
                ),
                created_at=_now(),
            )
        )
        if len(out) >= 2:
            break
    return out


def _segment_gap_insight(ctx: InsightContext) -> list[InsightModel]:
    """Best vs worst segment on a per-unit basis (not just totals)."""
    if not ctx.measure:
        return []
    from app.engine.stats import group_comparison

    for dimension in ctx.ranked_dimensions()[:3]:
        col = ctx.profile.column(dimension)
        if col is None or not (2 <= col.stats.unique_count <= 30):
            continue
        try:
            comparison = group_comparison(
                ctx.df, value_column=ctx.measure, group_column=dimension
            )
        except Exception:
            continue

        groups = comparison["groups"]
        # Ignore tiny groups (spelling variants, one-off codes): a mean over a
        # handful of rows is not a segment.
        min_rows = max(30, int(ctx.profile.row_count * 0.01))
        groups = [g for g in groups if g["count"] >= min_rows]
        if len(groups) < 2:
            continue

        by_mean = sorted(
            [g for g in groups if g["mean"] is not None],
            key=lambda g: -(g["mean"] or 0),
        )
        if len(by_mean) < 2:
            continue
        best, worst = by_mean[0], by_mean[-1]
        if not worst["mean"]:
            continue
        gap = (best["mean"] - worst["mean"]) / abs(worst["mean"]) * 100
        if gap < 25:
            continue

        test = comparison["test"]
        significant = bool(test.get("significant"))
        ratio = best["mean"] / worst["mean"] if worst["mean"] else None
        gap_text = (
            f"{ratio:.1f}× the" if ratio is not None and ratio >= 2 else f"{gap:.0f}% more"
        )

        return [
            InsightModel(
                id=_new_id(),
                kind="segment_gap",
                title=(
                    f"{dimension} '{best['group']}' averages {gap_text} "
                    f"{ctx.measure} per row {'of' if ratio and ratio >= 2 else 'than'} "
                    f"'{worst['group']}'"
                ),
                explanation=(
                    f"Average {ctx.measure} per row is "
                    f"{compact_number(best['mean'], unit=ctx.unit)} for "
                    f"'{best['group']}' ({best['count']:,} rows) versus "
                    f"{compact_number(worst['mean'], unit=ctx.unit)} for "
                    f"'{worst['group']}' ({worst['count']:,} rows). "
                    + (
                        test.get("interpretation", "")
                        if test.get("applicable")
                        else "Group sizes are too small for a significance test."
                    )
                ),
                severity=Severity.INFO,
                importance=_importance(
                    effect=min(gap / 150.0, 1.0),
                    confidence=0.85 if significant else 0.4,
                    relevance=0.5,
                ),
                metric={
                    "column": ctx.measure,
                    "unit": ctx.unit,
                    "dimension": dimension,
                    "best": best["group"],
                    "worst": worst["group"],
                    "gap_pct": round(gap, 2),
                },
                evidence=Evidence(
                    method=(
                        f"Mean '{ctx.measure}' per row by '{dimension}', compared with a "
                        "Kruskal-Wallis test (no normality assumption)"
                    ),
                    rows=[
                        {
                            dimension: g["group"],
                            "rows": g["count"],
                            f"mean_{ctx.measure}": g["mean"],
                            f"total_{ctx.measure}": g["sum"],
                        }
                        for g in by_mean[:6]
                    ],
                    values={
                        "test": test.get("name"),
                        "p_value": test.get("p_value"),
                        "significant": significant,
                    },
                    sample_size=int(sum(g["count"] for g in groups)),
                    confidence="high" if significant else "low",
                    caveats=(
                        []
                        if significant
                        else [
                            "The difference between groups is not statistically "
                            "significant; it may reflect sample size rather than a real gap."
                        ]
                    ),
                ),
                chart=charts_mod.category_chart(
                    [
                        {"value": g["group"], "metric": g["mean"]}
                        for g in by_mean[:12]
                    ],
                    dimension=dimension,
                    measure=f"Mean {ctx.measure}",
                    total_levels=comparison["group_count"],
                    chart_id=f"segment_gap_{dimension}",
                    title=f"Average {ctx.measure} per row by {dimension}",
                ),
                recommendation=(
                    f"Compare how '{best['group']}' operates against "
                    f"'{worst['group']}' — the per-row gap is large enough to be "
                    "worth replicating."
                ),
                created_at=_now(),
            )
        ]
    return []


def _correlation_insight(ctx: InsightContext) -> list[InsightModel]:
    from app.engine.stats import correlation_matrix

    numeric = [
        c.name
        for c in ctx.profile.columns
        if c.semantic_type.is_numeric and c.stats.unique_count > 5
    ]
    if len(numeric) < 2:
        return []
    try:
        matrix = correlation_matrix(
            ctx.df, columns=numeric[:15], profile=ctx.profile, min_abs=0.5
        )
    except Exception:
        return []

    pairs = [p for p in matrix["top_pairs"] if abs(float(p["r"])) >= 0.6]
    if not pairs:
        return []
    top = pairs[0]

    # Near-perfect correlation usually means one column is derived from the
    # other (revenue = quantity × price); say that rather than calling it a
    # discovery.
    derived = abs(float(top["r"])) > 0.985
    r = float(top["r"])

    return [
        InsightModel(
            id=_new_id(),
            kind="correlation",
            title=(
                f"'{top['x']}' and '{top['y']}' are "
                f"{'almost perfectly' if derived else top['strength']} "
                f"{top['direction']}ly correlated (r = {r:.2f})"
            ),
            explanation=(
                (
                    f"r = {r:.3f} across {top['n']:,} rows. A coefficient this close "
                    "to 1 usually means one column is calculated from the other "
                    "rather than being an independent finding."
                )
                if derived
                else (
                    f"'{top['x']}' and '{top['y']}' move together with r = {r:.3f} "
                    f"over {top['n']:,} rows ({top['strength']} {top['direction']} "
                    f"association, R² = {r * r:.2f}). Association is not causation — "
                    "this identifies a relationship worth investigating, not a cause."
                )
            ),
            severity=Severity.INFO,
            importance=_importance(
                effect=min(abs(r), 1.0) * (0.35 if derived else 1.0),
                confidence=0.8,
                relevance=0.55,
            ),
            metric={
                "x": top["x"],
                "y": top["y"],
                "r": r,
                "r_squared": round(r * r, 4),
                "n": top["n"],
            },
            evidence=Evidence(
                method=f"{matrix['method'].capitalize()} correlation over complete pairs",
                rows=[
                    {"pair": f"{p['x']} ↔ {p['y']}", "r": p["r"], "n": p["n"]}
                    for p in pairs[:5]
                ],
                values={"pairs_above_0.5": matrix["pair_count"]},
                sample_size=int(top["n"]),
                confidence="high",
                caveats=[
                    "Correlation does not establish causation.",
                    *(
                        ["The two columns are probably arithmetically related."]
                        if derived
                        else []
                    ),
                ],
            ),
            chart=charts_mod.scatter_chart(ctx.df, top["x"], top["y"], correlation=top),
            recommendation=(
                None
                if derived
                else (
                    f"If '{top['x']}' is controllable, test whether moving it shifts "
                    f"'{top['y']}' before assuming the link is causal."
                )
            ),
            created_at=_now(),
        )
    ]


def _anomaly_insight(ctx: InsightContext) -> list[InsightModel]:
    if not ctx.measure:
        return []
    from app.engine.anomaly import detect_anomalies

    try:
        result = detect_anomalies(
            ctx.df, profile=ctx.profile, semantics=ctx.semantics, max_results=12
        )
    except Exception:
        return []

    anomalies = result["anomalies"]
    if not anomalies:
        return []

    worst = anomalies[0]
    count = len(anomalies)
    chart = None
    if result["series"]:
        series_frame = pd.DataFrame(result["series"])
        chart = charts_mod.time_series_chart(
            series_frame,
            value_column=result["column"],
            grain=TimeGrain(result["grain"]) if result.get("grain") else TimeGrain.DAY,
            chart_id="anomaly_series",
            title=f"{result['column']} with detected anomalies",
            annotations=[
                ChartAnnotation(
                    x=a["label"],
                    label=(
                        f"{format_delta(a['deviation_pct'])} vs expected"
                        if a.get("deviation_pct") is not None
                        else f"score {a.get('score')}"
                    ),
                    severity=Severity(a["severity"]),
                    value=a.get("observed"),
                )
                for a in anomalies[:8]
            ],
        )

    deviation = worst.get("deviation_pct")
    title = (
        f"{count} unusual {'period' if result['method'] in {'rolling', 'seasonal'} else 'record'}"
        f"{'' if count == 1 else 's'} detected in {result['column']}"
    )

    return [
        InsightModel(
            id=_new_id(),
            kind="anomaly",
            title=title,
            explanation=result["summary"]
            + f" Detection method: {result['method']} — {result['method_reason']}",
            severity=Severity(worst["severity"]),
            importance=_importance(
                effect=min(abs(deviation or 0) / 60.0, 1.0) if deviation else 0.5,
                confidence=0.75,
                relevance=0.9,
            ),
            metric={
                "column": result["column"],
                "unit": result["unit"],
                "method": result["method"],
                "count": count,
                "worst_label": worst["label"],
                "worst_observed": worst["observed"],
                "worst_expected": worst["expected"],
                "worst_deviation_pct": deviation,
            },
            evidence=Evidence(
                method=f"{result['method']} anomaly detection — {result['method_reason']}",
                rows=[
                    {
                        "when": a["label"],
                        "observed": a["observed"],
                        "expected": a["expected"],
                        "deviation_%": a["deviation_pct"],
                        "score": a["score"],
                    }
                    for a in anomalies[:6]
                ],
                values={
                    "method": result["method"],
                    "sensitivity": result["sensitivity"],
                    "total_flagged": count,
                },
                sample_size=len(result["series"]) or int(ctx.profile.row_count),
                confidence="moderate",
                caveats=[
                    "A statistical anomaly is unusual, not necessarily wrong — "
                    "a promotion or a one-off bulk order looks identical to an error."
                ],
            ),
            chart=chart,
            recommendation=(
                f"Check {worst['label']} first: {result['column']} was "
                f"{compact_number(worst['observed'], unit=result['unit'])}"
                + (
                    f" against an expected "
                    f"{compact_number(worst['expected'], unit=result['unit'])}"
                    f" ({format_delta(deviation)})."
                    if worst.get("expected") is not None
                    else "."
                )
            ),
            created_at=_now(),
        )
    ]


def _quality_insight(ctx: InsightContext) -> list[InsightModel]:
    if ctx.quality is None or not ctx.quality.issues:
        return []
    critical = [
        i
        for i in ctx.quality.issues
        if i.severity in {Severity.CRITICAL, Severity.WARNING}
    ]
    if not critical:
        return []
    issue = critical[0]

    return [
        InsightModel(
            id=_new_id(),
            kind="data_quality",
            title=f"Data quality: {issue.title}",
            explanation=(
                issue.detail
                + f" Overall data-quality score is {ctx.quality.score:.0f}/100 "
                f"(grade {ctx.quality.grade}) across {len(ctx.quality.issues)} "
                "detected issue(s)."
            ),
            severity=issue.severity,
            importance=_importance(
                effect=min(issue.affected_ratio * 2.5, 1.0),
                confidence=0.95,
                relevance=0.7,
            ),
            metric={
                "quality_score": ctx.quality.score,
                "grade": ctx.quality.grade,
                "issue_count": len(ctx.quality.issues),
                "rule": issue.rule,
                "affected_rows": issue.affected_rows,
            },
            evidence=Evidence(
                method="Rule-based data-quality assessment across 9 rule families",
                rows=[
                    {
                        "rule": i.rule,
                        "column": i.column,
                        "severity": i.severity.value,
                        "affected_rows": i.affected_rows,
                    }
                    for i in ctx.quality.issues[:6]
                ],
                values=dict(ctx.quality.dimension_scores),
                sample_size=ctx.profile.row_count,
                confidence="high",
            ),
            chart=None,
            recommendation=issue.suggestion,
            created_at=_now(),
        )
    ]


def _distribution_insight(ctx: InsightContext) -> list[InsightModel]:
    if not ctx.measure:
        return []
    col = ctx.profile.column(ctx.measure)
    if col is None or col.stats.skewness is None or col.stats.count < 30:
        return []
    skew = col.stats.skewness
    if abs(skew) < 1.5:
        return []

    mean, median = col.stats.mean, col.stats.median
    if mean is None or median is None or not median:
        return []
    ratio = mean / median if median else 1.0

    return [
        InsightModel(
            id=_new_id(),
            kind="distribution",
            title=(
                f"{ctx.measure} is strongly "
                f"{'right' if skew > 0 else 'left'}-skewed — the mean overstates the typical value"
                if skew > 0
                else f"{ctx.measure} is strongly left-skewed"
            ),
            explanation=(
                f"The mean of {ctx.measure} is "
                f"{compact_number(mean, unit=ctx.unit)} but the median is "
                f"{compact_number(median, unit=ctx.unit)} "
                f"({ratio:.2f}× difference, skewness {skew:.2f}). "
                f"{col.stats.outlier_count:,} values sit beyond the Tukey fences. "
                "Reporting the average here would describe a value most rows do not have."
            ),
            severity=Severity.INFO,
            importance=_importance(
                effect=min(abs(skew) / 4.0, 1.0), confidence=0.9, relevance=0.6
            ),
            metric={
                "column": ctx.measure,
                "unit": ctx.unit,
                "mean": mean,
                "median": median,
                "skewness": skew,
                "p95": col.stats.p95,
                "outlier_count": col.stats.outlier_count,
            },
            evidence=Evidence(
                method="Moment-based skewness with quartile outlier fences",
                rows=[
                    {"statistic": "mean", "value": mean},
                    {"statistic": "median", "value": median},
                    {"statistic": "p95", "value": col.stats.p95},
                    {"statistic": "max", "value": col.stats.max},
                ],
                values={
                    "skewness": skew,
                    "kurtosis": col.stats.kurtosis,
                    "std": col.stats.std,
                },
                sample_size=col.stats.count,
                confidence="high",
            ),
            chart=charts_mod.distribution_chart(ctx.df, ctx.measure),
            recommendation=(
                f"Use the median ({compact_number(median, unit=ctx.unit)}) when "
                f"describing a typical {ctx.measure}, and report the mean only "
                "alongside it."
            ),
            created_at=_now(),
        )
    ]


def _coverage_insight(ctx: InsightContext) -> list[InsightModel]:
    """Flag when missing data materially shrinks the analysable dataset."""
    if not ctx.measure:
        return []
    col = ctx.profile.column(ctx.measure)
    if col is None or col.stats.null_ratio < 0.1:
        return []

    usable = ctx.profile.row_count - col.stats.null_count
    return [
        InsightModel(
            id=_new_id(),
            kind="coverage",
            title=(
                f"{col.stats.null_ratio * 100:.0f}% of rows have no {ctx.measure} value"
            ),
            explanation=(
                f"Every total, average and trend for {ctx.measure} is computed from "
                f"{usable:,} of {ctx.profile.row_count:,} rows. If the missing rows "
                "are not random, the headline figures are biased, not merely smaller."
            ),
            severity=Severity.WARNING
            if col.stats.null_ratio > 0.25
            else Severity.INFO,
            importance=_importance(
                effect=min(col.stats.null_ratio * 2, 1.0),
                confidence=0.95,
                relevance=0.85,
            ),
            metric={
                "column": ctx.measure,
                "missing_rows": col.stats.null_count,
                "missing_pct": round(col.stats.null_ratio * 100, 2),
                "usable_rows": usable,
                "total_rows": ctx.profile.row_count,
            },
            evidence=Evidence(
                method="Null count per column from the dataset profile",
                rows=[
                    {
                        "column": ctx.measure,
                        "present": usable,
                        "missing": col.stats.null_count,
                    }
                ],
                values={"missing_ratio": col.stats.null_ratio},
                sample_size=ctx.profile.row_count,
                confidence="high",
            ),
            chart=None,
            recommendation=(
                f"Check why {ctx.measure} is missing for those rows before "
                "presenting totals as complete."
            ),
            created_at=_now(),
        )
    ]


GENERATORS: tuple[Generator, ...] = (
    _trend_insight,
    _period_change_insight,
    _concentration_insight,
    _anomaly_insight,
    _segment_gap_insight,
    _correlation_insight,
    _distribution_insight,
    _coverage_insight,
    _quality_insight,
)


def generate_insights(
    df: pd.DataFrame,
    *,
    profile: DatasetProfile,
    semantics: SemanticModel,
    quality: QualityReport | None = None,
    limit: int = 8,
) -> list[InsightModel]:
    """Run every generator and return the highest-importance insights."""
    ctx = InsightContext(df, profile, semantics, quality)
    collected: list[InsightModel] = []

    for generator in GENERATORS:
        name = getattr(generator, "__name__", "generator")
        try:
            produced = generator(ctx)
        except Exception as exc:
            # One failing generator must never take down the batch.
            log.warning(
                "insight_generator_failed",
                extra={"generator": name, "error": str(exc)[:200]},
            )
            continue
        collected.extend(produced)

    collected.sort(key=lambda i: -i.importance)
    log.info(
        "insights_generated",
        extra={"produced": len(collected), "returned": min(len(collected), limit)},
    )
    return collected[:limit]
