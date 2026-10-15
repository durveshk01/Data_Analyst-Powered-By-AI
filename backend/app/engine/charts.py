"""Automatic chart selection and spec generation.

The *type* of chart is chosen from the data's structure, not from a dropdown,
using the rules in :func:`choose_chart_type`. Each spec records the reason it
was chosen so the UI can show it, and every spec carries explicit guard rails
against misleading output:

* bars always start at zero (`meta.zero_baseline`), because a truncated bar
  axis exaggerates differences;
* a donut is only used for a genuine part-of-whole with ≤ 8 non-negative
  slices — otherwise a ranked bar is used;
* a line chart is only used when the x axis is actually time-ordered;
* any ranking that shows a subset says so in the subtitle ("top 10 of 482"),
  so a partial view is never read as the whole.

The same spec is rendered by Recharts in the browser and by matplotlib in the
PDF report, so both always show identical numbers.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from app.engine.query import resolve_column
from app.engine.stats import histogram
from app.engine.types import (
    Aggregation,
    ChartSeries,
    ChartSpec,
    ChartType,
    ColumnRole,
    DatasetProfile,
    SemanticModel,
    SemanticType,
    TimeGrain,
)
from app.engine.utils import guess_unit, safe_float, slugify
from app.obs.logging import get_logger

log = get_logger(__name__)

_MAX_BARS = 15
_MAX_DONUT_SLICES = 8
_MAX_SCATTER_POINTS = 2_000
_LONG_LABEL_CHARS = 14

Purpose = Literal[
    "trend", "comparison", "distribution", "relationship", "composition", "ranking"
]


# ----------------------------------------------------------- type selection ---
def choose_chart_type(
    *,
    purpose: Purpose,
    x_is_time: bool,
    category_count: int = 0,
    label_length: int = 0,
    all_non_negative: bool = True,
    series_count: int = 1,
) -> tuple[ChartType, str]:
    """Return the chart type and the reason for choosing it."""
    if purpose == "trend":
        if not x_is_time:
            return (
                ChartType.BAR,
                "The x axis is categorical, not time-ordered, so a bar chart is used "
                "— connecting unordered categories with a line would imply a "
                "progression that does not exist.",
            )
        if series_count > 1:
            return (
                ChartType.LINE,
                "Multiple measures over time are compared, so each is drawn as its "
                "own line on a shared time axis.",
            )
        return (
            ChartType.LINE,
            "The x axis is a continuous time sequence, which a line chart reads most "
            "directly for direction and turning points.",
        )

    if purpose == "distribution":
        return (
            ChartType.HISTOGRAM,
            "A histogram shows where values concentrate and how long the tails are, "
            "which a summary statistic hides.",
        )

    if purpose == "relationship":
        return (
            ChartType.SCATTER,
            "Two numeric measures are compared point by point, where a scatter plot "
            "reveals the shape and spread of the relationship.",
        )

    if purpose == "composition":
        if (
            category_count <= _MAX_DONUT_SLICES
            and category_count >= 2
            and all_non_negative
        ):
            return (
                ChartType.DONUT,
                f"{category_count} non-negative parts make up a whole, which a donut "
                "shows as share of total.",
            )
        reason = (
            f"{category_count} categories is too many to read as slices"
            if category_count > _MAX_DONUT_SLICES
            else "some values are negative, which a share-of-total chart cannot represent"
        )
        return (
            ChartType.HORIZONTAL_BAR,
            f"A ranked bar chart is used instead of a donut: {reason}.",
        )

    # comparison / ranking
    if label_length > _LONG_LABEL_CHARS or category_count > 8:
        return (
            ChartType.HORIZONTAL_BAR,
            "Category names are long, so horizontal bars keep every label readable.",
        )
    return (
        ChartType.BAR,
        "Discrete categories are compared on one measure, which vertical bars make "
        "easy to rank by height.",
    )


# ---------------------------------------------------------------- builders ----
def time_series_chart(
    series: pd.DataFrame,
    *,
    value_column: str,
    grain: TimeGrain,
    chart_id: str | None = None,
    title: str | None = None,
    annotations: list[Any] | None = None,
) -> ChartSpec:
    unit = guess_unit(value_column)
    chart_type, reason = choose_chart_type(purpose="trend", x_is_time=True)
    data = [
        {"label": str(row["label"]), "value": safe_float(row["value"])}
        for _, row in series.iterrows()
    ]
    return ChartSpec(
        id=chart_id or f"trend_{slugify(value_column)}_{grain.value}",
        type=chart_type,
        title=title or f"{value_column} by {grain.label}",
        subtitle=f"{len(data)} {grain.label} period(s)",
        x_key="label",
        x_label=grain.label.capitalize(),
        y_label=value_column,
        series=[ChartSeries(key="value", label=value_column, unit=unit)],  # type: ignore[arg-type]
        data=data,
        annotations=annotations or [],
        reason=reason,
        meta={"zero_baseline": False, "grain": grain.value, "measure": value_column},
    )


def category_chart(
    rows: list[dict[str, Any]],
    *,
    dimension: str,
    measure: str,
    total_levels: int | None = None,
    chart_id: str | None = None,
    title: str | None = None,
    purpose: Purpose = "ranking",
) -> ChartSpec:
    unit = guess_unit(measure)
    values = [safe_float(r.get("metric", r.get("value"))) or 0.0 for r in rows]
    labels = [str(r.get("value", r.get("label", ""))) for r in rows]
    max_label = max((len(label) for label in labels), default=0)

    chart_type, reason = choose_chart_type(
        purpose=purpose,
        x_is_time=False,
        category_count=len(rows),
        label_length=max_label,
        all_non_negative=all(v >= 0 for v in values),
    )

    subtitle = f"{len(rows)} categories"
    if total_levels and total_levels > len(rows):
        subtitle = f"Top {len(rows)} of {total_levels} {dimension} values"

    return ChartSpec(
        id=chart_id or f"{slugify(dimension)}_by_{slugify(measure)}",
        type=chart_type,
        title=title or f"{measure} by {dimension}",
        subtitle=subtitle,
        x_key="label",
        x_label=dimension,
        y_label=measure,
        series=[ChartSeries(key="value", label=measure, unit=unit)],  # type: ignore[arg-type]
        data=[
            {
                "label": label,
                "value": value,
                "share_pct": rows[i].get("share_pct"),
            }
            for i, (label, value) in enumerate(zip(labels, values))
        ],
        reason=reason,
        meta={
            "zero_baseline": True,
            "dimension": dimension,
            "measure": measure,
            "truncated": bool(total_levels and total_levels > len(rows)),
            "total_levels": total_levels,
        },
    )


def composition_chart(
    rows: list[dict[str, Any]],
    *,
    dimension: str,
    measure: str,
    total_levels: int | None = None,
) -> ChartSpec:
    values = [safe_float(r.get("metric", r.get("value"))) or 0.0 for r in rows]
    chart_type, reason = choose_chart_type(
        purpose="composition",
        x_is_time=False,
        category_count=len(rows),
        label_length=max((len(str(r.get("value", ""))) for r in rows), default=0),
        all_non_negative=all(v >= 0 for v in values),
    )
    unit = guess_unit(measure)
    total = sum(values) or 1.0

    return ChartSpec(
        id=f"composition_{slugify(dimension)}_{slugify(measure)}",
        type=chart_type,
        title=f"{measure} share by {dimension}",
        subtitle=(
            f"Top {len(rows)} of {total_levels} {dimension} values"
            if total_levels and total_levels > len(rows)
            else f"{len(rows)} categories"
        ),
        x_key="label",
        x_label=dimension,
        y_label=measure,
        series=[ChartSeries(key="value", label=measure, unit=unit)],  # type: ignore[arg-type]
        data=[
            {
                "label": str(r.get("value", "")),
                "value": values[i],
                "share_pct": round(values[i] / total * 100, 2),
            }
            for i, r in enumerate(rows)
        ],
        reason=reason,
        meta={
            "zero_baseline": True,
            "truncated": bool(total_levels and total_levels > len(rows)),
            "total_levels": total_levels,
        },
    )


def distribution_chart(df: pd.DataFrame, column: str, *, bins: int = 24) -> ChartSpec:
    resolved = resolve_column(column, df)
    values = (
        pd.to_numeric(df[resolved], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype="float64")
    )
    buckets = histogram(values, bins=bins)
    chart_type, reason = choose_chart_type(purpose="distribution", x_is_time=False)
    unit = guess_unit(resolved)

    median = float(np.median(values)) if values.size else None
    return ChartSpec(
        id=f"distribution_{slugify(resolved)}",
        type=chart_type,
        title=f"Distribution of {resolved}",
        subtitle=(
            f"{values.size:,} values across {len(buckets)} bins"
            + (f" · median {median:,.2f}" if median is not None else "")
        ),
        x_key="bin",
        x_label=resolved,
        y_label="Row count",
        series=[ChartSeries(key="count", label="Rows", unit="count")],
        data=[
            {
                "bin": b["bin"],
                "count": b["count"],
                "share_pct": b["share_pct"],
                "bin_start": b["bin_start"],
            }
            for b in buckets
        ],
        reason=reason,
        meta={"zero_baseline": True, "column": resolved, "unit": unit, "median": median},
    )


def scatter_chart(
    df: pd.DataFrame, x: str, y: str, *, correlation: dict[str, Any] | None = None
) -> ChartSpec:
    xc, yc = resolve_column(x, df), resolve_column(y, df)
    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(df[xc], errors="coerce"),
            "y": pd.to_numeric(df[yc], errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()

    sampled = len(pair) > _MAX_SCATTER_POINTS
    if sampled:
        pair = pair.sample(n=_MAX_SCATTER_POINTS, random_state=42)

    chart_type, reason = choose_chart_type(purpose="relationship", x_is_time=False)
    subtitle = f"{len(pair):,} point(s)"
    if sampled:
        subtitle += " (random sample)"
    if correlation and correlation.get("r") is not None:
        subtitle += f" · r = {correlation['r']:.2f}"

    return ChartSpec(
        id=f"scatter_{slugify(xc)}_{slugify(yc)}",
        type=chart_type,
        title=f"{yc} vs {xc}",
        subtitle=subtitle,
        x_key="x",
        x_label=xc,
        y_label=yc,
        series=[ChartSeries(key="y", label=yc, unit=guess_unit(yc))],  # type: ignore[arg-type]
        data=[
            {"x": safe_float(row.x), "y": safe_float(row.y)}
            for row in pair.itertuples(index=False)
        ],
        reason=reason,
        meta={
            "zero_baseline": False,
            "sampled": sampled,
            "correlation": correlation or {},
            "x_unit": guess_unit(xc),
        },
    )


def correlation_heatmap_chart(matrix: dict[str, Any]) -> ChartSpec:
    columns = matrix["columns"]
    data: list[dict[str, Any]] = []
    for row in matrix["matrix"]:
        source = row["column"]
        for target in columns:
            data.append({"x": target, "y": source, "value": row.get(target)})
    return ChartSpec(
        id="correlation_heatmap",
        type=ChartType.HEATMAP,
        title="Correlation between numeric columns",
        subtitle=f"{matrix['method'].capitalize()} coefficient · {len(columns)} columns",
        x_key="x",
        x_label="Column",
        y_label="Column",
        series=[ChartSeries(key="value", label="r", unit="number")],
        data=data,
        reason=(
            "A heatmap compares every pair of numeric columns at once, which a list "
            "of coefficients makes hard to scan."
        ),
        meta={"columns": columns, "method": matrix["method"], "zero_baseline": False},
    )


def kpi_sparkline(series: pd.DataFrame, *, limit: int = 24) -> list[dict[str, Any]]:
    tail = series.tail(limit)
    return [
        {"label": str(row["label"]), "value": safe_float(row["value"])}
        for _, row in tail.iterrows()
    ]


# -------------------------------------------------------------- auto charts ---
def auto_charts(
    df: pd.DataFrame,
    *,
    profile: DatasetProfile,
    semantics: SemanticModel,
    limit: int = 6,
) -> list[ChartSpec]:
    """Build the default dashboard chart set for a dataset.

    Order matters: the trend goes first because it answers "what is happening",
    then the biggest categorical breakdown ("where"), then composition and
    distribution ("how concentrated / how spread").
    """
    from app.engine.semantics import choose_reporting_grain
    from app.engine.trends import build_series, top_contributors

    charts: list[ChartSpec] = []
    measure = semantics.primary_measure
    time_column = semantics.primary_time_column

    # 1. Primary measure over time.
    if measure and time_column:
        time_profile = profile.column(time_column)
        grain = choose_reporting_grain(
            time_profile.stats.span_days if time_profile else None,
            time_profile.stats.inferred_grain if time_profile else None,
        )
        try:
            series = build_series(
                df,
                time_column=time_column,
                value_column=measure,
                grain=grain,
                agg=Aggregation.SUM,
            )
            charts.append(
                time_series_chart(series, value_column=measure, grain=grain)
            )
        except Exception as exc:
            log.info("auto_chart_trend_skipped", extra={"error": str(exc)[:160]})

    # 2. Primary measure by the most informative dimension.
    ranked_dimensions = _rank_dimensions(profile, semantics)
    if measure:
        for dimension in ranked_dimensions[:2]:
            try:
                contributions = top_contributors(
                    df, dimension=dimension, measure=measure, top_n=_MAX_BARS
                )
                if not contributions["rows"]:
                    continue
                charts.append(
                    category_chart(
                        contributions["rows"],
                        dimension=dimension,
                        measure=measure,
                        total_levels=contributions["level_count"],
                    )
                )
            except Exception as exc:
                log.info(
                    "auto_chart_category_skipped",
                    extra={"dimension": dimension, "error": str(exc)[:160]},
                )
            if len(charts) >= limit:
                break

    # 3. Composition, when a dimension has few enough levels to read as shares.
    if measure:
        for dimension in ranked_dimensions:
            col = profile.column(dimension)
            if col is None or not (2 <= col.stats.unique_count <= _MAX_DONUT_SLICES):
                continue
            try:
                contributions = top_contributors(
                    df, dimension=dimension, measure=measure, top_n=_MAX_DONUT_SLICES
                )
                if len(contributions["rows"]) >= 2:
                    charts.append(
                        composition_chart(
                            contributions["rows"],
                            dimension=dimension,
                            measure=measure,
                            total_levels=contributions["level_count"],
                        )
                    )
                    break
            except Exception:
                continue

    # 4. Distribution of the primary measure.
    if measure and len(charts) < limit:
        try:
            charts.append(distribution_chart(df, measure))
        except Exception as exc:
            log.info("auto_chart_distribution_skipped", extra={"error": str(exc)[:160]})

    # 5. Strongest numeric relationship.
    if len(charts) < limit:
        numeric = [
            c.name
            for c in profile.columns
            if c.semantic_type.is_numeric
            and c.role == ColumnRole.MEASURE
            and c.stats.unique_count > 5
        ]
        if len(numeric) >= 2:
            try:
                from app.engine.stats import correlation_matrix

                matrix = correlation_matrix(df, columns=numeric[:12], profile=profile)
                top = next(
                    (p for p in matrix["top_pairs"] if abs(float(p["r"])) >= 0.3), None
                )
                if top:
                    charts.append(
                        scatter_chart(df, top["x"], top["y"], correlation=top)
                    )
            except Exception as exc:
                log.info("auto_chart_scatter_skipped", extra={"error": str(exc)[:160]})

    return charts[:limit]


def _rank_dimensions(profile: DatasetProfile, semantics: SemanticModel) -> list[str]:
    """Order dimensions by how useful they are to slice by.

    Preference: a handful of well-populated levels beats either a constant
    column or one with hundreds of levels.
    """
    scored: list[tuple[float, str]] = []
    concept_columns = set(semantics.detected_concepts.values())

    for name in semantics.dimensions:
        col = profile.column(name)
        if col is None:
            continue
        levels = col.stats.unique_count
        if levels < 2 or levels > 500:
            continue
        # Peak usefulness around 4-12 levels.
        if levels <= 12:
            shape = 1.0
        elif levels <= 50:
            shape = 0.6
        else:
            shape = 0.25
        completeness = 1.0 - col.stats.null_ratio
        bonus = 0.35 if name in concept_columns else 0.0
        geo_bonus = 0.15 if col.semantic_type == SemanticType.GEO else 0.0
        scored.append((shape * completeness + bonus + geo_bonus, name))

    scored.sort(key=lambda s: -s[0])
    return [name for _, name in scored]
