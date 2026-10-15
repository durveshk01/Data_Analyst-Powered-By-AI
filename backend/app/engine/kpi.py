"""Automatic KPI detection and computation.

KPI *candidates* come from the semantic model (`semantics.kpi_candidates`);
this module turns them into cards with a value, a period-over-period delta, a
sparkline and a plain-English basis line.

The `is_good` flag encodes direction-of-goodness per concept — a 12% rise in
cost is not the same news as a 12% rise in revenue, and a dashboard that paints
both green is worse than one with no colour at all. Where the direction is
genuinely ambiguous (discount, price, a generic measure), `is_good` is left
`None` and the UI shows the delta in a neutral colour.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.engine.charts import kpi_sparkline
from app.engine.query import resolve_column, scalar_aggregate
from app.engine.semantics import choose_reporting_grain
from app.engine.types import (
    Aggregation,
    DatasetProfile,
    KPI,
    SemanticModel,
    TimeGrain,
)
from app.engine.utils import compact_number, pct_change, safe_float, slugify
from app.obs.logging import get_logger

log = get_logger(__name__)

# Which direction is good, per detected business concept. Cost and discount are
# deliberately absent: cost falling because volume fell is not good news, so
# their deltas are shown neutrally rather than painted green.
_HIGHER_IS_BETTER: dict[str, bool] = {
    "revenue": True,
    "profit": True,
    "orders": True,
    "customers": True,
    "quantity": True,
    "products": True,
    "churn": False,
}
# Movements in these are context-dependent, so they are never coloured.
_NEUTRAL_CONCEPTS = frozenset({"cost", "discount", "price"})

_CONCEPT_LABELS: dict[str, str] = {
    "revenue": "Revenue",
    "profit": "Profit",
    "cost": "Cost",
    "quantity": "Units sold",
    "price": "Average price",
    "discount": "Discount",
    "orders": "Orders",
    "customers": "Customers",
    "products": "Products",
    "employee": "Employees",
}


def _label_for(column: str, concept: str | None, agg: Aggregation) -> str:
    if concept and concept in _CONCEPT_LABELS:
        base = _CONCEPT_LABELS[concept]
        if agg == Aggregation.COUNT_DISTINCT:
            return base
        if agg == Aggregation.MEAN:
            return f"Average {base.lower()}"
        return base
    pretty = column.replace("_", " ").strip().title()
    if agg == Aggregation.COUNT_DISTINCT:
        return f"Distinct {pretty}"
    if agg == Aggregation.MEAN:
        return f"Average {pretty.lower()}"
    return pretty


def _basis_sentence(column: str, agg: Aggregation, rows: int) -> str:
    verb = {
        Aggregation.SUM: "Sum of",
        Aggregation.MEAN: "Mean of",
        Aggregation.MEDIAN: "Median of",
        Aggregation.COUNT: "Non-empty count of",
        Aggregation.COUNT_DISTINCT: "Distinct values of",
        Aggregation.MIN: "Minimum of",
        Aggregation.MAX: "Maximum of",
    }.get(agg, f"{agg.value} of")
    return f"{verb} '{column}' across {rows:,} rows."


def compute_kpis(
    df: pd.DataFrame,
    *,
    profile: DatasetProfile,
    semantics: SemanticModel,
    limit: int = 6,
) -> list[KPI]:
    """Compute the dataset's headline KPI cards."""
    from app.engine.trends import build_series  # local import avoids a cycle

    out: list[KPI] = []
    rows = int(df.shape[0])

    grain: TimeGrain | None = None
    time_column = semantics.primary_time_column
    if time_column:
        time_profile = profile.column(time_column)
        grain = choose_reporting_grain(
            time_profile.stats.span_days if time_profile else None,
            time_profile.stats.inferred_grain if time_profile else None,
        )

    for candidate in semantics.kpi_candidates:
        if len(out) >= limit:
            break

        column = str(candidate["column"])
        if column not in df.columns:
            continue
        try:
            column = resolve_column(column, df)
        except Exception:
            continue

        concept = candidate.get("concept")
        unit = str(candidate.get("unit", "number"))
        try:
            agg = Aggregation(str(candidate.get("agg", "sum")))
        except ValueError:
            agg = Aggregation.SUM

        value = scalar_aggregate(df, column, agg)
        if value is None:
            continue

        kpi = KPI(
            key=f"{slugify(column)}_{agg.value}",
            label=_label_for(column, concept, agg),
            column=column,
            agg=agg,
            value=value,
            formatted=compact_number(value, unit=unit),
            unit=unit,  # type: ignore[arg-type]
            basis=_basis_sentence(column, agg, rows),
        )

        # Period-over-period delta plus sparkline, when there is a time axis.
        # Only complete periods are compared: a month still in progress would
        # otherwise show as a collapse.
        if time_column and grain:
            try:
                series = build_series(
                    df,
                    time_column=time_column,
                    value_column=column,
                    grain=grain,
                    agg=agg,
                    complete_only=True,
                )
            except Exception:
                series = None

            if series is not None and len(series) >= 2:
                current = safe_float(series["value"].iloc[-1])
                previous = safe_float(series["value"].iloc[-2])
                delta = pct_change(current, previous)
                kpi = kpi.model_copy(
                    update={
                        "delta_pct": delta,
                        "delta_label": (
                            f"{series['label'].iloc[-1]} vs {series['label'].iloc[-2]}"
                            if delta is not None
                            else None
                        ),
                        "direction": (
                            "up"
                            if delta is not None and delta > 0.5
                            else "down"
                            if delta is not None and delta < -0.5
                            else "flat"
                            if delta is not None
                            else "unknown"
                        ),
                        "is_good": _is_good(concept, column, delta),
                        "sparkline": kpi_sparkline(series),
                    }
                )

        out.append(kpi)

    log.info("kpis_computed", extra={"count": len(out)})
    return out


def _is_good(concept: str | None, column: str, delta: float | None) -> bool | None:
    """Was this movement good news? `None` when the direction is ambiguous."""
    if delta is None or abs(delta) < 0.5:
        return None
    if concept in _NEUTRAL_CONCEPTS:
        return None
    higher_is_better = _HIGHER_IS_BETTER.get(concept or "")
    if higher_is_better is None:
        lowered = column.lower()
        if any(token in lowered for token in ("churn", "refund", "defect", "complaint", "return")):
            higher_is_better = False
        elif any(
            token in lowered
            for token in ("revenue", "sales", "profit", "margin", "orders", "customers")
        ):
            higher_is_better = True
        else:
            return None
    return (delta > 0) == higher_is_better


def kpi_summary_table(kpis: list[KPI]) -> list[dict[str, Any]]:
    """Flatten KPI cards for the report writer."""
    return [
        {
            "label": kpi.label,
            "value": kpi.formatted,
            "raw": kpi.value,
            "change": (
                f"{'+' if (kpi.delta_pct or 0) >= 0 else ''}{kpi.delta_pct:.1f}%"
                if kpi.delta_pct is not None
                else "—"
            ),
            "basis": kpi.basis,
        }
        for kpi in kpis
    ]
