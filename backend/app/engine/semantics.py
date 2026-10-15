"""Semantic typing, role assignment and business-concept detection.

This module is why InsightOS can analyse a file it has never seen: it decides,
from dtypes plus name and value evidence, which columns are *measures* (things
you add up), *dimensions* (things you slice by), *time* axes and *identifiers*.
Every downstream feature — automatic KPIs, chart selection, the deterministic
NL parser, forecasting target choice — reads its answers from here.

It is entirely deterministic and uses no LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.engine.types import (
    ColumnRole,
    DatasetProfile,
    SemanticModel,
    SemanticType,
    TimeGrain,
)

# --------------------------------------------------------------- name hints ---
_H = re.IGNORECASE
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def normalise_name(name: str) -> str:
    """`UnitPrice` / `unit-price` / `Unit Price` -> `unit_price`."""
    spaced = _CAMEL_BOUNDARY.sub("_", str(name))
    return re.sub(r"[^a-z0-9]+", "_", spaced.lower()).strip("_")


class NameHint:
    """Matches whole name *tokens*, not substrings.

    A plain substring search mis-types columns constantly: "discount" contains
    "count", "page_views" contains "age", "internet" contains "net". Python's
    ``\\b`` does not help either, because it treats ``_`` as a word character and
    so fails to match "order_count". Instead the name is normalised to
    snake_case and each term must be bounded by a non-letter on both sides
    (start, end, ``_`` or a digit), with an optional plural "s".
    """

    def __init__(self, *terms: str, anchored: tuple[str, ...] = ()) -> None:
        body = "|".join(sorted({re.escape(t) for t in terms}, key=len, reverse=True))
        parts = [rf"(?<![a-z])(?:{body})s?(?![a-z])"] if body else []
        parts.extend(anchored)
        self.pattern = re.compile("|".join(parts), _H)

    def search(self, name: str) -> re.Match[str] | None:
        return self.pattern.search(normalise_name(name))


CURRENCY_NAMES = NameHint(
    "revenue", "sales", "amount", "amt", "price", "cost", "profit", "margin",
    "spend", "budget", "income", "salary", "wage", "fee", "charge", "payment",
    "invoice", "turnover", "gmv", "aov", "mrr", "arr", "value", "subtotal",
    "expense", "cogs", "ltv", "cac", "earning", "inr", "usd", "rupee",
)
PERCENT_NAMES = NameHint(
    "rate", "ratio", "percent", "percentage", "pct", "share", "conversion",
    "churn", "growth", "ctr", "cvr", "utilisation", "utilization", "occupancy",
    "completion", "bounce", "retention", "discount", "markup", "attach",
    anchored=(r"_pc$",),
)
COUNT_NAMES = NameHint(
    "count", "qty", "quantity", "quantities", "unit", "volume", "order",
    "customer", "user", "visitor", "visit", "session", "click", "impression",
    "item", "ticket", "lead", "signup", "transaction", "employee", "student",
    "pageview", "headcount", "stock", "inventory",
)
ID_NAMES = NameHint(
    "id", "uuid", "guid", "sku", "code", "number", "no", "ref", "serial",
    "barcode", "account", "tracking", "pan", "gstin", "isbn", "ean", "upc",
)
TIME_NAMES = NameHint(
    "date", "datetime", "timestamp", "time", "day", "week", "month", "quarter",
    "year", "period", "dob", "created", "updated", "ordered", "shipped",
    "delivered", "joined", "at", "on", "dt",
)
GEO_NAMES = NameHint(
    "country", "nation", "state", "province", "region", "zone", "territory",
    "district", "city", "town", "location", "area", "latitude", "longitude",
    "lat", "lon", "lng", "zip", "zipcode", "postcode", "pincode", "pin",
    "address", "continent", "market", "branch", "store", "warehouse", "site",
)
# Measures where a SUM is meaningless: per-unit prices, scores, rates and
# durations. Summing "unit_price" over 30,000 orders yields a number with no
# interpretation, so these default to MEAN.
MEAN_ONLY_NAMES = NameHint(
    "price", "rate", "score", "rating", "average", "avg", "mean", "median",
    "per", "ratio", "percent", "pct", "day", "duration", "age", "latency",
    "index", "temperature", "height", "speed", "share", "discount",
    "satisfaction", "nps", "csat", "utilisation", "utilization", "aov",
    "probability", "margin_pct", "tenure",
)
_YEAR_NAME = NameHint("year", "yr", "fy", "fiscal_year")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
URL_RE = re.compile(r"^(https?://|www\.)\S+$", _H)

# Canonical business concepts -> ordered synonyms. First match wins, so the
# more specific name appears earlier.
CONCEPT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "net_revenue", "gross_revenue", "sales", "sales_amount",
                "total_sales", "turnover", "gmv", "amount", "total", "value", "income"),
    "profit": ("profit", "net_profit", "gross_profit", "margin", "net_income",
               "earnings", "contribution"),
    "cost": ("cost", "cogs", "expense", "expenses", "spend", "unit_cost",
             "total_cost", "purchase_price"),
    "quantity": ("quantity", "qty", "units", "units_sold", "volume", "count"),
    "price": ("unit_price", "price", "rate", "mrp", "list_price", "selling_price"),
    "discount": ("discount", "discount_amount", "discount_pct", "rebate", "markdown"),
    "orders": ("order_id", "orders", "order_count", "invoice_no", "transaction_id",
               "bill_no", "num_orders"),
    "customers": ("customer_id", "customer", "customers", "client_id", "client",
                  "user_id", "account_id", "buyer"),
    "products": ("product", "product_name", "product_id", "item", "item_name",
                 "sku", "article"),
    "category": ("category", "sub_category", "subcategory", "segment", "class",
                 "product_category", "type", "group"),
    "region": ("region", "state", "country", "city", "zone", "territory",
               "market", "area", "province", "district"),
    "channel": ("channel", "source", "medium", "platform", "sales_channel",
                "marketing_channel", "campaign"),
    "date": ("order_date", "date", "transaction_date", "invoice_date", "created_at",
             "timestamp", "datetime", "period", "month"),
    "employee": ("employee_id", "employee", "staff_id", "agent", "salesperson", "rep"),
    "status": ("status", "state", "stage", "order_status", "payment_status", "result"),
}

# Which concepts, if present together, suggest a domain.
_DOMAIN_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("e-commerce / retail sales", ("revenue", "products", "orders")),
    ("sales performance", ("revenue", "region", "date")),
    ("financial performance", ("revenue", "cost", "profit")),
    ("marketing analytics", ("channel", "customers")),
    ("inventory / supply chain", ("quantity", "products")),
    ("customer analytics", ("customers", "revenue")),
    ("workforce / HR", ("employee", "status")),
)

_MAX_CATEGORICAL_CARDINALITY = 200
_CATEGORICAL_RATIO = 0.5


@dataclass(slots=True)
class ColumnClassification:
    semantic_type: SemanticType
    role: ColumnRole
    unit: str
    notes: list[str]
    is_kpi_candidate: bool


def _ratio_matching(series: pd.Series, pattern: re.Pattern[str], limit: int = 2000) -> float:
    sample = series.dropna().astype(str).head(limit)
    if sample.empty:
        return 0.0
    return float(sample.str.match(pattern).mean())


def classify_column(
    series: pd.Series,
    name: str,
    *,
    row_count: int,
) -> ColumnClassification:
    """Classify one column from its dtype, name and values."""
    notes: list[str] = []
    non_null = series.dropna()
    unique_count = int(non_null.nunique()) if not non_null.empty else 0
    unique_ratio = unique_count / len(non_null) if len(non_null) else 0.0

    # ---- empty -----------------------------------------------------------
    if non_null.empty:
        return ColumnClassification(
            SemanticType.EMPTY, ColumnRole.IGNORED, "none",
            ["Column contains no values."], False,
        )

    # ---- boolean ---------------------------------------------------------
    if pd.api.types.is_bool_dtype(series) or isinstance(series.dtype, pd.BooleanDtype):
        return ColumnClassification(
            SemanticType.BOOLEAN, ColumnRole.DIMENSION, "none", notes, False
        )

    # ---- datetime --------------------------------------------------------
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnClassification(
            SemanticType.DATETIME, ColumnRole.TIME, "none", notes, False
        )

    # ---- numeric ---------------------------------------------------------
    if pd.api.types.is_numeric_dtype(series):
        is_integral = pd.api.types.is_integer_dtype(series) or (
            not non_null.empty and bool((non_null % 1 == 0).all())
        )

        # A high-cardinality integer named like an id is an identifier, not a
        # measure — summing order ids is meaningless.
        if is_integral and ID_NAMES.search(name) and unique_ratio > 0.8:
            notes.append("Treated as an identifier (name and near-unique values).")
            return ColumnClassification(
                SemanticType.IDENTIFIER, ColumnRole.IDENTIFIER, "none", notes, False
            )

        lo, hi = float(non_null.min()), float(non_null.max())

        # A 4-digit integer named 'year' is a time dimension, not a measure.
        if _YEAR_NAME.search(name) and is_integral:
            if 1900 <= lo <= 2200 and 1900 <= hi <= 2200:
                notes.append("Recognised as a calendar year.")
                return ColumnClassification(
                    SemanticType.INTEGER, ColumnRole.TIME, "none", notes, False
                )

        # A rate-like name is only a percentage if the values agree. Otherwise
        # "conversion_value" (in rupees) or "hourly_rate" (₹1,500) would be shown
        # with a % sign.
        if PERCENT_NAMES.search(name) and lo >= -1.0:
            if hi <= 1.0:
                notes.append(
                    "Values lie between 0 and 1, so they are read as fractions "
                    "(0.15 = 15%)."
                )
                return ColumnClassification(
                    SemanticType.PERCENTAGE, ColumnRole.MEASURE, "ratio", notes, True
                )
            # 0-100 scale. Small-cardinality integers here are more likely codes
            # than percentages, so require either decimals or real variety.
            if hi <= 100.0 and (not is_integral or unique_count > 12):
                return ColumnClassification(
                    SemanticType.PERCENTAGE, ColumnRole.MEASURE, "percent", notes, True
                )
        # Currency before count: "order_value" and "unit_price" are money even
        # though "order" and "unit" are count words.
        if CURRENCY_NAMES.search(name):
            return ColumnClassification(
                SemanticType.CURRENCY, ColumnRole.MEASURE, "currency", notes, True
            )
        if COUNT_NAMES.search(name):
            return ColumnClassification(
                SemanticType.INTEGER if is_integral else SemanticType.FLOAT,
                ColumnRole.MEASURE, "count", notes, True,
            )
        # Scores and ratings are measures to average, even on a 1-5 scale.
        if MEAN_ONLY_NAMES.search(name):
            return ColumnClassification(
                SemanticType.INTEGER if is_integral else SemanticType.FLOAT,
                ColumnRole.MEASURE, "number", notes, True,
            )

        # Low-cardinality small integers (0/1 flags, store numbers) behave like
        # dimensions.
        if is_integral and unique_count <= 12 and row_count > 50:
            notes.append(
                f"Only {unique_count} distinct values; usable as a dimension as well as a measure."
            )
            return ColumnClassification(
                SemanticType.INTEGER, ColumnRole.DIMENSION, "count", notes, False
            )

        return ColumnClassification(
            SemanticType.INTEGER if is_integral else SemanticType.FLOAT,
            ColumnRole.MEASURE,
            "count" if is_integral else "number",
            notes,
            True,
        )

    # ---- string-ish ------------------------------------------------------
    if _ratio_matching(series, EMAIL_RE) > 0.8:
        return ColumnClassification(
            SemanticType.EMAIL, ColumnRole.IDENTIFIER, "none", notes, False
        )
    if _ratio_matching(series, URL_RE) > 0.8:
        return ColumnClassification(
            SemanticType.URL, ColumnRole.IDENTIFIER, "none", notes, False
        )

    lengths = non_null.astype(str).str.len()
    avg_len = float(lengths.mean())

    if GEO_NAMES.search(name) and unique_count <= max(500, row_count * 0.2):
        return ColumnClassification(
            SemanticType.GEO, ColumnRole.DIMENSION, "none", notes, False
        )

    if ID_NAMES.search(name) and unique_ratio > 0.7:
        notes.append("Treated as an identifier (name and high distinctness).")
        return ColumnClassification(
            SemanticType.IDENTIFIER, ColumnRole.IDENTIFIER, "none", notes, False
        )

    if unique_ratio > 0.95 and len(non_null) > 100 and avg_len <= 64:
        notes.append("Nearly every value is distinct; treated as an identifier.")
        return ColumnClassification(
            SemanticType.IDENTIFIER, ColumnRole.IDENTIFIER, "none", notes, False
        )

    if unique_count <= _MAX_CATEGORICAL_CARDINALITY or unique_ratio <= _CATEGORICAL_RATIO:
        return ColumnClassification(
            SemanticType.CATEGORICAL, ColumnRole.DIMENSION, "none", notes, False
        )

    notes.append(
        f"Free-text column ({unique_count:,} distinct values, average length "
        f"{avg_len:.0f}); excluded from grouping."
    )
    return ColumnClassification(
        SemanticType.TEXT, ColumnRole.IGNORED, "none", notes, False
    )


# ------------------------------------------------------------ time grain ------
def infer_time_grain(series: pd.Series) -> TimeGrain | None:
    """Infer the natural grain of a datetime column from gap spacing."""
    values = pd.to_datetime(series.dropna(), errors="coerce").dropna()
    if len(values) < 3:
        return None
    unique_sorted = pd.Series(values.unique()).sort_values()
    if len(unique_sorted) < 3:
        return None
    deltas = unique_sorted.diff().dropna()
    if deltas.empty:
        return None
    median_seconds = float(deltas.dt.total_seconds().median())
    if median_seconds <= 0:
        return None
    hour, day = 3600.0, 86_400.0
    if median_seconds < 20 * hour:
        return TimeGrain.HOUR
    if median_seconds < 4 * day:
        return TimeGrain.DAY
    if median_seconds < 20 * day:
        return TimeGrain.WEEK
    if median_seconds < 70 * day:
        return TimeGrain.MONTH
    if median_seconds < 200 * day:
        return TimeGrain.QUARTER
    return TimeGrain.YEAR


def choose_reporting_grain(span_days: int | None, native: TimeGrain | None) -> TimeGrain:
    """Pick the grain to *report* at, which is not always the native grain.

    Five years of daily rows should be charted monthly; three weeks of daily
    rows should stay daily.
    """
    if span_days is None:
        return native or TimeGrain.DAY
    if span_days <= 3:
        return TimeGrain.HOUR
    if span_days <= 70:
        return TimeGrain.DAY
    if span_days <= 400:
        return TimeGrain.WEEK if span_days <= 180 else TimeGrain.MONTH
    if span_days <= 365 * 4:
        return TimeGrain.MONTH
    return TimeGrain.QUARTER


# ------------------------------------------------------- concept detection ----
def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def detect_concepts(column_names: list[str]) -> dict[str, str]:
    """Map canonical business concepts to actual column names."""
    normalised = {_normalise(c): c for c in column_names}
    concepts: dict[str, str] = {}
    used: set[str] = set()

    for concept, synonyms in CONCEPT_SYNONYMS.items():
        # Pass 1: exact normalised match.
        for syn in synonyms:
            if syn in normalised and normalised[syn] not in used:
                concepts[concept] = normalised[syn]
                used.add(normalised[syn])
                break
        if concept in concepts:
            continue
        # Pass 2: substring match, preferring the shortest column name so
        # 'revenue' wins over 'revenue_forecast_adjusted'.
        matches = [
            original
            for norm, original in normalised.items()
            if original not in used and any(syn in norm for syn in synonyms)
        ]
        if matches:
            concepts[concept] = min(matches, key=len)
            used.add(concepts[concept])
    return concepts


def guess_domain(concepts: dict[str, str]) -> str | None:
    best: tuple[str, int] | None = None
    for label, required in _DOMAIN_RULES:
        hits = sum(1 for key in required if key in concepts)
        if hits == len(required) and (best is None or hits > best[1]):
            best = (label, hits)
    if best:
        return best[0]
    if "revenue" in concepts:
        return "sales / revenue analysis"
    return None


def build_synonym_index(column_names: list[str]) -> dict[str, list[str]]:
    """Reverse index used by the deterministic NL parser.

    Maps a spoken term ('sales', 'turnover', 'city') to the columns it could
    mean, so "what was total sales?" resolves even when the column is
    `net_revenue`.
    """
    index: dict[str, list[str]] = {}

    def add(term: str, column: str) -> None:
        term = term.strip().lower()
        if not term:
            return
        bucket = index.setdefault(term, [])
        if column not in bucket:
            bucket.append(column)

    for col in column_names:
        norm = _normalise(col)
        add(col, col)
        add(norm, col)
        add(norm.replace("_", " "), col)
        for token in norm.split("_"):
            if len(token) > 2:
                add(token, col)
        for concept, synonyms in CONCEPT_SYNONYMS.items():
            if any(syn in norm or norm in syn for syn in synonyms):
                add(concept, col)
                for syn in synonyms:
                    add(syn.replace("_", " "), col)
    return index


# ------------------------------------------------------------- the model ------
def build_semantic_model(profile: DatasetProfile) -> SemanticModel:
    """Derive the dataset's analytic shape from its profile."""
    measures = profile.names_by_role(ColumnRole.MEASURE)
    dimensions = profile.names_by_role(ColumnRole.DIMENSION)
    identifiers = profile.names_by_role(ColumnRole.IDENTIFIER)
    time_columns = profile.names_by_role(ColumnRole.TIME)

    names = [c.name for c in profile.columns]
    concepts = detect_concepts(names)

    # --- primary time column: widest real span, preferring a 'date' concept --
    primary_time: str | None = None
    primary_grain: TimeGrain | None = None
    best_score = -1.0
    for col_name in time_columns:
        col = profile.column(col_name)
        if col is None:
            continue
        span = col.stats.span_days or 0
        filled = 1.0 - col.stats.null_ratio
        score = span * filled
        if concepts.get("date") == col_name:
            score *= 1.5
        if score > best_score:
            best_score = score
            primary_time = col_name
            primary_grain = col.stats.inferred_grain
    if primary_time is None and concepts.get("date") in time_columns:
        primary_time = concepts["date"]

    # --- primary measure: concept priority, then variance-weighted magnitude -
    primary_measure: str | None = None
    for concept in ("revenue", "profit", "quantity", "cost"):
        candidate = concepts.get(concept)
        if candidate and candidate in measures:
            primary_measure = candidate
            break
    if primary_measure is None and measures:
        def measure_rank(name: str) -> tuple[int, float]:
            col = profile.column(name)
            if col is None:
                return (0, 0.0)
            unit_bonus = {"currency": 3, "count": 2, "number": 1, "percent": 0}.get(col.unit, 1)
            magnitude = abs(col.stats.sum or 0.0)
            return (unit_bonus, magnitude)

        primary_measure = max(measures, key=measure_rank)

    # --- KPI candidates ------------------------------------------------------
    kpi_candidates: list[dict[str, Any]] = []
    for col_name in measures:
        col = profile.column(col_name)
        if col is None or not col.is_kpi_candidate:
            continue
        concept = next((k for k, v in concepts.items() if v == col_name), None)
        kpi_candidates.append(
            {
                "column": col_name,
                "concept": concept,
                "unit": col.unit,
                "agg": default_aggregation(col_name, col.unit),
                "priority": _kpi_priority(concept, col.unit),
            }
        )
    # Distinct-count KPIs for entity identifiers (customers, orders, products).
    for concept in ("customers", "orders", "products"):
        col_name = concepts.get(concept)
        if not col_name:
            continue
        col = profile.column(col_name)
        if col is None or col.role not in {ColumnRole.IDENTIFIER, ColumnRole.DIMENSION}:
            continue
        kpi_candidates.append(
            {
                "column": col_name,
                "concept": concept,
                "unit": "count",
                "agg": "count_distinct",
                "priority": _kpi_priority(concept, "count"),
            }
        )

    kpi_candidates.sort(key=lambda k: -float(k["priority"]))

    reporting_grain = (
        choose_reporting_grain(
            (profile.column(primary_time).stats.span_days if primary_time else None),
            primary_grain,
        )
        if primary_time
        else None
    )

    return SemanticModel(
        measures=measures,
        dimensions=dimensions,
        time_columns=time_columns,
        identifiers=identifiers,
        primary_time_column=primary_time,
        primary_time_grain=primary_grain,
        reporting_grain=reporting_grain,
        primary_measure=primary_measure,
        kpi_candidates=kpi_candidates[:12],
        detected_concepts=concepts,
        domain_guess=guess_domain(concepts),
        column_synonyms=build_synonym_index(names),
    )


def default_aggregation(column: str, unit: str) -> str:
    """The aggregation that is *meaningful* for this column.

    Chosen from the column's meaning rather than defaulting to SUM, because a
    summed average or a summed unit price is a number with no interpretation.
    """
    if unit in {"percent", "ratio"} or MEAN_ONLY_NAMES.search(column):
        return "mean"
    return "sum"


def _kpi_priority(concept: str | None, unit: str) -> float:
    base = {
        "revenue": 100.0,
        "profit": 95.0,
        "orders": 90.0,
        "customers": 85.0,
        "quantity": 80.0,
        "cost": 70.0,
        "products": 65.0,
        "discount": 50.0,
        "price": 45.0,
    }.get(concept or "", 30.0)
    base += {"currency": 6.0, "count": 4.0, "percent": 2.0}.get(unit, 0.0)
    return base
