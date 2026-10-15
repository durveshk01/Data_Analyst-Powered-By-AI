"""Deterministic natural-language intent parser.

This is InsightOS's answer to "what happens with no LLM configured?". It maps a
plain-English question onto one of the safe analytics tools using lexical rules
plus the dataset's own semantic model — the same synonym index that lets
"total sales" resolve to a column literally named `net_revenue`.

It is not a general language model and does not pretend to be. It recognises
the question shapes that account for the overwhelming majority of real
analytical requests, reports a confidence score, and when confidence is low it
says what it *did* understand and offers concrete alternatives rather than
guessing.

Crucially, the output is a :class:`ParsedIntent` that the planner converts into
the identical tool call an LLM would request, so both paths execute through
one audited code path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.engine.types import Aggregation, SemanticModel, TimeGrain
from app.obs.logging import get_logger

log = get_logger(__name__)

_MONTHS: dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November",
    12: "December",
}
# Words that describe time spans; never reported as "not understood".
_TIME_WORDS = frozenset(
    {
        "hour", "hours", "day", "days", "week", "weeks", "month", "months",
        "quarter", "quarters", "year", "years", "period", "periods", "daily",
        "weekly", "monthly", "quarterly", "yearly", "annual", "annually", "today",
        "yesterday", "recent", "latest", "ahead", "future", "upcoming",
    }
)
# Suffixes stripped when a word does not match a column directly, so
# "profitable" finds `profit` and "selling" finds `sales`-like synonyms.
_SUFFIXES = ("ability", "able", "ably", "ing", "ers", "ed", "es", "s", "ly")
# "average order value", "revenue per customer", "spend per user"...
_PER_PATTERN = re.compile(
    r"\b(?:per|each|for every)\s+(?P<entity>[a-z_][a-z_ ]{1,30}?)(?=\s*(?:[?.!,]|$|\b(?:in|by|for|over|during)\b))",
    re.I,
)
_AOV_PATTERN = re.compile(r"\b(average|avg|mean)\s+order\s+value\b|\baov\b", re.I)

_NUMBER_WORDS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
}

_GRAIN_WORDS: tuple[tuple[re.Pattern[str], TimeGrain], ...] = (
    (re.compile(r"\b(hourly|per hour|by hour|each hour)\b", re.I), TimeGrain.HOUR),
    (re.compile(r"\b(daily|per day|by day|each day|day[- ]?over[- ]?day|dod)\b", re.I), TimeGrain.DAY),
    (re.compile(r"\b(weekly|per week|by week|each week|week[- ]?over[- ]?week|wow)\b", re.I), TimeGrain.WEEK),
    (re.compile(r"\b(monthly|per month|by month|each month|month[- ]?over[- ]?month|mom|month[ly]*\b)", re.I), TimeGrain.MONTH),
    (re.compile(r"\b(quarterly|per quarter|by quarter|each quarter|q[o0]q)\b", re.I), TimeGrain.QUARTER),
    (re.compile(r"\b(yearly|annually|per year|by year|each year|year[- ]?over[- ]?year|yoy)\b", re.I), TimeGrain.YEAR),
)

_AGG_WORDS: tuple[tuple[re.Pattern[str], Aggregation], ...] = (
    (re.compile(r"\b(average|avg|mean)\b", re.I), Aggregation.MEAN),
    (re.compile(r"\bmedian\b", re.I), Aggregation.MEDIAN),
    (re.compile(r"\b(distinct|unique|how many different)\b", re.I), Aggregation.COUNT_DISTINCT),
    (re.compile(r"\b(maximum|max|highest value|largest value)\b", re.I), Aggregation.MAX),
    (re.compile(r"\b(minimum|min|lowest value|smallest value)\b", re.I), Aggregation.MIN),
    (re.compile(r"\b(std|standard deviation|variance|spread)\b", re.I), Aggregation.STD),
    (re.compile(r"\b(total|sum|overall|combined|aggregate)\b", re.I), Aggregation.SUM),
)

# Intent patterns, evaluated in order — the first match wins, so the more
# specific question shapes are listed first.
_INTENT_RULES: tuple[tuple[str, re.Pattern[str], float], ...] = (
    (
        "why_change",
        re.compile(
            r"\bwhy\b.*\b(fall|fell|drop|dropped|decline|declined|decrease|decreased|"
            r"rise|rose|increase|increased|grow|grew|spike|spiked|change|changed|down|up)\b",
            re.I,
        ),
        0.92,
    ),
    (
        "forecast",
        re.compile(
            r"\b(forecast|predict|projection|project|expected|estimate|will be|"
            r"next \d+|next (?:few|three|six|twelve)|outlook)\b",
            re.I,
        ),
        0.93,
    ),
    (
        "anomaly",
        re.compile(
            r"\b(anomal\w*|unusual|outlier\w*|strange|weird|suspicious|irregular|"
            r"abnormal|spike\w*|unexpected|odd)\b",
            re.I,
        ),
        0.94,
    ),
    (
        "compare_periods",
        re.compile(
            r"\b(compare|comparison|versus|vs\.?|against)\b|"
            r"\b(\d{4})\s*(?:and|vs\.?|versus|to)\s*(\d{4})\b",
            re.I,
        ),
        0.88,
    ),
    (
        "correlation",
        re.compile(
            r"\b(correlat\w*|relationship|related to|associated with|"
            r"does .* affect|impact of|influence|driven by|depend on)\b",
            re.I,
        ),
        0.88,
    ),
    (
        "distribution",
        re.compile(
            r"\b(distribut\w*|histogram|spread of|range of|percentile|quartile|"
            r"how (?:is|are) .* spread)\b",
            re.I,
        ),
        0.86,
    ),
    (
        "growth",
        re.compile(
            r"\b(growth|growing|grew|trend|trending|over time|trajectory|"
            r"increase[d]? by|decrease[d]? by|change over)\b",
            re.I,
        ),
        0.84,
    ),
    (
        "top_n",
        re.compile(
            r"\b(top|best|highest|most|largest|biggest|leading|strongest)\b",
            re.I,
        ),
        0.86,
    ),
    (
        "bottom_n",
        re.compile(
            r"\b(bottom|worst|lowest|least|smallest|weakest|poorest|under[- ]?perform\w*)\b",
            re.I,
        ),
        0.86,
    ),
    (
        "breakdown",
        re.compile(r"\b(by|per|across|for each|split by|group(?:ed)? by|broken down by)\b", re.I),
        0.78,
    ),
    (
        "time_series",
        re.compile(
            r"\b(monthly|daily|weekly|quarterly|yearly|annually|hourly|over time|"
            r"by month|by day|by week|by quarter|by year|time series)\b",
            re.I,
        ),
        0.85,
    ),
    (
        "schema",
        re.compile(
            r"\b(what columns|which columns|what fields|schema|structure|"
            r"what data|describe the (?:data|dataset)|column names)\b",
            re.I,
        ),
        0.95,
    ),
    (
        "sample",
        re.compile(r"\b(show me (?:some|a few|the first)|sample|preview|first \d+ rows|head)\b", re.I),
        0.9,
    ),
    (
        "quality",
        re.compile(
            r"\b(data quality|missing|nulls?|duplicates?|clean|dirty|problems?|issues?)\b",
            re.I,
        ),
        0.84,
    ),
    (
        "count",
        re.compile(r"\bhow many\b|\b(count|number of|no\. of)\b", re.I),
        0.85,
    ),
    (
        "aggregate",
        re.compile(
            r"\b(total|sum|average|avg|mean|median|how much|what was|what is|"
            r"maximum|minimum|max|min)\b",
            re.I,
        ),
        0.8,
    ),
)

_FILTER_PATTERN = re.compile(
    r"\b(?:in|for|from|within|where|at)\s+(?:the\s+)?"
    r"(?P<value>[A-Za-z][A-Za-z0-9&'\- ]{1,40}?)"
    r"(?=\s+(?:region|category|channel|product|state|city|segment|status|market)\b|[,.?!]|$)",
    re.I,
)

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "for", "in", "on", "at", "by", "to", "and", "or",
        "what", "which", "who", "whom", "was", "were", "is", "are", "be", "been",
        "show", "me", "my", "give", "tell", "find", "list", "get", "please",
        "how", "much", "many", "total", "sum", "average", "avg", "mean", "median",
        "did", "do", "does", "has", "have", "had", "can", "could", "would",
        "top", "best", "highest", "most", "largest", "biggest", "worst", "lowest",
        "least", "bottom", "smallest", "per", "each", "across", "over", "time",
        "last", "this", "next", "previous", "between", "compare", "versus", "vs",
        "it", "its", "there", "that", "then", "than", "with", "from", "about",
        "why", "when", "where", "all", "any", "some", "more", "less",
    }
)


@dataclass(slots=True)
class ParsedIntent:
    """A question, understood."""

    intent: str
    confidence: float
    measure: str | None = None
    aggregation: Aggregation = Aggregation.SUM
    dimension: str | None = None
    time_grain: TimeGrain | None = None
    limit: int = 10
    periods: list[str] = field(default_factory=list)
    filters: list[dict[str, Any]] = field(default_factory=list)
    second_measure: str | None = None
    per_entity: str | None = None
    horizon: int = 3
    column: str | None = None
    matched_terms: list[str] = field(default_factory=list)
    unresolved_terms: list[str] = field(default_factory=list)
    original: str = ""

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.55


# ------------------------------------------------------------- extraction ----
def _extract_limit(text: str, default: int = 10) -> int:
    match = re.search(r"\b(?:top|bottom|first|last|best|worst)\s+(\d{1,3})\b", text, re.I)
    if match:
        return max(1, min(int(match.group(1)), 100))
    match = re.search(r"\b(\d{1,3})\s+(?:rows|records|products|items|customers|results)\b", text, re.I)
    if match:
        return max(1, min(int(match.group(1)), 100))
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b(?:top|bottom|first|last)\s+{word}\b", text, re.I):
            return value
    return default


def _extract_horizon(text: str, grain: TimeGrain | None) -> int:
    match = re.search(
        r"\bnext\s+(\d{1,2}|one|two|three|four|five|six|twelve)\s*"
        r"(hour|day|week|month|quarter|year)s?\b",
        text,
        re.I,
    )
    if match:
        raw = match.group(1).lower()
        count = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw, 3)
        return max(1, min(count, 36))
    match = re.search(r"\bnext\s+(\d{1,2})\b", text, re.I)
    if match:
        return max(1, min(int(match.group(1)), 36))
    if re.search(r"\bnext\s+(?:few|couple)\b", text, re.I):
        return 3
    return 3


def _extract_grain(text: str) -> TimeGrain | None:
    for pattern, grain in _GRAIN_WORDS:
        if pattern.search(text):
            return grain
    match = re.search(
        r"\bnext\s+(?:\d{1,2}|one|two|three|four|five|six|twelve)\s*"
        r"(hour|day|week|month|quarter|year)s?\b",
        text,
        re.I,
    )
    if match:
        return {
            "hour": TimeGrain.HOUR,
            "day": TimeGrain.DAY,
            "week": TimeGrain.WEEK,
            "month": TimeGrain.MONTH,
            "quarter": TimeGrain.QUARTER,
            "year": TimeGrain.YEAR,
        }[match.group(1).lower()]
    return None


def _extract_periods(text: str) -> list[str]:
    """Pull explicit period references: 'September', '2025', 'Q3 2026'."""
    found: list[str] = []

    for match in re.finditer(r"\bq([1-4])\s*(?:of\s*)?(\d{4})?\b", text, re.I):
        quarter, year = match.group(1), match.group(2)
        found.append(f"Q{quarter} {year}" if year else f"Q{quarter}")

    for match in re.finditer(r"\b([a-z]{3,9})\s*(\d{4})?\b", text, re.I):
        name = match.group(1).lower()
        if name not in _MONTHS:
            continue
        month = _MONTHS[name]
        year = match.group(2)
        label = _MONTH_NAMES.get(month, name.capitalize())
        found.append(f"{label} {year}" if year else label)

    for match in re.finditer(r"\b(20\d{2})\b", text):
        year = match.group(1)
        if not any(year in f for f in found):
            found.append(year)

    # Preserve order of appearance, drop duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in found:
        if item.lower() not in seen:
            seen.add(item.lower())
            ordered.append(item)
    return ordered[:4]


def _extract_aggregation(text: str, measure_unit_hint: str | None = None) -> Aggregation:
    for pattern, agg in _AGG_WORDS:
        if pattern.search(text):
            return agg
    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", text, re.I):
        return Aggregation.COUNT
    # A rate should be averaged, not summed — summing percentages is meaningless.
    if measure_unit_hint == "percent":
        return Aggregation.MEAN
    return Aggregation.SUM


# ------------------------------------------------------- column resolution ---
def _candidate_terms(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\s&]", " ", text.lower())
    tokens = [t for t in cleaned.split() if t and t not in _STOPWORDS and len(t) > 2]
    # Bigrams first so "unit price" beats "price".
    terms: list[str] = []
    for i in range(len(tokens) - 1):
        terms.append(f"{tokens[i]} {tokens[i + 1]}")
    terms.extend(tokens)
    return terms


def _lookup(term: str, index: dict[str, list[str]]) -> list[str]:
    """Exact synonym lookup, then a light stem as a second chance."""
    hits = index.get(term)
    if hits:
        return hits
    if " " in term:
        return []
    for suffix in _SUFFIXES:
        if term.endswith(suffix) and len(term) - len(suffix) >= 3:
            stem = term[: -len(suffix)]
            for candidate in (stem, stem + "e", stem + "s"):
                if candidate in index:
                    return index[candidate]
    return []


def resolve_columns(
    text: str, semantics: SemanticModel
) -> tuple[list[str], list[str], list[str]]:
    """Resolve column references in `text`.

    Returns (measures, dimensions, matched_terms).
    """
    index = semantics.column_synonyms
    measure_set = set(semantics.measures)
    dimension_set = set(semantics.dimensions) | set(semantics.identifiers)
    time_set = set(semantics.time_columns)

    measures: list[str] = []
    dimensions: list[str] = []
    matched: list[str] = []
    claimed: set[str] = set()

    for term in _candidate_terms(text):
        for column in _lookup(term, index):
            if column in claimed:
                continue
            if column in measure_set:
                measures.append(column)
            elif column in dimension_set or column in time_set:
                dimensions.append(column)
            else:
                continue
            claimed.add(column)
            matched.append(term)
            break

    return measures, dimensions, matched


def _extract_filters(
    text: str, semantics: SemanticModel, known_dimensions: list[str]
) -> list[dict[str, Any]]:
    """Find simple equality filters such as 'in the West region'."""
    filters: list[dict[str, Any]] = []
    for match in _FILTER_PATTERN.finditer(text):
        value = match.group("value").strip()
        if not value or value.lower() in _STOPWORDS or len(value) < 2:
            continue
        # The dimension is the word that follows, when it names a column.
        tail = text[match.end() : match.end() + 24].strip().lower()
        column: str | None = None
        for candidate in re.findall(r"[a-z_]+", tail)[:2]:
            hits = semantics.column_synonyms.get(candidate, [])
            if hits:
                column = hits[0]
                break
        if column is None and known_dimensions:
            column = known_dimensions[0]
        if column is None:
            continue
        filters.append({"column": column, "op": "eq", "value": value})
        if len(filters) >= 3:
            break
    return filters


# ------------------------------------------------------------------ parse ----
def parse_question(question: str, semantics: SemanticModel) -> ParsedIntent:
    """Parse a question into a structured intent."""
    text = (question or "").strip()
    if not text:
        return ParsedIntent(intent="unknown", confidence=0.0, original=question)

    normalised = re.sub(r"\s+", " ", text)[:500]

    measures, dimensions, matched = resolve_columns(normalised, semantics)
    measure = measures[0] if measures else semantics.primary_measure
    second_measure = measures[1] if len(measures) > 1 else None

    # A time column mentioned explicitly is an axis, not a dimension to slice by.
    time_mentioned = [d for d in dimensions if d in set(semantics.time_columns)]
    dimensions = [d for d in dimensions if d not in set(semantics.time_columns)]
    dimension = dimensions[0] if dimensions else None

    grain = _extract_grain(normalised)
    periods = _extract_periods(normalised)
    limit = _extract_limit(normalised)
    aggregation = _extract_aggregation(normalised)

    intent = "aggregate"
    confidence = 0.5
    for name, pattern, score in _INTENT_RULES:
        if pattern.search(normalised):
            intent, confidence = name, score
            break

    # ---- intent-specific refinement -------------------------------------
    per_entity: str | None = None
    if _AOV_PATTERN.search(normalised):
        # "Average order value" is total revenue over distinct orders.
        concepts = semantics.detected_concepts
        per_entity = concepts.get("orders")
        revenue = concepts.get("revenue")
        if revenue and revenue in semantics.measures:
            measure = revenue
        if per_entity:
            intent, confidence = "aggregate", 0.92
            aggregation = Aggregation.SUM
    elif (per_match := _PER_PATTERN.search(normalised)) and intent in {
        "aggregate", "breakdown",
    }:
        phrase = per_match.group("entity").strip().lower()
        if not any(word in _TIME_WORDS for word in phrase.split()):
            entity_columns: list[str] = []
            for word in phrase.split():
                entity_columns.extend(_lookup(word, semantics.column_synonyms))
            identifiers = set(semantics.identifiers) | {
                semantics.detected_concepts.get(c) for c in ("customers", "orders")
            }
            entity = next((c for c in entity_columns if c in identifiers), None)
            if entity and re.search(r"\b(average|avg|mean|typical)\b", normalised, re.I):
                per_entity = entity
                intent, confidence = "aggregate", 0.88
                aggregation = Aggregation.SUM
                dimension = None

    if intent in {"top_n", "bottom_n"} and dimension is None:
        # "top products" with no explicit dimension: fall back to the most
        # natural entity dimension available.
        for concept in ("products", "customers", "category", "region", "channel"):
            candidate = semantics.detected_concepts.get(concept)
            if candidate and candidate not in set(semantics.measures):
                dimension = candidate
                break
        if dimension is None and semantics.dimensions:
            dimension = semantics.dimensions[0]
        if dimension is None:
            intent, confidence = "aggregate", 0.45

    reporting_grain = (
        semantics.reporting_grain or semantics.primary_time_grain or TimeGrain.MONTH
    )

    if intent == "breakdown":
        # "revenue by month" is a time series, not a categorical breakdown.
        if grain or (time_mentioned and not dimension):
            intent = "time_series"
            confidence = 0.85
        elif dimension is None:
            intent, confidence = "aggregate", 0.45

    if intent == "time_series" and grain is None:
        grain = reporting_grain

    if intent == "growth" and grain is None:
        grain = reporting_grain

    if intent == "compare_periods":
        if len(periods) < 2:
            # "compare regions" is a breakdown, not a period comparison.
            if dimension:
                intent, confidence = "breakdown", 0.7
            elif len(periods) == 1:
                confidence = 0.6
        if grain is None:
            grain = _infer_grain_from_periods(periods) or reporting_grain

    if intent == "why_change" and grain is None:
        # The dimension is deliberately left open unless the user named one:
        # the engine tries several and reports where the change concentrates.
        grain = _infer_grain_from_periods(periods) or reporting_grain

    if intent == "correlation" and second_measure is None:
        numeric_others = [m for m in semantics.measures if m != measure]
        second_measure = numeric_others[0] if numeric_others else None
        if second_measure is None:
            intent, confidence = "aggregate", 0.4

    if intent == "distribution":
        aggregation = Aggregation.SUM

    count_column: str | None = None
    if intent == "count":
        if dimension:
            # "How many customers?" counts distinct customers, not rows.
            count_column = dimension
            aggregation = Aggregation.COUNT_DISTINCT
        elif measures:
            count_column = measures[0]
            aggregation = Aggregation.COUNT
        else:
            aggregation = Aggregation.COUNT

    horizon = _extract_horizon(normalised, grain)
    filters = _extract_filters(normalised, semantics, dimensions)

    # Respect a rate column's natural aggregation.
    if measure and intent not in {"count"} and not per_entity:
        from app.engine.utils import guess_unit

        if guess_unit(measure) in {"percent", "ratio"} and aggregation == Aggregation.SUM:
            aggregation = Aggregation.MEAN

    # Confidence penalty when we had to guess the measure.
    if (
        not measures
        and not per_entity
        and intent not in {"schema", "sample", "quality", "anomaly", "count", "forecast"}
    ):
        confidence *= 0.8

    unresolved = _unresolved_terms(normalised, matched, semantics)

    parsed = ParsedIntent(
        intent=intent,
        confidence=round(min(confidence, 0.99), 3),
        measure=measure,
        aggregation=aggregation,
        dimension=dimension,
        time_grain=grain,
        limit=limit,
        periods=periods,
        filters=filters,
        second_measure=second_measure,
        per_entity=per_entity,
        horizon=horizon,
        column=count_column if intent == "count" else (measure or dimension),
        matched_terms=matched,
        unresolved_terms=unresolved,
        original=question,
    )
    log.info(
        "question_parsed",
        extra={
            "intent": parsed.intent,
            "confidence": parsed.confidence,
            "measure": parsed.measure,
            "dimension": parsed.dimension,
        },
    )
    return parsed


def _infer_grain_from_periods(periods: list[str]) -> TimeGrain | None:
    if not periods:
        return None
    first = periods[0].lower()
    if re.fullmatch(r"20\d{2}", first):
        return TimeGrain.YEAR
    if first.startswith("q"):
        return TimeGrain.QUARTER
    if any(first.startswith(name[:3]) for name in _MONTHS):
        return TimeGrain.MONTH
    return None


def _unresolved_terms(
    text: str, matched: list[str], semantics: SemanticModel
) -> list[str]:
    """Content words the parser could not tie to anything in the dataset.

    Surfaced to the user so a wrong answer is traceable to a misread word.
    """
    matched_tokens = {t for term in matched for t in term.split()}
    known = set(semantics.column_synonyms.keys())
    out: list[str] = []
    for token in re.findall(r"[a-z]{4,}", text.lower()):
        if token in _STOPWORDS or token in matched_tokens or token in known:
            continue
        if token in _MONTHS or token in _NUMBER_WORDS or token in _TIME_WORDS:
            continue
        if _lookup(token, semantics.column_synonyms):
            continue
        if any(
            pattern.search(token)
            for _, pattern, _ in _INTENT_RULES
        ):
            continue
        if token not in out:
            out.append(token)
    return out[:6]


SUPPORTED_QUESTION_SHAPES: tuple[str, ...] = (
    "What was total <measure>?",
    "What is the average <measure>?",
    "What is the average order value?",
    "How many <entity>?",
    "Top 10 <dimension> by <measure>",
    "Worst performing <dimension>",
    "<measure> by <dimension>",
    "Show monthly / weekly / daily <measure>",
    "<measure> growth",
    "Compare <period A> and <period B>",
    "Why did <measure> fall in <period>?",
    "Find anomalies in <measure>",
    "Forecast the next 3 months",
    "Is <measure A> correlated with <measure B>?",
    "Distribution of <measure>",
    "What columns are in this dataset?",
    "Show me a sample of the data",
    "What are the data quality issues?",
)
