"""Engine-wide typed contracts.

These Pydantic models are the boundary between untrusted input (an HTTP body,
or a JSON tool call proposed by an LLM) and the analytics engine. Because the
engine only ever accepts a validated :class:`QuerySpec` — never a string of
code or SQL — a model that is fully compromised by prompt injection still
cannot do anything outside this grammar.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Display unit of a measure. "ratio" is a fraction (0.15) shown as a
# percentage (15%); "percent" is already on a 0-100 scale.
Unit = Literal["currency", "percent", "ratio", "count", "number", "none"]


# ------------------------------------------------------------- enumerations --
class SemanticType(str, Enum):
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    TEXT = "text"
    IDENTIFIER = "identifier"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    GEO = "geo"
    EMAIL = "email"
    URL = "url"
    EMPTY = "empty"
    UNKNOWN = "unknown"

    @property
    def is_numeric(self) -> bool:
        return self in {
            SemanticType.INTEGER,
            SemanticType.FLOAT,
            SemanticType.CURRENCY,
            SemanticType.PERCENTAGE,
        }


class ColumnRole(str, Enum):
    MEASURE = "measure"
    DIMENSION = "dimension"
    TIME = "time"
    IDENTIFIER = "identifier"
    IGNORED = "ignored"


class Aggregation(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    STD = "std"
    VAR = "var"
    P25 = "p25"
    P75 = "p75"
    P90 = "p90"
    P95 = "p95"


class FilterOp(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    BETWEEN = "between"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


class TimeGrain(str, Enum):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

    @property
    def pandas_freq(self) -> str:
        return {
            TimeGrain.HOUR: "h",
            TimeGrain.DAY: "D",
            TimeGrain.WEEK: "W-MON",
            TimeGrain.MONTH: "MS",
            TimeGrain.QUARTER: "QS",
            TimeGrain.YEAR: "YS",
        }[self]

    @property
    def label(self) -> str:
        return {
            TimeGrain.HOUR: "hour",
            TimeGrain.DAY: "day",
            TimeGrain.WEEK: "week",
            TimeGrain.MONTH: "month",
            TimeGrain.QUARTER: "quarter",
            TimeGrain.YEAR: "year",
        }[self]

    @property
    def periods_per_year(self) -> int:
        return {
            TimeGrain.HOUR: 8760,
            TimeGrain.DAY: 365,
            TimeGrain.WEEK: 52,
            TimeGrain.MONTH: 12,
            TimeGrain.QUARTER: 4,
            TimeGrain.YEAR: 1,
        }[self]


class ChartType(str, Enum):
    LINE = "line"
    AREA = "area"
    BAR = "bar"
    HORIZONTAL_BAR = "hbar"
    STACKED_BAR = "stacked_bar"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    DONUT = "donut"
    HEATMAP = "heatmap"
    KPI = "kpi"
    TABLE = "table"
    FORECAST = "forecast"
    BOXPLOT = "boxplot"


class Severity(str, Enum):
    INFO = "info"
    POSITIVE = "positive"
    WARNING = "warning"
    CRITICAL = "critical"


# ----------------------------------------------------------------- profiling --
class ColumnStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = 0
    null_count: int = 0
    null_ratio: float = 0.0
    unique_count: int = 0
    unique_ratio: float = 0.0
    # numeric
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    variance: float | None = None
    sum: float | None = None
    p01: float | None = None
    p05: float | None = None
    p25: float | None = None
    p75: float | None = None
    p95: float | None = None
    p99: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    zero_count: int = 0
    negative_count: int = 0
    outlier_count: int = 0
    # temporal
    min_date: str | None = None
    max_date: str | None = None
    span_days: int | None = None
    inferred_grain: TimeGrain | None = None
    # categorical / text
    top_values: list[dict[str, Any]] = Field(default_factory=list)
    mode: Any | None = None
    avg_length: float | None = None
    max_length: int | None = None
    # sampling
    sample_values: list[Any] = Field(default_factory=list)


class ColumnProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    position: int
    dtype: str
    semantic_type: SemanticType
    role: ColumnRole
    unit: Unit = "none"
    stats: ColumnStats
    is_kpi_candidate: bool = False
    notes: list[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_count: int
    column_count: int
    memory_bytes: int = 0
    duplicate_rows: int = 0
    duplicate_ratio: float = 0.0
    total_cells: int = 0
    missing_cells: int = 0
    missing_ratio: float = 0.0
    columns: list[ColumnProfile] = Field(default_factory=list)
    profiled_rows: int = 0
    sampled: bool = False
    profiling_ms: int = 0

    def column(self, name: str) -> ColumnProfile | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def names_by_role(self, role: ColumnRole) -> list[str]:
        return [c.name for c in self.columns if c.role == role]


# ------------------------------------------------------------------ queries ---
class MetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    agg: Aggregation = Aggregation.SUM
    alias: str | None = None

    @property
    def output_name(self) -> str:
        return self.alias or f"{self.column}_{self.agg.value}"


class DimensionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    time_grain: TimeGrain | None = None
    alias: str | None = None

    @property
    def output_name(self) -> str:
        if self.alias:
            return self.alias
        if self.time_grain:
            return f"{self.column}_{self.time_grain.value}"
        return self.column


class FilterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    op: FilterOp = FilterOp.EQ
    value: Any = None

    @model_validator(mode="after")
    def _check_value(self) -> "FilterSpec":
        if self.op in {FilterOp.IS_NULL, FilterOp.NOT_NULL}:
            return self
        if self.value is None:
            raise ValueError(f"Filter op '{self.op.value}' requires a value.")
        if self.op == FilterOp.BETWEEN:
            if not isinstance(self.value, (list, tuple)) or len(self.value) != 2:
                raise ValueError("Filter op 'between' requires exactly two values.")
        if self.op in {FilterOp.IN, FilterOp.NOT_IN}:
            if not isinstance(self.value, (list, tuple)) or not self.value:
                raise ValueError(f"Filter op '{self.op.value}' requires a non-empty list.")
            if len(self.value) > 1000:
                raise ValueError("Filter list cannot exceed 1000 values.")
        return self


class SortSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by: str
    desc: bool = True


class QuerySpec(BaseModel):
    """The only grammar the engine will execute. Validated against the real
    dataset schema in :func:`app.engine.query.run_query`."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[MetricSpec] = Field(default_factory=list, max_length=12)
    dimensions: list[DimensionSpec] = Field(default_factory=list, max_length=4)
    filters: list[FilterSpec] = Field(default_factory=list, max_length=20)
    sort: list[SortSpec] = Field(default_factory=list, max_length=4)
    limit: int = Field(default=100, ge=1, le=10_000)
    offset: int = Field(default=0, ge=0, le=1_000_000)
    include_totals: bool = False

    @field_validator("metrics", "dimensions")
    @classmethod
    def _no_duplicate_outputs(cls, v: list[Any]) -> list[Any]:
        names = [item.output_name for item in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate output names in metrics/dimensions.")
        return v

    @model_validator(mode="after")
    def _non_empty(self) -> "QuerySpec":
        if not self.metrics and not self.dimensions:
            raise ValueError("A query must request at least one metric or dimension.")
        return self


class QueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool = False
    totals: dict[str, Any] | None = None
    duration_ms: int = 0
    spec: QuerySpec | None = None
    applied_filters: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------- charts ---
class ChartSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    unit: Unit = "none"
    kind: ChartType | None = None


class ChartAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: Any
    label: str
    severity: Severity = Severity.WARNING
    value: float | None = None


class ChartSpec(BaseModel):
    """A declarative chart description. The frontend renders it with Recharts;
    the report writer renders the same spec to PNG with matplotlib, so both
    surfaces are guaranteed to show the same numbers."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: ChartType
    title: str
    subtitle: str | None = None
    x_key: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    series: list[ChartSeries] = Field(default_factory=list)
    data: list[dict[str, Any]] = Field(default_factory=list)
    annotations: list[ChartAnnotation] = Field(default_factory=list)
    stacked: bool = False
    reason: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------- output ---
class KPI(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    column: str
    agg: Aggregation
    value: float | None
    formatted: str
    unit: Unit = "number"
    delta_pct: float | None = None
    delta_label: str | None = None
    direction: Literal["up", "down", "flat", "unknown"] = "unknown"
    is_good: bool | None = None
    sparkline: list[dict[str, Any]] = Field(default_factory=list)
    basis: str = ""


class Evidence(BaseModel):
    """Every insight must carry the numbers it was derived from."""

    model_config = ConfigDict(extra="forbid")

    method: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    values: dict[str, Any] = Field(default_factory=dict)
    query: QuerySpec | None = None
    sample_size: int | None = None
    confidence: str | None = None
    caveats: list[str] = Field(default_factory=list)


class InsightModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    title: str
    explanation: str
    severity: Severity = Severity.INFO
    importance: float = 0.0
    metric: dict[str, Any] = Field(default_factory=dict)
    evidence: Evidence
    chart: ChartSpec | None = None
    recommendation: str | None = None
    created_at: str | None = None


class AnomalyModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    method: str
    column: str
    label: str
    observed: float | None = None
    expected: float | None = None
    deviation_pct: float | None = None
    score: float | None = None
    severity: Severity = Severity.WARNING
    row_index: int | None = None
    explanation: str = ""
    context: dict[str, Any] = Field(default_factory=dict)


class ForecastPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    period: str
    value: float
    lower: float | None = None
    upper: float | None = None


class ForecastModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_column: str
    time_column: str
    grain: TimeGrain
    horizon: int
    model_name: str
    model_reason: str = ""
    metrics: dict[str, float | None] = Field(default_factory=dict)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    points: list[ForecastPoint] = Field(default_factory=list)
    chart: ChartSpec | None = None
    caveats: list[str] = Field(default_factory=list)


class QualityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    rule: str
    column: str | None = None
    severity: Severity = Severity.WARNING
    title: str
    detail: str
    affected_rows: int = 0
    affected_ratio: float = 0.0
    examples: list[Any] = Field(default_factory=list)
    suggestion: str | None = None
    # A machine-applicable fix, when one exists. Always opt-in.
    fix: dict[str, Any] | None = None


class QualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float
    grade: str
    issues: list[QualityIssue] = Field(default_factory=list)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    injection_findings: list[dict[str, Any]] = Field(default_factory=list)
    checked_rules: int = 0


class SemanticModel(BaseModel):
    """What the dataset appears to be *about* — the basis for automatic KPIs,
    chart choices and the deterministic NL parser."""

    model_config = ConfigDict(extra="forbid")

    measures: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    time_columns: list[str] = Field(default_factory=list)
    identifiers: list[str] = Field(default_factory=list)
    primary_time_column: str | None = None
    # The grain the data is recorded at (minute-stamped orders -> "hour")...
    primary_time_grain: TimeGrain | None = None
    # ...versus the grain worth reporting at (21 months of orders -> "month").
    reporting_grain: TimeGrain | None = None
    primary_measure: str | None = None
    kpi_candidates: list[dict[str, Any]] = Field(default_factory=list)
    detected_concepts: dict[str, str] = Field(default_factory=dict)
    domain_guess: str | None = None
    column_synonyms: dict[str, list[str]] = Field(default_factory=dict)
