"""Dataset profiling: per-column statistics and dataset-level shape.

Performance strategy for the free tier: numeric aggregates run on the full
column (NumPy makes that cheap even at 250k rows), while the genuinely
expensive operations — distinct counts and top-value tallies on wide text
columns — run on a bounded sample above `_SAMPLE_THRESHOLD` rows. When
sampling happens the profile says so, so no number is presented as exact when
it is an estimate.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.engine.semantics import classify_column, infer_time_grain
from app.engine.types import (
    ColumnProfile,
    ColumnStats,
    DatasetProfile,
    SemanticType,
)
from app.engine.utils import safe_float, to_jsonable
from app.obs.logging import get_logger
from app.obs.metrics import metrics

log = get_logger(__name__)

_SAMPLE_THRESHOLD = 150_000
_SAMPLE_SIZE = 100_000
_TOP_VALUES = 12
_SAMPLE_VALUES = 5


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {k: None for k in ("p01", "p05", "p25", "p75", "p95", "p99")}
    qs = np.percentile(values, [1, 5, 25, 75, 95, 99])
    return {
        "p01": safe_float(qs[0]),
        "p05": safe_float(qs[1]),
        "p25": safe_float(qs[2]),
        "p75": safe_float(qs[3]),
        "p95": safe_float(qs[4]),
        "p99": safe_float(qs[5]),
    }


def _numeric_stats(series: pd.Series, base: ColumnStats) -> ColumnStats:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return base

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = int(np.count_nonzero((values < lo) | (values > hi)))
    else:
        outliers = 0

    skew = kurt = None
    if values.size >= 8 and float(np.std(values)) > 0:
        skew = safe_float(scipy_stats.skew(values, bias=False))
        kurt = safe_float(scipy_stats.kurtosis(values, bias=False))

    updates: dict[str, Any] = {
        "min": safe_float(values.min()),
        "max": safe_float(values.max()),
        "mean": safe_float(values.mean()),
        "median": safe_float(np.median(values)),
        "std": safe_float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "variance": safe_float(values.var(ddof=1)) if values.size > 1 else 0.0,
        "sum": safe_float(values.sum()),
        "skewness": skew,
        "kurtosis": kurt,
        "zero_count": int(np.count_nonzero(values == 0)),
        "negative_count": int(np.count_nonzero(values < 0)),
        "outlier_count": outliers,
        **_percentiles(values),
    }
    return base.model_copy(update=updates)


def _datetime_stats(series: pd.Series, base: ColumnStats) -> ColumnStats:
    values = pd.to_datetime(series, errors="coerce").dropna()
    if values.empty:
        return base
    lo, hi = values.min(), values.max()
    span = int((hi - lo).days) if pd.notna(lo) and pd.notna(hi) else None
    return base.model_copy(
        update={
            "min_date": lo.isoformat() if pd.notna(lo) else None,
            "max_date": hi.isoformat() if pd.notna(hi) else None,
            "span_days": span,
            "inferred_grain": infer_time_grain(values),
        }
    )


def _categorical_stats(
    series: pd.Series, base: ColumnStats, *, total_rows: int
) -> ColumnStats:
    non_null = series.dropna()
    if non_null.empty:
        return base
    counts = non_null.value_counts().head(_TOP_VALUES)
    denominator = max(total_rows, 1)
    top_values = [
        {
            "value": to_jsonable(value),
            "count": int(count),
            "ratio": round(float(count) / denominator, 6),
        }
        for value, count in counts.items()
    ]
    lengths = non_null.astype(str).str.len()
    return base.model_copy(
        update={
            "top_values": top_values,
            "mode": to_jsonable(counts.index[0]) if len(counts) else None,
            "avg_length": safe_float(lengths.mean()),
            "max_length": int(lengths.max()) if not lengths.empty else None,
        }
    )


def profile_column(
    series: pd.Series,
    name: str,
    position: int,
    *,
    total_rows: int,
    sample: pd.Series | None = None,
) -> ColumnProfile:
    """Profile a single column. `sample` is used for cardinality work."""
    card_source = sample if sample is not None else series
    non_null = series.dropna()
    card_non_null = card_source.dropna()

    null_count = int(series.isna().sum())
    unique_count = int(card_non_null.nunique()) if not card_non_null.empty else 0
    # When profiling a sample, the distinct count is a lower bound, not an
    # estimate — extrapolating distinct counts is statistically unsound, so we
    # report what we measured and label it below.
    unique_count = min(unique_count, max(len(non_null), 1)) if len(non_null) else 0

    base = ColumnStats(
        count=int(len(non_null)),
        null_count=null_count,
        null_ratio=round(null_count / total_rows, 6) if total_rows else 0.0,
        unique_count=unique_count,
        unique_ratio=round(unique_count / len(non_null), 6) if len(non_null) else 0.0,
        sample_values=[
            to_jsonable(v) for v in card_non_null.drop_duplicates().head(_SAMPLE_VALUES)
        ],
    )

    classification = classify_column(card_source, name, row_count=total_rows)
    stype = classification.semantic_type

    if stype.is_numeric or stype == SemanticType.INTEGER:
        stats = _numeric_stats(series, base)
    elif stype == SemanticType.DATETIME:
        stats = _datetime_stats(series, base)
    elif pd.api.types.is_numeric_dtype(series):
        stats = _numeric_stats(series, base)
    else:
        stats = _categorical_stats(card_source, base, total_rows=total_rows)

    if stype == SemanticType.BOOLEAN:
        stats = _categorical_stats(card_source, stats, total_rows=total_rows)

    notes = list(classification.notes)
    if sample is not None:
        notes.append(
            f"Cardinality measured on a {len(card_source):,}-row sample; "
            "distinct counts are a lower bound."
        )
    if stats.null_ratio > 0.5:
        notes.append(f"{stats.null_ratio * 100:.1f}% of values are missing.")
    if stats.unique_count == 1 and stats.count > 1:
        notes.append("Every value is identical; this column carries no information.")

    return ColumnProfile(
        name=name,
        position=position,
        dtype=str(series.dtype),
        semantic_type=stype,
        role=classification.role,
        unit=classification.unit,  # type: ignore[arg-type]
        stats=stats,
        is_kpi_candidate=classification.is_kpi_candidate,
        notes=notes,
    )


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    """Build the full dataset profile."""
    started = time.perf_counter()
    total_rows = int(df.shape[0])
    total_cols = int(df.shape[1])

    sampled = total_rows > _SAMPLE_THRESHOLD
    sample_df = (
        df.sample(n=_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        if sampled
        else None
    )

    with metrics.timer("profile.duplicates"):
        try:
            duplicate_rows = int(df.duplicated().sum())
        except (TypeError, ValueError):
            # Unhashable cell values (nested lists from JSON) break duplicated().
            duplicate_rows = int(df.astype(str).duplicated().sum())

    columns: list[ColumnProfile] = []
    with metrics.timer("profile.columns", columns=total_cols):
        for position, name in enumerate(df.columns):
            columns.append(
                profile_column(
                    df[name],
                    str(name),
                    position,
                    total_rows=total_rows,
                    sample=sample_df[name] if sample_df is not None else None,
                )
            )

    missing_cells = int(sum(c.stats.null_count for c in columns))
    total_cells = total_rows * total_cols

    profile = DatasetProfile(
        row_count=total_rows,
        column_count=total_cols,
        memory_bytes=int(df.memory_usage(deep=False).sum()),
        duplicate_rows=duplicate_rows,
        duplicate_ratio=round(duplicate_rows / total_rows, 6) if total_rows else 0.0,
        total_cells=total_cells,
        missing_cells=missing_cells,
        missing_ratio=round(missing_cells / total_cells, 6) if total_cells else 0.0,
        columns=columns,
        profiled_rows=len(sample_df) if sample_df is not None else total_rows,
        sampled=sampled,
        profiling_ms=int((time.perf_counter() - started) * 1000),
    )
    log.info(
        "dataset_profiled",
        extra={
            "rows": total_rows,
            "columns": total_cols,
            "ms": profile.profiling_ms,
            "sampled": sampled,
        },
    )
    return profile
