"""Statistical analysis: correlation, distribution and relationship strength.

All of this is classical statistics computed with NumPy/SciPy. Correlation is
reported with its p-value and sample size, and the narrative text never says
"causes" — a correlation insight that implies causation is a wrong insight, not
a stylistic choice.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.engine.query import resolve_column
from app.engine.types import DatasetProfile
from app.engine.utils import safe_float
from app.errors import AnalysisError, BadRequestError
from app.obs.logging import get_logger

log = get_logger(__name__)

Method = Literal["pearson", "spearman"]

_MIN_PAIRS = 10
_MAX_CORR_COLUMNS = 25


def _numeric_columns(df: pd.DataFrame, profile: DatasetProfile | None) -> list[str]:
    if profile is not None:
        cols = [
            c.name
            for c in profile.columns
            if c.semantic_type.is_numeric and c.stats.unique_count > 1
        ]
        if cols:
            return cols
    return [
        str(c)
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1
    ]


def describe_column(
    df: pd.DataFrame, column: str, *, bins: int = 20
) -> dict[str, Any]:
    """Full descriptive statistics plus a histogram for one column."""
    resolved = resolve_column(column, df)
    series = df[resolved]
    non_null = series.dropna()

    out: dict[str, Any] = {
        "column": resolved,
        "dtype": str(series.dtype),
        "count": int(len(non_null)),
        "missing": int(series.isna().sum()),
        "missing_pct": round(float(series.isna().mean() * 100), 3),
        "unique": int(non_null.nunique()) if not non_null.empty else 0,
    }

    if non_null.empty:
        out["note"] = "Column has no values."
        return out

    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(non_null, errors="coerce").dropna().to_numpy(dtype="float64")
        values = values[np.isfinite(values)]
        if values.size == 0:
            out["note"] = "Column has no finite numeric values."
            return out

        q = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
        out.update(
            {
                "kind": "numeric",
                "min": safe_float(values.min()),
                "max": safe_float(values.max()),
                "mean": safe_float(values.mean()),
                "median": safe_float(q[3]),
                "std": safe_float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "variance": safe_float(values.var(ddof=1)) if values.size > 1 else 0.0,
                "sum": safe_float(values.sum()),
                "range": safe_float(values.max() - values.min()),
                "iqr": safe_float(q[4] - q[2]),
                "cv": safe_float(values.std(ddof=1) / values.mean())
                if values.size > 1 and values.mean() != 0
                else None,
                "percentiles": {
                    "p01": safe_float(q[0]),
                    "p05": safe_float(q[1]),
                    "p25": safe_float(q[2]),
                    "p50": safe_float(q[3]),
                    "p75": safe_float(q[4]),
                    "p95": safe_float(q[5]),
                    "p99": safe_float(q[6]),
                },
            }
        )
        if values.size >= 8 and values.std() > 0:
            out["skewness"] = safe_float(scipy_stats.skew(values, bias=False))
            out["kurtosis"] = safe_float(scipy_stats.kurtosis(values, bias=False))
            out["distribution_shape"] = _describe_shape(out.get("skewness"), out.get("kurtosis"))
        out["histogram"] = histogram(values, bins=bins)
        return out

    if pd.api.types.is_datetime64_any_dtype(series):
        values = pd.to_datetime(non_null, errors="coerce").dropna()
        out.update(
            {
                "kind": "datetime",
                "min": values.min().isoformat(),
                "max": values.max().isoformat(),
                "span_days": int((values.max() - values.min()).days),
            }
        )
        return out

    counts = non_null.value_counts()
    total = int(counts.sum())
    out.update(
        {
            "kind": "categorical",
            "mode": str(counts.index[0]),
            "mode_count": int(counts.iloc[0]),
            "mode_share_pct": round(float(counts.iloc[0]) / total * 100, 2),
            "top_values": [
                {
                    "value": str(value),
                    "count": int(count),
                    "share_pct": round(float(count) / total * 100, 2),
                }
                for value, count in counts.head(20).items()
            ],
            "entropy": safe_float(scipy_stats.entropy(counts.to_numpy(dtype="float64"))),
        }
    )
    lengths = non_null.astype(str).str.len()
    out["avg_length"] = safe_float(lengths.mean())
    out["max_length"] = int(lengths.max())
    return out


def _describe_shape(skew: float | None, kurt: float | None) -> str:
    if skew is None:
        return "unknown"
    if abs(skew) < 0.5:
        shape = "approximately symmetric"
    elif skew > 1.0:
        shape = "strongly right-skewed (a few large values pull the mean up)"
    elif skew > 0.5:
        shape = "moderately right-skewed"
    elif skew < -1.0:
        shape = "strongly left-skewed"
    else:
        shape = "moderately left-skewed"
    if kurt is not None and kurt > 3:
        shape += ", heavy-tailed"
    return shape


def histogram(values: np.ndarray, *, bins: int = 20) -> list[dict[str, Any]]:
    """Histogram with the Freedman-Diaconis rule as the default bin count."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return []
    if values.size > 1:
        q25, q75 = np.percentile(values, [25, 75])
        iqr = q75 - q25
        if iqr > 0:
            width = 2 * iqr / (values.size ** (1 / 3))
            if width > 0:
                suggested = int(np.ceil((values.max() - values.min()) / width))
                bins = max(5, min(suggested, bins if bins > 5 else 30, 60))
    bins = max(1, min(int(bins), 100))

    counts, edges = np.histogram(values, bins=bins)
    total = max(int(counts.sum()), 1)
    return [
        {
            "bin": f"{_short(edges[i])} – {_short(edges[i + 1])}",
            "bin_start": safe_float(edges[i]),
            "bin_end": safe_float(edges[i + 1]),
            "count": int(counts[i]),
            "share_pct": round(float(counts[i]) / total * 100, 2),
        }
        for i in range(len(counts))
    ]


def _short(value: float) -> str:
    a = abs(value)
    if a >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if a >= 1_000:
        return f"{value / 1_000:.1f}K"
    if a >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}"


# -------------------------------------------------------------- correlation ---
def strength_label(r: float) -> str:
    a = abs(r)
    if a >= 0.9:
        return "very strong"
    if a >= 0.7:
        return "strong"
    if a >= 0.5:
        return "moderate"
    if a >= 0.3:
        return "weak"
    return "negligible"


def correlate_pair(
    df: pd.DataFrame, x: str, y: str, *, method: Method = "pearson"
) -> dict[str, Any]:
    """Correlation between two numeric columns, with significance."""
    xc, yc = resolve_column(x, df), resolve_column(y, df)
    if xc == yc:
        raise BadRequestError("Cannot correlate a column with itself.")

    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(df[xc], errors="coerce"),
            "y": pd.to_numeric(df[yc], errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if len(pair) < _MIN_PAIRS:
        raise AnalysisError(
            f"Only {len(pair)} rows have values for both '{xc}' and '{yc}'; "
            f"at least {_MIN_PAIRS} are needed for a correlation."
        )
    if pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        raise AnalysisError(
            f"'{xc}' or '{yc}' is constant over the overlapping rows, so no "
            "correlation can be computed."
        )

    if method == "spearman":
        result = scipy_stats.spearmanr(pair["x"], pair["y"])
        r, p = float(result.statistic), float(result.pvalue)
    else:
        result = scipy_stats.pearsonr(pair["x"], pair["y"])
        r, p = float(result.statistic), float(result.pvalue)

    n = len(pair)
    out: dict[str, Any] = {
        "x": xc,
        "y": yc,
        "method": method,
        "r": safe_float(round(r, 6)),
        "r_squared": safe_float(round(r * r, 6)),
        "p_value": safe_float(p),
        "n": n,
        "significant": bool(p < 0.05),
        "strength": strength_label(r),
        "direction": "positive" if r > 0 else ("negative" if r < 0 else "none"),
        # Stated explicitly so no downstream narrative can imply causation.
        "interpretation": (
            f"{strength_label(r)} {'positive' if r >= 0 else 'negative'} association "
            f"(r={r:.2f}, n={n:,}, p={'<0.001' if p < 0.001 else f'{p:.3f}'}). "
            "Association is not causation."
        ),
    }

    # Fisher z confidence interval for Pearson's r.
    if method == "pearson" and n > 3 and abs(r) < 1:
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        out["ci_95"] = [safe_float(round(float(lo), 4)), safe_float(round(float(hi), 4))]
    return out


def correlation_matrix(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    method: Method = "pearson",
    profile: DatasetProfile | None = None,
    min_abs: float = 0.0,
) -> dict[str, Any]:
    """Pairwise correlation over numeric columns, plus the ranked top pairs."""
    if columns:
        selected = [resolve_column(c, df) for c in columns]
    else:
        selected = _numeric_columns(df, profile)

    if len(selected) < 2:
        raise AnalysisError(
            "At least two numeric columns with variation are needed for a "
            "correlation analysis; this dataset has "
            f"{len(selected)}."
        )
    if len(selected) > _MAX_CORR_COLUMNS:
        # Keep the highest-variance columns so the matrix stays readable.
        variances = {
            c: float(pd.to_numeric(df[c], errors="coerce").var(ddof=1) or 0.0)
            for c in selected
        }
        selected = sorted(selected, key=lambda c: -variances[c])[:_MAX_CORR_COLUMNS]

    numeric = df[selected].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    matrix = numeric.corr(method=method, min_periods=_MIN_PAIRS)

    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(selected):
        for b in selected[i + 1 :]:
            r = matrix.loc[a, b]
            value = safe_float(r)
            if value is None or abs(value) < min_abs:
                continue
            overlap = int(numeric[[a, b]].dropna().shape[0])
            pairs.append(
                {
                    "x": a,
                    "y": b,
                    "r": round(value, 6),
                    "abs_r": round(abs(value), 6),
                    "strength": strength_label(value),
                    "direction": "positive" if value > 0 else "negative",
                    "n": overlap,
                }
            )
    pairs.sort(key=lambda p: -float(p["abs_r"]))

    heatmap: list[dict[str, Any]] = []
    for a in selected:
        row: dict[str, Any] = {"column": a}
        for b in selected:
            row[b] = safe_float(matrix.loc[a, b])
        heatmap.append(row)

    return {
        "method": method,
        "columns": selected,
        "matrix": heatmap,
        "top_pairs": pairs[:20],
        "pair_count": len(pairs),
        "note": (
            "Correlation measures co-movement only. A strong relationship here is a "
            "starting point for investigation, not evidence that one column causes "
            "the other."
        ),
    }


def group_comparison(
    df: pd.DataFrame, *, value_column: str, group_column: str, max_groups: int = 12
) -> dict[str, Any]:
    """Compare a measure across the levels of a dimension, with a significance
    test (Kruskal-Wallis — no normality assumption)."""
    vcol = resolve_column(value_column, df)
    gcol = resolve_column(group_column, df)
    if not pd.api.types.is_numeric_dtype(df[vcol]):
        raise BadRequestError(f"'{vcol}' must be numeric to compare across groups.")

    work = pd.DataFrame(
        {"value": pd.to_numeric(df[vcol], errors="coerce"), "group": df[gcol]}
    ).dropna()
    if work.empty:
        raise AnalysisError(f"No rows have both '{vcol}' and '{gcol}'.")

    top = work["group"].value_counts().head(max_groups).index
    work = work[work["group"].isin(top)]

    summary = (
        work.groupby("group", observed=True)["value"]
        .agg(["count", "sum", "mean", "median", "std"])
        .reset_index()
        .sort_values("sum", ascending=False)
    )
    groups = [g["value"].to_numpy() for _, g in work.groupby("group", observed=True)]
    test: dict[str, Any] = {"name": "kruskal-wallis", "applicable": False}
    if len(groups) >= 2 and all(len(g) >= 5 for g in groups):
        try:
            stat, p = scipy_stats.kruskal(*groups)
            test = {
                "name": "kruskal-wallis",
                "applicable": True,
                "statistic": safe_float(stat),
                "p_value": safe_float(p),
                "significant": bool(p < 0.05),
                "interpretation": (
                    f"Differences between groups are statistically significant (p="
                    f"{'<0.001' if p < 0.001 else f'{p:.3f}'})."
                    if p < 0.05
                    else f"Differences between groups are not statistically "
                    f"significant (p={p:.3f}); they may be noise."
                ),
            }
        except ValueError:
            pass

    rows = [
        {
            "group": str(row["group"]),
            "count": int(row["count"]),
            "sum": safe_float(row["sum"]),
            "mean": safe_float(row["mean"]),
            "median": safe_float(row["median"]),
            "std": safe_float(row["std"]),
        }
        for _, row in summary.iterrows()
    ]
    total = sum(r["sum"] or 0.0 for r in rows) or 1.0
    for r in rows:
        r["share_pct"] = round((r["sum"] or 0.0) / total * 100, 2)

    return {
        "value_column": vcol,
        "group_column": gcol,
        "groups": rows,
        "group_count": int(work["group"].nunique()),
        "test": test,
    }
