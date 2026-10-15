"""Data-quality engine.

Produces a 0-100 score, a letter grade, per-dimension sub-scores and a list of
concrete, individually-actionable issues. Every issue that *can* be fixed
mechanically carries a `fix` payload that :mod:`app.engine.cleaning` knows how
to apply — but nothing is applied automatically. The original upload is never
mutated; accepting fixes produces a new dataset version.

Quality dimensions (classic DQ framework, weighted for business analytics):

* completeness — are values present?
* uniqueness   — are rows distinct?
* validity     — are values possible given the column's meaning?
* consistency  — is the same thing spelled the same way?
* accuracy     — do values look plausible (outliers, impossible ranges)?
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from app.engine.semantics import CURRENCY_NAMES, NameHint
from app.engine.types import (
    ColumnRole,
    DatasetProfile,
    QualityIssue,
    QualityReport,
    SemanticType,
    Severity,
)
from app.engine.utils import safe_float, to_jsonable
from app.obs.logging import get_logger
from app.security import prompt_guard

log = get_logger(__name__)

_NON_NEGATIVE_NAMES = NameHint(
    "quantity", "qty", "unit", "count", "price", "amount", "cost", "revenue",
    "sales", "age", "stock", "inventory", "duration", "weight", "distance",
    "salary", "population", "view", "click", "day",
)
_AGE_NAMES = NameHint("age")
_EXPECTED_FUTURE_NAMES = NameHint(
    "due", "expiry", "expires", "target", "forecast", "renewal", "end",
    "deadline", "scheduled", "planned", "eta",
)
_AMPERSAND = re.compile(r"\s*&\s*")
_WHITESPACE_RE = re.compile(r"^\s|\s$|\s{2,}")

_DIMENSION_WEIGHTS = {
    "completeness": 0.30,
    "uniqueness": 0.15,
    "validity": 0.20,
    "consistency": 0.15,
    "accuracy": 0.20,
}

_MAX_ISSUES = 60
_CATEGORY_SCAN_LIMIT = 50_000


def _issue_id() -> str:
    return uuid.uuid4().hex[:12]


def _severity_for_ratio(ratio: float) -> Severity:
    if ratio >= 0.30:
        return Severity.CRITICAL
    if ratio >= 0.05:
        return Severity.WARNING
    return Severity.INFO


# ------------------------------------------------------------- individual ----
def _check_empty_and_constant(profile: DatasetProfile) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for col in profile.columns:
        if col.semantic_type == SemanticType.EMPTY:
            issues.append(
                QualityIssue(
                    id=_issue_id(),
                    rule="empty_column",
                    column=col.name,
                    severity=Severity.WARNING,
                    title=f"'{col.name}' is completely empty",
                    detail=(
                        f"All {profile.row_count:,} rows are missing a value for "
                        f"'{col.name}'. It cannot contribute to any analysis."
                    ),
                    affected_rows=profile.row_count,
                    affected_ratio=1.0,
                    suggestion=f"Remove '{col.name}' from the working dataset.",
                    fix={"action": "drop_column", "column": col.name},
                )
            )
        elif col.stats.unique_count == 1 and col.stats.count > 1:
            issues.append(
                QualityIssue(
                    id=_issue_id(),
                    rule="constant_column",
                    column=col.name,
                    severity=Severity.INFO,
                    title=f"'{col.name}' has a single repeated value",
                    detail=(
                        f"Every non-empty row holds {col.stats.mode!r}. A constant "
                        "column cannot explain variation in any metric."
                    ),
                    affected_rows=col.stats.count,
                    affected_ratio=1.0,
                    examples=[to_jsonable(col.stats.mode)],
                    suggestion=f"Exclude '{col.name}' from grouping and correlation.",
                    fix={"action": "drop_column", "column": col.name},
                )
            )
    return issues


def _check_missing(profile: DatasetProfile) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for col in profile.columns:
        ratio = col.stats.null_ratio
        if ratio <= 0 or col.semantic_type == SemanticType.EMPTY:
            continue
        if ratio < 0.001 and col.stats.null_count < 5:
            continue  # negligible
        strategy = (
            "median"
            if col.semantic_type.is_numeric
            else ("mode" if col.role == ColumnRole.DIMENSION else None)
        )
        issues.append(
            QualityIssue(
                id=_issue_id(),
                rule="missing_values",
                column=col.name,
                severity=_severity_for_ratio(ratio),
                title=f"'{col.name}' is missing {ratio * 100:.1f}% of values",
                detail=(
                    f"{col.stats.null_count:,} of {profile.row_count:,} rows have no "
                    f"value for '{col.name}'. Aggregations over this column "
                    "silently skip those rows."
                ),
                affected_rows=col.stats.null_count,
                affected_ratio=round(ratio, 6),
                suggestion=(
                    f"Fill with the column {strategy}, or exclude rows where "
                    f"'{col.name}' is empty."
                    if strategy
                    else f"Decide whether rows missing '{col.name}' should be excluded."
                ),
                fix=(
                    {"action": "fill_missing", "column": col.name, "strategy": strategy}
                    if strategy
                    else None
                ),
            )
        )
    return issues


def _check_duplicates(profile: DatasetProfile) -> list[QualityIssue]:
    if profile.duplicate_rows <= 0:
        return []
    ratio = profile.duplicate_ratio
    return [
        QualityIssue(
            id=_issue_id(),
            rule="duplicate_rows",
            column=None,
            severity=_severity_for_ratio(ratio),
            title=f"{profile.duplicate_rows:,} duplicate rows",
            detail=(
                f"{profile.duplicate_rows:,} rows ({ratio * 100:.2f}%) are exact copies "
                "of an earlier row. Totals and counts are inflated by these."
            ),
            affected_rows=profile.duplicate_rows,
            affected_ratio=round(ratio, 6),
            suggestion="Remove exact duplicate rows, keeping the first occurrence.",
            fix={"action": "drop_duplicates"},
        )
    ]


def _check_inconsistent_categories(
    df: pd.DataFrame, profile: DatasetProfile
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    scan = df.head(_CATEGORY_SCAN_LIMIT)

    for col in profile.columns:
        if col.semantic_type not in {SemanticType.CATEGORICAL, SemanticType.GEO}:
            continue
        if col.stats.unique_count > 2000:
            continue
        series = scan[col.name].dropna().astype(str)
        if series.empty:
            continue

        counts = series.value_counts()
        groups: dict[str, list[str]] = defaultdict(list)
        for raw in counts.index:
            text = _AMPERSAND.sub(" and ", str(raw))
            canonical = re.sub(r"[\s_\-]+", " ", text).strip().casefold()
            if canonical:
                groups[canonical].append(str(raw))

        mapping: dict[str, str] = {}
        examples: list[Any] = []
        affected = 0
        for variants in groups.values():
            if len(variants) < 2:
                continue
            winner = max(variants, key=lambda v: (int(counts.get(v, 0)), -len(v)))
            for variant in variants:
                if variant != winner:
                    mapping[variant] = winner
                    affected += int(counts.get(variant, 0))
            if len(examples) < 6:
                examples.append(" | ".join(variants[:4]))

        if mapping:
            ratio = affected / max(profile.row_count, 1)
            issues.append(
                QualityIssue(
                    id=_issue_id(),
                    rule="inconsistent_categories",
                    column=col.name,
                    severity=Severity.WARNING if ratio > 0.01 else Severity.INFO,
                    title=(
                        f"'{col.name}' has {len(mapping)} spelling variants of "
                        f"{len(set(mapping.values()))} value(s)"
                    ),
                    detail=(
                        "The same category is written several ways, so grouping "
                        "splits one real category into multiple rows. Variants found: "
                        + "; ".join(examples[:3])
                    ),
                    affected_rows=affected,
                    affected_ratio=round(ratio, 6),
                    examples=examples,
                    suggestion=(
                        f"Normalise '{col.name}' so each category has one spelling."
                    ),
                    fix={
                        "action": "normalise_categories",
                        "column": col.name,
                        "mapping": mapping,
                    },
                )
            )
    return issues


def _check_whitespace(df: pd.DataFrame, profile: DatasetProfile) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    scan = df.head(_CATEGORY_SCAN_LIMIT)
    for col in profile.columns:
        if col.semantic_type not in {
            SemanticType.CATEGORICAL,
            SemanticType.TEXT,
            SemanticType.GEO,
            SemanticType.IDENTIFIER,
        }:
            continue
        series = scan[col.name].dropna().astype(str)
        if series.empty:
            continue
        bad = series.str.contains(_WHITESPACE_RE, regex=True, na=False)
        count = int(bad.sum())
        if count == 0:
            continue
        ratio = count / max(len(series), 1)
        issues.append(
            QualityIssue(
                id=_issue_id(),
                rule="formatting_whitespace",
                column=col.name,
                severity=Severity.INFO,
                title=f"'{col.name}' has {count:,} values with stray whitespace",
                detail=(
                    "Leading, trailing or doubled spaces make otherwise identical "
                    "values group separately."
                ),
                affected_rows=count,
                affected_ratio=round(ratio, 6),
                examples=[repr(v) for v in series[bad].head(4).tolist()],
                suggestion=f"Trim and collapse whitespace in '{col.name}'.",
                fix={"action": "trim_whitespace", "column": col.name},
            )
        )
    return issues


def _check_impossible_values(
    df: pd.DataFrame, profile: DatasetProfile
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []

    for col in profile.columns:
        if not col.semantic_type.is_numeric:
            continue
        series = pd.to_numeric(df[col.name], errors="coerce").dropna()
        if series.empty:
            continue

        # Negative values in a quantity-like column.
        if _NON_NEGATIVE_NAMES.search(col.name) and not CURRENCY_NAMES.search(col.name):
            negatives = int((series < 0).sum())
            if negatives:
                ratio = negatives / max(profile.row_count, 1)
                issues.append(
                    QualityIssue(
                        id=_issue_id(),
                        rule="impossible_negative",
                        column=col.name,
                        severity=Severity.WARNING,
                        title=f"'{col.name}' has {negatives:,} negative values",
                        detail=(
                            f"'{col.name}' reads as a non-negative quantity, but "
                            f"{negatives:,} rows are below zero (minimum "
                            f"{series.min():,.2f}). These are usually returns coded "
                            "incorrectly or sign errors."
                        ),
                        affected_rows=negatives,
                        affected_ratio=round(ratio, 6),
                        examples=[safe_float(v) for v in series[series < 0].head(4)],
                        suggestion=(
                            f"Confirm whether negatives in '{col.name}' are returns. "
                            "If not, blank them so they do not distort totals."
                        ),
                        fix={"action": "null_negatives", "column": col.name},
                    )
                )
        elif CURRENCY_NAMES.search(col.name):
            negatives = int((series < 0).sum())
            if negatives and negatives / max(len(series), 1) > 0.25:
                issues.append(
                    QualityIssue(
                        id=_issue_id(),
                        rule="negative_amounts",
                        column=col.name,
                        severity=Severity.INFO,
                        title=f"'{col.name}' is negative in {negatives:,} rows",
                        detail=(
                            "More than a quarter of amounts are negative. If these "
                            "are refunds, totals mix charges and refunds together."
                        ),
                        affected_rows=negatives,
                        affected_ratio=round(negatives / max(profile.row_count, 1), 6),
                        suggestion=(
                            "Consider splitting refunds into their own metric before "
                            "reading totals."
                        ),
                    )
                )

        # Percentages outside a plausible range.
        if (
            col.unit in {"percent", "ratio"}
            or col.semantic_type == SemanticType.PERCENTAGE
        ):
            upper = 1.0 if col.unit == "ratio" else 100.0
            bad = int(((series < 0) | (series > upper)).sum())
            if bad:
                issues.append(
                    QualityIssue(
                        id=_issue_id(),
                        rule="impossible_percentage",
                        column=col.name,
                        severity=Severity.WARNING,
                        title=f"'{col.name}' has {bad:,} values outside 0-{upper:g}",
                        detail=(
                            f"'{col.name}' looks like a rate on a 0-{upper:g} scale, "
                            f"but {bad:,} rows fall outside it (range "
                            f"{series.min():,.2f} to {series.max():,.2f})."
                        ),
                        affected_rows=bad,
                        affected_ratio=round(bad / max(profile.row_count, 1), 6),
                        examples=[
                            safe_float(v)
                            for v in series[(series < 0) | (series > upper)].head(4)
                        ],
                        suggestion=(
                            f"Check whether '{col.name}' mixes fractions and "
                            "percentages in the same column."
                        ),
                        fix={
                            "action": "clip_range",
                            "column": col.name,
                            "lower": 0.0,
                            "upper": upper,
                        },
                    )
                )

        # Implausible ages.
        if _AGE_NAMES.search(col.name):
            bad = int(((series < 0) | (series > 120)).sum())
            if bad:
                issues.append(
                    QualityIssue(
                        id=_issue_id(),
                        rule="impossible_age",
                        column=col.name,
                        severity=Severity.WARNING,
                        title=f"'{col.name}' has {bad:,} implausible ages",
                        detail=f"{bad:,} rows are below 0 or above 120 years.",
                        affected_rows=bad,
                        affected_ratio=round(bad / max(profile.row_count, 1), 6),
                        suggestion=f"Treat out-of-range '{col.name}' values as missing.",
                        fix={
                            "action": "clip_range",
                            "column": col.name,
                            "lower": 0.0,
                            "upper": 120.0,
                        },
                    )
                )
    return issues


def _tukey_outliers(values: np.ndarray, *, k: float = 3.0) -> tuple[int, float, float]:
    """Count values beyond Tukey's *far-out* fences (Q1 - k·IQR, Q3 + k·IQR).

    k=3 rather than 1.5: the quality score should penalise values that are
    implausible, not merely the top few percent of an ordinary distribution.
    """
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return 0, float(q1), float(q3)
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return int(np.count_nonzero((values < lo) | (values > hi))), float(lo), float(hi)


def _check_outliers(df: pd.DataFrame, profile: DatasetProfile) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for col in profile.columns:
        if not col.semantic_type.is_numeric or col.role != ColumnRole.MEASURE:
            continue
        if col.stats.count < 20:
            continue

        values = pd.to_numeric(df[col.name], errors="coerce").dropna().to_numpy("float64")
        values = values[np.isfinite(values)]
        if values.size < 20:
            continue

        # Money, quantities and durations are naturally long-tailed: a linear
        # fence flags the entire tail. For a right-skewed, non-negative measure,
        # test on log scale, where "unusual" means unusual in relative terms.
        skewed = (col.stats.skewness or 0) > 1.0 and float(values.min()) >= 0
        if skewed:
            count, lo, hi = _tukey_outliers(np.log1p(values))
            lo, hi = float(np.expm1(lo)), float(np.expm1(hi))
            basis = "log scale (the distribution is right-skewed)"
        else:
            count, lo, hi = _tukey_outliers(values)
            basis = "linear scale"

        if count <= 0:
            continue
        ratio = count / max(values.size, 1)
        if ratio < 0.0005 and count < 3:
            continue

        severity = Severity.WARNING if ratio > 0.01 else Severity.INFO
        examples = values[(values < lo) | (values > hi)]
        issues.append(
            QualityIssue(
                id=_issue_id(),
                rule="outliers",
                column=col.name,
                severity=severity,
                title=f"'{col.name}' has {count:,} extreme values",
                detail=(
                    f"{count:,} values ({ratio * 100:.2f}%) fall outside Tukey's far-out "
                    f"fences [{lo:,.2f}, {hi:,.2f}], measured on a {basis}. Means and "
                    "growth rates over this column are sensitive to them."
                ),
                affected_rows=count,
                affected_ratio=round(ratio, 6),
                examples=[safe_float(v) for v in np.sort(examples)[-4:]],
                suggestion=(
                    f"Check the most extreme '{col.name}' values against source "
                    "records; prefer the median over the mean while they remain."
                ),
            )
        )
    return issues


def _check_dates(df: pd.DataFrame, profile: DatasetProfile) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    now = pd.Timestamp.now().normalize()
    for col in profile.columns:
        if col.semantic_type != SemanticType.DATETIME:
            continue
        series = pd.to_datetime(df[col.name], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue

        # Values that failed to parse while most of the column did.
        unparsed = int(series.isna().sum() - df[col.name].isna().sum())
        if unparsed > 0:
            issues.append(
                QualityIssue(
                    id=_issue_id(),
                    rule="invalid_dates",
                    column=col.name,
                    severity=Severity.WARNING,
                    title=f"'{col.name}' has {unparsed:,} unparseable dates",
                    detail=(
                        f"{unparsed:,} values in '{col.name}' are present but are not "
                        "recognisable dates, so those rows drop out of any "
                        "time-based analysis."
                    ),
                    affected_rows=unparsed,
                    affected_ratio=round(unparsed / max(profile.row_count, 1), 6),
                    suggestion=f"Standardise '{col.name}' to ISO format (YYYY-MM-DD).",
                )
            )

        future = int((valid > now).sum())
        if future and not _EXPECTED_FUTURE_NAMES.search(col.name):
            issues.append(
                QualityIssue(
                    id=_issue_id(),
                    rule="future_dates",
                    column=col.name,
                    severity=Severity.WARNING if future / len(valid) > 0.01 else Severity.INFO,
                    title=f"'{col.name}' has {future:,} dates in the future",
                    detail=(
                        f"{future:,} rows carry a '{col.name}' later than today "
                        f"(latest {valid.max().date()}). For a transactional date "
                        "this usually means a data-entry or timezone error."
                    ),
                    affected_rows=future,
                    affected_ratio=round(future / max(profile.row_count, 1), 6),
                    examples=[str(v.date()) for v in valid[valid > now].head(4)],
                    suggestion=f"Verify how '{col.name}' is populated upstream.",
                )
            )

        ancient = int((valid < pd.Timestamp("1900-01-01")).sum())
        if ancient:
            issues.append(
                QualityIssue(
                    id=_issue_id(),
                    rule="implausible_dates",
                    column=col.name,
                    severity=Severity.INFO,
                    title=f"'{col.name}' has {ancient:,} dates before 1900",
                    detail=(
                        "Dates before 1900 are usually placeholders such as "
                        "1970-01-01 (Unix epoch) or 1900-01-00 (Excel zero date)."
                    ),
                    affected_rows=ancient,
                    affected_ratio=round(ancient / max(profile.row_count, 1), 6),
                    suggestion=f"Treat pre-1900 '{col.name}' values as missing.",
                )
            )
    return issues


def _check_high_cardinality(profile: DatasetProfile) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for col in profile.columns:
        if col.role != ColumnRole.DIMENSION:
            continue
        if col.stats.unique_count > 1000 and col.stats.unique_ratio > 0.5:
            issues.append(
                QualityIssue(
                    id=_issue_id(),
                    rule="high_cardinality_dimension",
                    column=col.name,
                    severity=Severity.INFO,
                    title=f"'{col.name}' has {col.stats.unique_count:,} distinct values",
                    detail=(
                        "A dimension this granular produces charts with one bar per "
                        "row. Grouping by it is unlikely to be readable."
                    ),
                    affected_rows=col.stats.unique_count,
                    affected_ratio=round(col.stats.unique_ratio, 6),
                    suggestion=(
                        f"Group '{col.name}' into buckets, or slice by a coarser "
                        "column instead."
                    ),
                )
            )
    return issues


def _check_prompt_injection(
    df: pd.DataFrame, profile: DatasetProfile
) -> tuple[list[QualityIssue], list[dict[str, Any]]]:
    """Scan text columns for content trying to steer the assistant.

    The cells are never treated as instructions regardless; flagging them tells
    the user their file has been tampered with.
    """
    findings: list[dict[str, Any]] = []
    scan = df.head(20_000)
    for col in profile.columns:
        if col.semantic_type not in {
            SemanticType.TEXT,
            SemanticType.CATEGORICAL,
            SemanticType.URL,
        }:
            continue
        series = scan[col.name].dropna().astype(str)
        if series.empty:
            continue
        # Only long-ish values can plausibly carry an injection payload.
        candidates = series[series.str.len() > 24].head(5_000)
        if candidates.empty:
            continue
        hits = prompt_guard.scan_values(candidates.tolist(), column=col.name, limit=3)
        findings.extend(f.as_dict() for f in hits)

    issues: list[QualityIssue] = []
    if findings:
        issues.append(
            QualityIssue(
                id=_issue_id(),
                rule="prompt_injection_content",
                column=findings[0].get("location"),
                severity=Severity.CRITICAL,
                title=f"{len(findings)} cell(s) contain assistant-directed instructions",
                detail=(
                    "Some cells contain text written to manipulate an AI assistant "
                    "(for example 'ignore previous instructions'). InsightOS always "
                    "treats file contents as data and never executes instructions "
                    "found inside them, but you should check where this file came "
                    "from."
                ),
                affected_rows=len(findings),
                affected_ratio=0.0,
                examples=[f["excerpt"] for f in findings[:4]],
                suggestion="Verify the source of this file before sharing results.",
            )
        )
    return issues, findings


# ------------------------------------------------------------------ scoring ---
def _dimension_scores(
    profile: DatasetProfile, issues: list[QualityIssue]
) -> dict[str, float]:
    rows = max(profile.row_count, 1)

    completeness = 100.0 * (1.0 - profile.missing_ratio)
    uniqueness = 100.0 * (1.0 - min(profile.duplicate_ratio * 2.0, 1.0))

    def penalty(rules: set[str], *, weight: float) -> float:
        total = 0.0
        for issue in issues:
            if issue.rule not in rules:
                continue
            severity_factor = {
                Severity.CRITICAL: 1.0,
                Severity.WARNING: 0.6,
                Severity.INFO: 0.25,
                Severity.POSITIVE: 0.0,
            }[issue.severity]
            share = min(issue.affected_ratio if issue.affected_ratio else
                        issue.affected_rows / rows, 1.0)
            total += weight * severity_factor * (0.25 + 0.75 * share)
        return min(total, 100.0)

    validity = 100.0 - penalty(
        {
            "impossible_negative",
            "impossible_percentage",
            "impossible_age",
            "invalid_dates",
            "future_dates",
            "implausible_dates",
            "empty_column",
        },
        weight=35.0,
    )
    consistency = 100.0 - penalty(
        {
            "inconsistent_categories",
            "formatting_whitespace",
            "constant_column",
            "high_cardinality_dimension",
        },
        weight=30.0,
    )
    accuracy = 100.0 - penalty(
        {"outliers", "negative_amounts", "prompt_injection_content"}, weight=30.0
    )

    return {
        "completeness": round(max(0.0, min(100.0, completeness)), 1),
        "uniqueness": round(max(0.0, min(100.0, uniqueness)), 1),
        "validity": round(max(0.0, min(100.0, validity)), 1),
        "consistency": round(max(0.0, min(100.0, consistency)), 1),
        "accuracy": round(max(0.0, min(100.0, accuracy)), 1),
    }


def _grade(score: float) -> str:
    if score >= 95:
        return "A+"
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 55:
        return "D"
    return "E"


_SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.WARNING: 1,
    Severity.INFO: 2,
    Severity.POSITIVE: 3,
}


def assess_quality(df: pd.DataFrame, profile: DatasetProfile) -> QualityReport:
    """Run every quality rule and compute the composite score."""
    issues: list[QualityIssue] = []
    issues += _check_empty_and_constant(profile)
    issues += _check_missing(profile)
    issues += _check_duplicates(profile)
    issues += _check_inconsistent_categories(df, profile)
    issues += _check_whitespace(df, profile)
    issues += _check_impossible_values(df, profile)
    issues += _check_outliers(df, profile)
    issues += _check_dates(df, profile)
    issues += _check_high_cardinality(profile)

    injection_issues, injection_findings = _check_prompt_injection(df, profile)
    issues += injection_issues

    issues.sort(
        key=lambda i: (_SEVERITY_ORDER[i.severity], -i.affected_ratio, i.rule)
    )
    issues = issues[:_MAX_ISSUES]

    dimensions = _dimension_scores(profile, issues)
    score = sum(dimensions[k] * w for k, w in _DIMENSION_WEIGHTS.items())
    score = round(max(0.0, min(100.0, score)), 1)

    log.info(
        "quality_assessed",
        extra={"score": score, "issues": len(issues), "rows": profile.row_count},
    )
    return QualityReport(
        score=score,
        grade=_grade(score),
        issues=issues,
        dimension_scores=dimensions,
        injection_findings=injection_findings[:20],
        checked_rules=9,
    )
