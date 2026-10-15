"""Dataset cleaning — immutable by construction.

The original upload is version 1 and is never written to. Accepting fixes
produces a **new** DataFrame plus a transform log, which becomes a new
`dataset_versions` row. Rejecting a fix is simply not including its id.

Only the actions in `_HANDLERS` can ever run, and each one re-validates its own
arguments against the live frame, so a malformed (or model-proposed) fix payload
cannot do anything other than fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from app.engine.types import QualityIssue
from app.errors import BadRequestError
from app.obs.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class TransformRecord:
    action: str
    column: str | None
    detail: str
    rows_affected: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "column": self.column,
            "detail": self.detail,
            "rows_affected": self.rows_affected,
        }


@dataclass(slots=True)
class CleaningResult:
    df: pd.DataFrame
    transforms: list[TransformRecord] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.transforms)


# A handler takes the current frame plus a fix payload and returns the new frame
# and a record of what it did. Handlers are pure with respect to their input.
Handler = Callable[[pd.DataFrame, dict[str, Any]], tuple[pd.DataFrame, TransformRecord]]


def _require_column(df: pd.DataFrame, fix: dict[str, Any]) -> str:
    column = fix.get("column")
    if not isinstance(column, str) or column not in df.columns:
        raise BadRequestError(f"Column {column!r} is not present in this dataset.")
    return column


def _require_float(fix: dict[str, Any], key: str) -> float | None:
    value = fix.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BadRequestError(f"'{key}' must be a number, got {value!r}.") from exc


# ---------------------------------------------------------------- handlers ----
def _drop_duplicates(
    df: pd.DataFrame, fix: dict[str, Any]
) -> tuple[pd.DataFrame, TransformRecord]:
    before = len(df)
    try:
        out = df.drop_duplicates(keep="first")
    except TypeError:
        # Unhashable values (lists from nested JSON): compare stringified rows.
        out = df[~df.astype(str).duplicated(keep="first")]
    out = out.reset_index(drop=True)
    removed = before - len(out)
    return out, TransformRecord(
        "drop_duplicates", None, f"Removed {removed:,} exact duplicate rows.", removed
    )


def _fill_missing(
    df: pd.DataFrame, fix: dict[str, Any]
) -> tuple[pd.DataFrame, TransformRecord]:
    column = _require_column(df, fix)
    strategy = str(fix.get("strategy") or "median")
    series = df[column]
    missing = int(series.isna().sum())
    if missing == 0:
        return df, TransformRecord("fill_missing", column, "No missing values to fill.", 0)

    out = df.copy(deep=False)

    if strategy in {"median", "mean"}:
        numeric = pd.to_numeric(series, errors="coerce")
        value = numeric.median() if strategy == "median" else numeric.mean()
        if pd.isna(value):
            raise BadRequestError(
                f"Cannot compute the {strategy} of '{column}' — it has no numeric values."
            )
        out[column] = numeric.fillna(value)
        detail = f"Filled {missing:,} missing values with the {strategy} ({value:,.4g})."
    elif strategy == "mode":
        modes = series.dropna().mode()
        if modes.empty:
            raise BadRequestError(f"Column '{column}' has no values to take a mode from.")
        value = modes.iloc[0]
        out[column] = series.fillna(value)
        detail = (
            f"Filled {missing:,} missing values with the most common value ({value!r})."
        )
    elif strategy == "zero":
        out[column] = pd.to_numeric(series, errors="coerce").fillna(0)
        detail = f"Filled {missing:,} missing values with 0."
    elif strategy in {"ffill", "forward"}:
        out[column] = series.ffill()
        detail = f"Carried the previous value forward over {missing:,} gaps."
    elif strategy == "unknown":
        out[column] = series.astype("object").where(series.notna(), "Unknown")
        detail = f"Labelled {missing:,} missing values as 'Unknown'."
    else:
        raise BadRequestError(
            f"Unknown fill strategy '{strategy}'. Allowed: median, mean, mode, zero, "
            "ffill, unknown."
        )

    return out, TransformRecord("fill_missing", column, detail, missing)


def _normalise_categories(
    df: pd.DataFrame, fix: dict[str, Any]
) -> tuple[pd.DataFrame, TransformRecord]:
    column = _require_column(df, fix)
    mapping = fix.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise BadRequestError("normalise_categories requires a non-empty mapping.")
    if len(mapping) > 5_000:
        raise BadRequestError("Category mapping is too large (limit 5000 entries).")

    clean_map = {str(k): str(v) for k, v in mapping.items()}
    out = df.copy(deep=False)
    series = df[column].astype("object")
    affected = int(series.astype("string").isin(list(clean_map.keys())).sum())
    out[column] = series.map(
        lambda v: clean_map.get(str(v), v) if pd.notna(v) else v
    )
    return out, TransformRecord(
        "normalise_categories",
        column,
        f"Merged {len(clean_map)} spelling variants into "
        f"{len(set(clean_map.values()))} canonical value(s), affecting {affected:,} rows.",
        affected,
    )


def _trim_whitespace(
    df: pd.DataFrame, fix: dict[str, Any]
) -> tuple[pd.DataFrame, TransformRecord]:
    column = _require_column(df, fix)
    out = df.copy(deep=False)
    series = df[column].astype("string")
    cleaned = series.str.replace(r"\s+", " ", regex=True).str.strip()
    affected = int((series.fillna("") != cleaned.fillna("")).sum())
    out[column] = cleaned
    return out, TransformRecord(
        "trim_whitespace", column, f"Trimmed whitespace in {affected:,} values.", affected
    )


def _drop_column(
    df: pd.DataFrame, fix: dict[str, Any]
) -> tuple[pd.DataFrame, TransformRecord]:
    column = _require_column(df, fix)
    if df.shape[1] <= 1:
        raise BadRequestError("Cannot drop the only remaining column.")
    out = df.drop(columns=[column])
    return out, TransformRecord(
        "drop_column", column, f"Removed column '{column}'.", 0
    )


def _clip_range(
    df: pd.DataFrame, fix: dict[str, Any]
) -> tuple[pd.DataFrame, TransformRecord]:
    column = _require_column(df, fix)
    lower = _require_float(fix, "lower")
    upper = _require_float(fix, "upper")
    if lower is None and upper is None:
        raise BadRequestError("clip_range requires at least one of 'lower'/'upper'.")

    numeric = pd.to_numeric(df[column], errors="coerce")
    mask = pd.Series(False, index=numeric.index)
    if lower is not None:
        mask = mask | (numeric < lower)
    if upper is not None:
        mask = mask | (numeric > upper)
    affected = int(mask.fillna(False).sum())

    out = df.copy(deep=False)
    # Out-of-range values become missing rather than being squashed onto the
    # boundary: clamping would invent data that was never measured.
    out[column] = numeric.mask(mask.fillna(False))
    return out, TransformRecord(
        "clip_range",
        column,
        f"Blanked {affected:,} values outside [{lower}, {upper}].",
        affected,
    )


def _null_negatives(
    df: pd.DataFrame, fix: dict[str, Any]
) -> tuple[pd.DataFrame, TransformRecord]:
    column = _require_column(df, fix)
    numeric = pd.to_numeric(df[column], errors="coerce")
    mask = (numeric < 0).fillna(False)
    affected = int(mask.sum())
    out = df.copy(deep=False)
    out[column] = numeric.mask(mask)
    return out, TransformRecord(
        "null_negatives", column, f"Blanked {affected:,} negative values.", affected
    )


def _drop_missing_rows(
    df: pd.DataFrame, fix: dict[str, Any]
) -> tuple[pd.DataFrame, TransformRecord]:
    column = _require_column(df, fix)
    before = len(df)
    out = df[df[column].notna()].reset_index(drop=True)
    removed = before - len(out)
    if out.empty:
        raise BadRequestError(
            f"Dropping rows where '{column}' is empty would remove every row."
        )
    return out, TransformRecord(
        "drop_missing_rows",
        column,
        f"Removed {removed:,} rows where '{column}' was empty.",
        removed,
    )


_HANDLERS: dict[str, Handler] = {
    "drop_duplicates": _drop_duplicates,
    "fill_missing": _fill_missing,
    "normalise_categories": _normalise_categories,
    "trim_whitespace": _trim_whitespace,
    "drop_column": _drop_column,
    "clip_range": _clip_range,
    "null_negatives": _null_negatives,
    "drop_missing_rows": _drop_missing_rows,
}

AVAILABLE_ACTIONS: tuple[str, ...] = tuple(sorted(_HANDLERS))


def apply_fixes(
    df: pd.DataFrame,
    issues: list[QualityIssue],
    accepted_ids: list[str],
) -> CleaningResult:
    """Apply the accepted subset of `issues`, leaving `df` untouched."""
    working = df.copy(deep=True)
    result = CleaningResult(df=working)

    by_id = {issue.id: issue for issue in issues}
    seen_actions: set[str] = set()

    for issue_id in accepted_ids:
        issue = by_id.get(issue_id)
        if issue is None:
            result.skipped.append({"id": issue_id, "reason": "Unknown issue id."})
            continue
        if not issue.fix:
            result.skipped.append(
                {"id": issue_id, "reason": "This issue has no automatic fix."}
            )
            continue

        action = str(issue.fix.get("action", ""))
        handler = _HANDLERS.get(action)
        if handler is None:
            result.skipped.append(
                {"id": issue_id, "reason": f"Unsupported action '{action}'."}
            )
            continue

        signature = f"{action}:{issue.fix.get('column', '')}"
        if signature in seen_actions:
            result.skipped.append({"id": issue_id, "reason": "Already applied."})
            continue

        try:
            working, record = handler(working, issue.fix)
        except BadRequestError as exc:
            result.skipped.append({"id": issue_id, "reason": exc.message})
            continue
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "cleaning_action_failed",
                extra={"action": action, "error": str(exc)[:200]},
            )
            result.skipped.append(
                {"id": issue_id, "reason": f"Could not apply: {str(exc)[:120]}"}
            )
            continue

        seen_actions.add(signature)
        result.transforms.append(record)

    result.df = working
    log.info(
        "cleaning_applied",
        extra={
            "applied": len(result.transforms),
            "skipped": len(result.skipped),
            "rows": int(working.shape[0]),
        },
    )
    return result
