"""File ingestion: bytes -> a typed, bounded pandas DataFrame.

Design constraints that shape this module:

* **Bounded.** Rows, columns and individual cell length are capped before the
  frame is materialised, so a hostile or merely enormous file cannot exhaust
  the free-tier container's memory.
* **Honest.** When a cap truncates data, the result says so; the row count the
  user sees is the row count that was analysed.
* **Type-recovering.** Real-world exports arrive as strings: `"₹1,23,456.00"`,
  `"45%"`, `"(1,200)"`, `"2026-03-01"`, `"01/03/2026"`. Inference runs on a
  sample, decides per column, then applies once — which keeps it fast on
  250k rows.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.config import settings
from app.errors import ValidationError
from app.obs.logging import get_logger
from app.obs.metrics import metrics
from app.security.file_validation import ValidatedUpload, decode_text

log = get_logger(__name__)

_INFER_SAMPLE = 4_000
_NUMERIC_CONFIDENCE = 0.90
_DATE_CONFIDENCE = 0.85
_BOOL_TRUE = {"true", "yes", "y", "t", "1", "on"}
_BOOL_FALSE = {"false", "no", "n", "f", "0", "off"}

# Currency/grouping noise that must be stripped before numeric coercion.
_NUMERIC_NOISE = re.compile(r"[₹$€£¥,\s_]")
_PARENTHESISED = re.compile(r"^\((.*)\)$")
_PERCENT_SUFFIX = re.compile(r"%\s*$")

_DATE_HINT_NAME = re.compile(
    r"(date|time|day|month|year|week|quarter|timestamp|_at$|_on$|dt$|period)",
    re.IGNORECASE,
)
_DATE_SHAPE = re.compile(
    r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"          # 2026-03-01, 01/03/2026
    r"|^\s*\d{4}[-/]\d{2}\s*$"                        # 2026-03
    r"|^\s*\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}"         # 1 March 2026
    r"|^\s*[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}"       # March 1, 2026
    r"|^\s*\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"          # ISO timestamp
)


@dataclass(slots=True)
class IngestResult:
    df: pd.DataFrame
    source_rows: int
    truncated_rows: bool = False
    truncated_columns: bool = False
    dropped_columns: list[str] = field(default_factory=list)
    sheet_names: list[str] = field(default_factory=list)
    selected_sheet: str | None = None
    delimiter: str | None = None
    encoding: str | None = None
    notes: list[str] = field(default_factory=list)
    conversions: dict[str, str] = field(default_factory=dict)


# ------------------------------------------------------------------ headers ---
def _normalise_headers(columns: list[Any]) -> tuple[list[str], list[str]]:
    """Trim, de-blank and de-duplicate column names."""
    notes: list[str] = []
    out: list[str] = []
    seen: dict[str, int] = {}
    for idx, raw in enumerate(columns):
        name = "" if raw is None else str(raw)
        name = re.sub(r"\s+", " ", name.replace("﻿", "")).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{idx + 1}"
            notes.append(f"Unnamed column at position {idx + 1} renamed to '{name}'.")
        if len(name) > 200:
            name = name[:200]
        key = name.lower()
        if key in seen:
            seen[key] += 1
            duplicate = f"{name}_{seen[key]}"
            notes.append(f"Duplicate column '{name}' renamed to '{duplicate}'.")
            name = duplicate
        else:
            seen[key] = 0
        out.append(name)
    return out, notes


def _looks_headerless(columns: list[Any]) -> bool:
    """True when the first row is clearly data rather than a header."""
    if not columns:
        return False
    unnamed = sum(
        1 for c in columns if c is None or str(c).strip() == "" or str(c).startswith("Unnamed:")
    )
    if unnamed > len(columns) / 2:
        return True
    numericish = 0
    for c in columns:
        try:
            float(str(c).replace(",", ""))
            numericish += 1
        except (TypeError, ValueError):
            pass
    return numericish == len(columns) and len(columns) > 1


# --------------------------------------------------------------- delimiters ---
def sniff_delimiter(text: str, *, default: str = ",") -> str:
    sample = text[:64_000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return str(dialect.delimiter)
    except csv.Error:
        pass
    first_lines = [ln for ln in sample.splitlines()[:20] if ln.strip()]
    if not first_lines:
        return default
    best, best_score = default, -1.0
    for candidate in (",", ";", "\t", "|"):
        counts = [ln.count(candidate) for ln in first_lines]
        if not counts or max(counts) == 0:
            continue
        # Prefer a delimiter that appears a consistent number of times per line.
        score = float(np.mean(counts)) - float(np.std(counts)) * 2
        if score > best_score:
            best, best_score = candidate, score
    return best


# ------------------------------------------------------------ type recovery ---
def _clean_numeric_token(token: str) -> str | None:
    t = token.strip()
    if not t:
        return None
    negative = False
    paren = _PARENTHESISED.match(t)
    if paren:  # accounting negatives: (1,200)
        negative = True
        t = paren.group(1)
    percent = bool(_PERCENT_SUFFIX.search(t))
    t = _PERCENT_SUFFIX.sub("", t)
    t = _NUMERIC_NOISE.sub("", t)
    if t in {"", "-", "+", ".", "-.", "NA", "N/A", "null", "None", "nan"}:
        return None
    if not re.fullmatch(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", t):
        return None
    if negative and not t.startswith("-"):
        t = "-" + t
    if percent:
        try:
            return str(float(t))
        except ValueError:
            return None
    return t


def _try_numeric(series: pd.Series) -> tuple[pd.Series | None, bool]:
    """Attempt numeric coercion. Returns (converted, was_percent)."""
    sample = series.dropna().astype(str)
    if sample.empty:
        return None, False
    sample = sample.head(_INFER_SAMPLE)
    cleaned = sample.map(_clean_numeric_token)
    ok = cleaned.notna().mean()
    if ok < _NUMERIC_CONFIDENCE:
        return None, False
    was_percent = bool(sample.str.contains("%", regex=False).mean() > 0.5)

    full = series.astype("string").map(_clean_numeric_token)
    converted = pd.to_numeric(full, errors="coerce")
    if was_percent:
        converted = converted  # values kept as-is (45% -> 45), unit flagged later
    if converted.notna().sum() == 0:
        return None, False
    # Prefer a nullable integer dtype when every value is whole.
    non_null = converted.dropna()
    if not non_null.empty and (non_null % 1 == 0).all() and non_null.abs().max() < 2**53:
        try:
            return converted.astype("Int64"), was_percent
        except (TypeError, ValueError):
            return converted, was_percent
    return converted, was_percent


def _try_datetime(series: pd.Series, column_name: str) -> pd.Series | None:
    sample = series.dropna().astype(str).head(_INFER_SAMPLE)
    if sample.empty:
        return None

    name_hint = bool(_DATE_HINT_NAME.search(column_name))
    shape_hits = float(sample.str.match(_DATE_SHAPE).mean())
    # Without a name hint, demand that the values *look* like dates. This stops
    # order ids and postcodes being silently reinterpreted as timestamps.
    if not name_hint and shape_hits < _DATE_CONFIDENCE:
        return None
    if name_hint and shape_hits < 0.5:
        # Name says date but values do not look like dates; try anyway but
        # require a high parse rate below.
        pass

    best: pd.Series | None = None
    best_rate = 0.0
    best_dayfirst = False
    for dayfirst in (False, True):
        with pd.option_context("mode.chained_assignment", None):
            try:
                parsed = pd.to_datetime(
                    sample, errors="coerce", dayfirst=dayfirst, format="mixed"
                )
            except (ValueError, TypeError):
                try:
                    parsed = pd.to_datetime(sample, errors="coerce", dayfirst=dayfirst)
                except (ValueError, TypeError):
                    continue
        rate = float(parsed.notna().mean())
        if rate > best_rate:
            best_rate, best = rate, parsed
            best_dayfirst = dayfirst

    if best is None or best_rate < _DATE_CONFIDENCE:
        return None

    try:
        full = pd.to_datetime(
            series, errors="coerce", dayfirst=best_dayfirst, format="mixed"
        )
    except (ValueError, TypeError):
        full = pd.to_datetime(series, errors="coerce", dayfirst=best_dayfirst)
    if full.notna().sum() == 0:
        return None
    # Normalise away timezones so downstream resampling is comparable.
    if isinstance(full.dtype, pd.DatetimeTZDtype):
        full = full.dt.tz_convert("UTC").dt.tz_localize(None)
    return full


def _try_boolean(series: pd.Series) -> pd.Series | None:
    sample = series.dropna().astype(str).str.strip().str.lower().head(_INFER_SAMPLE)
    if sample.empty:
        return None
    vocab = set(sample.unique())
    if not vocab or not vocab <= (_BOOL_TRUE | _BOOL_FALSE):
        return None
    if len(vocab) < 2 and len(sample) > 20:
        return None  # a single repeated token is a constant, not a boolean

    def to_bool(v: Any) -> Any:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return pd.NA
        s = str(v).strip().lower()
        if s in _BOOL_TRUE:
            return True
        if s in _BOOL_FALSE:
            return False
        return pd.NA

    return series.map(to_bool).astype("boolean")


def infer_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Recover real dtypes from string columns. Returns (df, conversions)."""
    conversions: dict[str, str] = {}
    for col in df.columns:
        series = df[col]
        if not (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.StringDtype)
        ):
            continue
        if series.dropna().empty:
            continue

        as_bool = _try_boolean(series)
        if as_bool is not None:
            df[col] = as_bool
            conversions[col] = "boolean"
            continue

        as_dt = _try_datetime(series, str(col))
        if as_dt is not None:
            df[col] = as_dt
            conversions[col] = "datetime"
            continue

        as_num, was_percent = _try_numeric(series)
        if as_num is not None:
            df[col] = as_num
            conversions[col] = "percentage" if was_percent else "numeric"
            continue

        # Keep as text, but bound cell length so one giant cell cannot blow up
        # memory or a downstream prompt.
        trimmed = series.astype("string").str.slice(0, settings.max_cell_length)
        df[col] = trimmed
        conversions[col] = "text"
    return df, conversions


# -------------------------------------------------------------- CSV / TSV -----
def _read_delimited(upload: ValidatedUpload, result_notes: list[str]) -> tuple[pd.DataFrame, str, str]:
    text, encoding = decode_text(upload.content)
    delimiter = "\t" if upload.kind == "tsv" else sniff_delimiter(text)

    read_kwargs: dict[str, Any] = {
        "sep": delimiter,
        "dtype": "object",
        "keep_default_na": True,
        "na_values": ["", "NA", "N/A", "n/a", "null", "NULL", "None", "-", "--", "#N/A"],
        "skip_blank_lines": True,
        "nrows": settings.max_rows + 1,
        "engine": "python",       # tolerant of ragged rows, and no C-level crash path
        "on_bad_lines": "skip",
    }
    try:
        df = pd.read_csv(io.StringIO(text), **read_kwargs)
    except Exception as exc:
        raise ValidationError(
            f"CSV could not be parsed: {str(exc)[:200]}. "
            "Check that the delimiter is consistent and the file is not truncated."
        ) from exc

    if df.empty and df.columns.empty:
        raise ValidationError("CSV contains no columns.")

    if _looks_headerless(list(df.columns)):
        result_notes.append(
            "No header row detected; columns were named column_1..column_n."
        )
        df = pd.read_csv(io.StringIO(text), header=None, **read_kwargs)
        df.columns = [f"column_{i + 1}" for i in range(df.shape[1])]

    return df, delimiter, encoding


# ------------------------------------------------------------------- XLSX -----
def _read_xlsx(upload: ValidatedUpload, result: IngestResult) -> pd.DataFrame:
    try:
        book = pd.ExcelFile(io.BytesIO(upload.content), engine="openpyxl")
    except Exception as exc:
        raise ValidationError(
            f"Excel workbook could not be opened: {str(exc)[:200]}"
        ) from exc

    result.sheet_names = [str(s) for s in book.sheet_names]
    if not result.sheet_names:
        raise ValidationError("Excel workbook contains no sheets.")

    # Pick the sheet with the most usable data rather than blindly taking the
    # first one — exports frequently lead with a cover/notes sheet.
    best_name, best_df, best_score = None, None, -1.0
    for name in result.sheet_names[:12]:
        try:
            candidate = book.parse(
                name, dtype="object", nrows=settings.max_rows + 1, na_values=["", "NA", "N/A", "null", "-"]
            )
        except Exception:
            continue
        if candidate.empty:
            continue
        filled = float(candidate.notna().to_numpy().mean()) if candidate.size else 0.0
        score = candidate.shape[0] * max(candidate.shape[1], 1) * max(filled, 0.01)
        if score > best_score:
            best_name, best_df, best_score = name, candidate, score

    if best_df is None:
        raise ValidationError(
            "Every sheet in the workbook is empty. Add data and upload again."
        )

    result.selected_sheet = best_name
    if len(result.sheet_names) > 1:
        result.notes.append(
            f"Workbook has {len(result.sheet_names)} sheets; analysed "
            f"'{best_name}' (the sheet with the most data)."
        )
    if _looks_headerless(list(best_df.columns)):
        result.notes.append("No header row detected; columns were auto-named.")
        best_df = book.parse(
            best_name, header=None, dtype="object", nrows=settings.max_rows + 1
        )
        best_df.columns = [f"column_{i + 1}" for i in range(best_df.shape[1])]
    return best_df


# ------------------------------------------------------------------- JSON ------
def _flatten_json(parsed: Any, notes: list[str]) -> pd.DataFrame:
    if isinstance(parsed, list):
        records = [r for r in parsed if isinstance(r, dict)]
        if not records and parsed:
            raise ValidationError(
                "JSON array does not contain objects; InsightOS needs tabular records."
            )
        return pd.json_normalize(records, max_level=2)

    if isinstance(parsed, dict):
        # Common wrapper shapes: {"data": [...]}, {"results": [...]}
        for key in ("data", "records", "rows", "results", "items", "values"):
            inner = parsed.get(key)
            if isinstance(inner, list) and inner and isinstance(inner[0], dict):
                notes.append(f"Records were read from the top-level '{key}' key.")
                return pd.json_normalize(inner, max_level=2)
        # Dict of equal-length lists -> columnar
        list_values = {k: v for k, v in parsed.items() if isinstance(v, list)}
        if list_values and len({len(v) for v in list_values.values()}) == 1:
            notes.append("JSON was read as a columnar object (key -> list of values).")
            return pd.DataFrame(list_values)
        notes.append("JSON object was read as a single record.")
        return pd.json_normalize([parsed], max_level=2)

    raise ValidationError("JSON root must be an array or object.")


def _read_json(upload: ValidatedUpload, result: IngestResult) -> pd.DataFrame:
    text, encoding = decode_text(upload.content)
    result.encoding = encoding

    if upload.kind == "ndjson":
        records: list[dict[str, Any]] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            if len(records) > settings.max_rows:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"NDJSON line {lineno} is not valid JSON: {exc.msg}"
                ) from exc
            if isinstance(obj, dict):
                records.append(obj)
        if not records:
            raise ValidationError("NDJSON file contained no JSON objects.")
        return pd.json_normalize(records, max_level=2)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"JSON is malformed: {exc.msg} (line {exc.lineno})") from exc
    return _flatten_json(parsed, result.notes)


# ---------------------------------------------------------------- pipeline ----
def read_dataframe(upload: ValidatedUpload) -> IngestResult:
    """Parse a validated upload into a bounded, typed DataFrame."""
    result = IngestResult(df=pd.DataFrame(), source_rows=0)

    with metrics.timer("ingest.parse", kind=upload.kind):
        if upload.kind in {"csv", "tsv"}:
            df, delimiter, encoding = _read_delimited(upload, result.notes)
            result.delimiter = delimiter
            result.encoding = encoding
        elif upload.kind == "xlsx":
            df = _read_xlsx(upload, result)
        else:
            df = _read_json(upload, result)

    if df.shape[1] == 0:
        raise ValidationError("File contains no columns.")

    # --- headers -------------------------------------------------------------
    names, header_notes = _normalise_headers(list(df.columns))
    df.columns = names
    result.notes.extend(header_notes)

    # --- column cap ----------------------------------------------------------
    if df.shape[1] > settings.max_columns:
        result.dropped_columns = list(df.columns[settings.max_columns :])
        df = df.iloc[:, : settings.max_columns]
        result.truncated_columns = True
        result.notes.append(
            f"Dataset had {len(result.dropped_columns) + settings.max_columns} columns; "
            f"the first {settings.max_columns} were kept (MAX_COLUMNS)."
        )

    # --- drop fully-empty rows/columns --------------------------------------
    before_rows = df.shape[0]
    df = df.dropna(axis=0, how="all")
    if df.shape[0] < before_rows:
        result.notes.append(f"Removed {before_rows - df.shape[0]} completely empty rows.")

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    # Empty columns are kept (data quality reports them) but flagged.

    # --- row cap -------------------------------------------------------------
    result.source_rows = int(df.shape[0])
    if df.shape[0] > settings.max_rows:
        df = df.head(settings.max_rows)
        result.truncated_rows = True
        result.notes.append(
            f"File contains more than {settings.max_rows:,} rows; the first "
            f"{settings.max_rows:,} were analysed (MAX_ROWS). "
            "Raise MAX_ROWS or pre-aggregate to analyse the full file."
        )

    if df.shape[0] == 0:
        raise ValidationError("File contains headers but no data rows.")

    # --- types ---------------------------------------------------------------
    df = df.reset_index(drop=True)
    with metrics.timer("ingest.infer_types"):
        df, conversions = infer_types(df)
    result.conversions = conversions
    result.df = df

    if empty_cols:
        result.notes.append(
            f"{len(empty_cols)} column(s) contain no values: "
            + ", ".join(empty_cols[:5])
            + ("…" if len(empty_cols) > 5 else "")
        )

    log.info(
        "dataset_ingested",
        extra={
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "kind": upload.kind,
            "truncated": result.truncated_rows,
        },
    )
    return result


# ------------------------------------------------------------- parquet IO -----
def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serialise to Parquet (columnar, compressed) for cheap re-reads."""
    buf = io.BytesIO()
    safe = df.copy()
    # Parquet cannot store mixed-type object columns; normalise them to string.
    for col in safe.columns:
        if pd.api.types.is_object_dtype(safe[col]):
            safe[col] = safe[col].astype("string")
    safe.to_parquet(buf, engine="pyarrow", compression="zstd", index=False)
    return buf.getvalue()


def from_parquet_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(data), engine="pyarrow")


def from_parquet_path(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")
