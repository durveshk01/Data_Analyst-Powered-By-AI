"""Shared helpers for the analytics engine.

Two concerns live here:

* **JSON safety.** NumPy scalars, `NaN`, `±Inf`, `Timestamp` and `Decimal` are
  not JSON-serialisable (or not *validly* serialisable — `NaN` is accepted by
  Python's `json` but rejected by strict parsers including the browser's).
  Everything leaving the engine passes through :func:`to_jsonable`.
* **Formatting.** Human-readable numbers for insight sentences and reports,
  including Indian-digit grouping for rupee amounts, since that is what a
  lakh/crore-denominated dataset should read like.
"""

from __future__ import annotations

import contextvars
import math
import re
from contextlib import contextmanager
from datetime import date, datetime, time as dtime
from decimal import Decimal
from typing import Any, Iterator

import numpy as np
import pandas as pd

# Units measured by the profiler for the dataset currently being analysed.
# Name-based guessing cannot tell a 0-1 fraction from a 0-100 percentage, so
# when a profile is available its verdict wins. Request-scoped via a ContextVar,
# so concurrent requests on different datasets never see each other's units.
_unit_overrides: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "insightos_unit_overrides", default=None
)


@contextmanager
def unit_scope(units: dict[str, str]) -> Iterator[None]:
    """Make profiled column units authoritative for the enclosed block."""
    token = _unit_overrides.set(dict(units))
    try:
        yield
    finally:
        _unit_overrides.reset(token)


# --------------------------------------------------------------- JSON safety --
def to_jsonable(value: Any) -> Any:
    """Recursively convert `value` into something strictly JSON-serialisable."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, Decimal):
        return to_jsonable(float(value))
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return to_jsonable(float(value))
    if isinstance(value, (np.datetime64, pd.Timestamp)):
        ts = pd.Timestamp(value)
        return None if pd.isna(ts) else ts.isoformat()
    if isinstance(value, (datetime, date, dtime)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, pd.Series):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return dataframe_to_records(value)
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def dataframe_to_records(df: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records without the `NaN` landmine."""
    frame = df.head(limit) if limit is not None else df
    frame = frame.replace({np.nan: None, pd.NaT: None})
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        records.append({str(k): to_jsonable(v) for k, v in row.items()})
    return records


def safe_float(value: Any) -> float | None:
    """Coerce to a finite float, or None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def safe_div(numerator: Any, denominator: Any) -> float | None:
    n, d = safe_float(numerator), safe_float(denominator)
    if n is None or d is None or d == 0:
        return None
    return safe_float(n / d)


def pct_change(current: Any, previous: Any) -> float | None:
    """Percentage change, guarding against a zero or negative base."""
    c, p = safe_float(current), safe_float(previous)
    if c is None or p is None or p == 0:
        return None
    return safe_float((c - p) / abs(p) * 100.0)


# ---------------------------------------------------------------- formatting --
def guess_unit(column_name: str) -> str:
    """Display unit for a column. Presentation only — never used in arithmetic.

    Prefers the unit the profiler measured for the active dataset (see
    :func:`unit_scope`), then falls back to name hints.
    """
    overrides = _unit_overrides.get()
    if overrides and column_name in overrides:
        return overrides[column_name]

    from app.engine.semantics import COUNT_NAMES, CURRENCY_NAMES, PERCENT_NAMES

    if PERCENT_NAMES.search(column_name):
        return "percent"
    if CURRENCY_NAMES.search(column_name):
        return "currency"
    if COUNT_NAMES.search(column_name):
        return "count"
    return "number"


def format_indian(number: float, *, decimals: int = 0) -> str:
    """Group digits in the Indian system (12,34,567) used for lakh/crore."""
    negative = number < 0
    number = abs(number)
    whole = int(number)
    frac = number - whole
    s = str(whole)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups + [tail])
    out = s
    if decimals > 0:
        out += f".{int(round(frac * 10**decimals)):0{decimals}d}"
    return ("-" if negative else "") + out


def compact_number(value: Any, *, unit: str = "number", currency: str = "₹") -> str:
    """Short human label: ₹46.7L, 12.4K, 18.2%."""
    v = safe_float(value)
    if v is None:
        return "—"
    if unit == "ratio":
        return f"{v * 100:,.1f}%"
    if unit == "percent":
        return f"{v:,.1f}%"
    prefix = currency if unit == "currency" else ""
    sign = "-" if v < 0 else ""
    a = abs(v)
    if unit == "currency":
        # Lakh/crore scale, which is what Indian business reporting expects.
        if a >= 1_00_00_000:
            return f"{sign}{prefix}{a / 1_00_00_000:.2f}Cr"
        if a >= 1_00_000:
            return f"{sign}{prefix}{a / 1_00_000:.2f}L"
        if a >= 1_000:
            return f"{sign}{prefix}{format_indian(a)}"
        return f"{sign}{prefix}{a:,.2f}".rstrip("0").rstrip(".")
    if a >= 1_000_000_000:
        return f"{sign}{a / 1_000_000_000:.2f}B"
    if a >= 1_000_000:
        return f"{sign}{a / 1_000_000:.2f}M"
    if a >= 1_000:
        return f"{sign}{a / 1_000:.1f}K"
    if unit == "count" or float(a).is_integer():
        return f"{sign}{int(a):,}"
    return f"{sign}{a:,.2f}"


def format_delta(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{'+' if value >= 0 else ''}{value:.1f}%"


def slugify(text: str, *, max_length: int = 60) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return (out or "field")[:max_length]


def truncate(text: str, limit: int = 160) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"
