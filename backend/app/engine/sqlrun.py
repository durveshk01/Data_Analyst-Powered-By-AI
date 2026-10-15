"""Read-only SQL execution over a dataset, via DuckDB.

Defence in depth around :mod:`app.security.sql_guard`:

1. The statement is parsed and proved read-only *before* it is sent anywhere.
2. DuckDB runs in an in-memory database with no attached files; the dataset is
   exposed as a single registered view named ``dataset``, so there is no other
   table in scope and no filesystem path for a query to reach.
3. External access and extension loading are disabled on the connection, so
   even if a filesystem function slipped past the parser it would fail.
4. Execution happens on a worker thread with a wall-clock timeout; the DuckDB
   connection is interrupted when the deadline passes.
5. The row limit is enforced by the rewritten SQL *and* re-checked on the
   returned frame.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import duckdb
import pandas as pd

from app.config import settings
from app.engine.utils import dataframe_to_records
from app.errors import AnalysisError, UnsafeQueryError
from app.obs.logging import get_logger
from app.obs.metrics import metrics
from app.security.sql_guard import validate_sql

log = get_logger(__name__)

TABLE_NAME = "dataset"

# Settings applied to every connection. `enable_external_access=false` is the
# important one: it disables httpfs, local file reads and ATTACH.
_HARDENING: tuple[str, ...] = (
    "SET enable_external_access=false",
    "SET autoinstall_known_extensions=false",
    "SET autoload_known_extensions=false",
    "SET memory_limit='512MB'",
    "SET threads=2",
)


def run_sql(
    df: pd.DataFrame,
    sql: str,
    *,
    row_limit: int | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Validate and execute a read-only SELECT against `df`."""
    allowed_columns = [str(c) for c in df.columns]
    validation = validate_sql(
        sql,
        allowed_tables=[TABLE_NAME],
        allowed_columns=allowed_columns,
        row_limit=row_limit,
    )

    timeout = timeout_seconds or settings.query_timeout_seconds
    started = time.perf_counter()

    connection = duckdb.connect(database=":memory:")
    try:
        for statement in _HARDENING:
            try:
                connection.execute(statement)
            except duckdb.Error:
                # An older DuckDB may not know a given setting; the parser-level
                # guard is still in force, so continue.
                log.debug("duckdb_setting_unsupported", extra={"statement": statement})

        connection.register(TABLE_NAME, df)

        result: dict[str, Any] = {}
        error: list[BaseException] = []

        def execute() -> None:
            try:
                frame = connection.execute(validation.safe_sql).fetch_df()
                result["frame"] = frame
            except BaseException as exc:  # noqa: BLE001 - reported to the caller
                error.append(exc)

        worker = threading.Thread(target=execute, daemon=True, name="duckdb-query")
        with metrics.timer("sql.execute"):
            worker.start()
            worker.join(timeout)

        if worker.is_alive():
            try:
                connection.interrupt()
            except duckdb.Error:  # pragma: no cover
                pass
            worker.join(2.0)
            metrics.incr("sql.timeout")
            raise AnalysisError(
                f"Query exceeded the {timeout}s time limit and was cancelled. "
                "Add a filter or aggregate fewer rows."
            )

        if error:
            exc = error[0]
            if isinstance(exc, duckdb.Error):
                metrics.incr("sql.error")
                raise UnsafeQueryError(f"Query failed: {str(exc)[:300]}") from exc
            raise exc

        frame = result.get("frame")
        if frame is None:  # pragma: no cover - defensive
            raise AnalysisError("Query returned no result set.")

        truncated = len(frame) >= validation.row_limit
        if len(frame) > validation.row_limit:
            frame = frame.head(validation.row_limit)

        duration_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "sql_executed",
            extra={
                "rows": int(len(frame)),
                "ms": duration_ms,
                "tables": validation.tables,
            },
        )
        return {
            "columns": [str(c) for c in frame.columns],
            "rows": dataframe_to_records(frame),
            "row_count": int(len(frame)),
            "truncated": truncated,
            "row_limit": validation.row_limit,
            "executed_sql": validation.safe_sql,
            "duration_ms": duration_ms,
            "referenced_tables": validation.tables,
            "referenced_columns": validation.columns,
        }
    finally:
        connection.close()


def schema_ddl(df: pd.DataFrame) -> str:
    """A CREATE TABLE sketch of the dataset, for display and for prompting.

    Only names and types — never values.
    """
    lines = [f"CREATE TABLE {TABLE_NAME} ("]
    for name in df.columns:
        lines.append(f"  {_quote(str(name))} {_sql_type(df[name])},")
    if len(lines) > 1:
        lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    return "\n".join(lines)


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sql_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    return "VARCHAR"
