"""Read-only SQL guard.

InsightOS can let a user (or the AI layer) express an analysis as SQL, but a
generated string is never executed as-is. Every statement goes through this
module first, which parses it into an AST with ``sqlglot`` and proves it is
read-only before DuckDB ever sees it.

Guarantees enforced here
------------------------
* exactly one statement (no ``;`` chaining),
* the root node is a ``SELECT`` / ``WITH`` / set-operation — never DML or DDL,
* no node anywhere in the tree belongs to a mutating or side-effecting class
  (``INSERT``/``UPDATE``/``DELETE``/``DROP``/``CREATE``/``ALTER``/``TRUNCATE``/
  ``ATTACH``/``COPY``/``PRAGMA``/``SET``/transaction control/...),
* no filesystem or network function (``read_csv``, ``read_parquet``, ``glob``,
  ``httpfs`` helpers, ...),
* every referenced table is on the caller's allow-list,
* every referenced column is on the caller's allow-list (or is an alias the
  query itself defines),
* the result is wrapped so a hard ``LIMIT`` always applies.

The execution side additionally applies a wall-clock timeout and runs DuckDB
against a read-only view of a single Parquet file, so even a validator bypass
cannot reach other tenants' data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import sqlglot
from sqlglot import exp

from app.config import settings
from app.errors import UnsafeQueryError

_DIALECT = "duckdb"
_RESULT_ALIAS = "insightos_result"

# Statement/clause classes that must never appear anywhere in the tree. Looked
# up defensively so a sqlglot upgrade that renames a class cannot silently
# disable a check (a missing name is reported by `missing_denied_nodes()`).
_DENIED_NODE_NAMES: tuple[str, ...] = (
    "Insert", "Update", "Delete", "Merge", "Drop", "Create", "Alter",
    "AlterTable", "AlterColumn", "TruncateTable", "Command", "Attach", "Detach",
    "Copy", "Grant", "Revoke", "Use", "Set", "SetItem", "Transaction", "Commit",
    "Rollback", "Pragma", "Analyze", "Call", "Export", "LoadData", "Vacuum",
    "Refresh", "Describe", "Kill", "Comment", "RenameTable", "Index",
)

# DuckDB functions that read the filesystem/network or escape the sandbox.
_DENIED_FUNCTIONS: frozenset[str] = frozenset(
    {
        "read_csv", "read_csv_auto", "read_parquet", "read_json",
        "read_json_auto", "read_ndjson", "read_ndjson_auto", "read_text",
        "read_blob", "parquet_scan", "csv_scan", "json_scan", "glob",
        "sniff_csv", "copy", "install", "load", "attach", "detach",
        "httpfs", "url", "read_csv_gz", "delta_scan", "iceberg_scan",
        "postgres_scan", "postgres_query", "mysql_scan", "mysql_query",
        "sqlite_scan", "sqlite_query", "duckdb_extensions", "shell",
        "system", "getenv", "pg_terminate_backend", "lambda",
    }
)

# Obvious textual escapes that should never survive even before parsing.
_FORBIDDEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"/\*", re.I), "Block comments are not allowed in SQL queries."),
    (re.compile(r"\binto\s+outfile\b", re.I), "File output is not allowed."),
    (re.compile(r"\bcopy\b", re.I), "COPY is not allowed."),
    (re.compile(r"\battach\b", re.I), "ATTACH is not allowed."),
    (re.compile(r"\bpragma\b", re.I), "PRAGMA is not allowed."),
    (re.compile(r"\binstall\b", re.I), "INSTALL is not allowed."),
)

_MAX_SQL_LENGTH = 8_000


@dataclass(slots=True)
class SQLValidationResult:
    """The outcome of validating a statement."""

    safe_sql: str
    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_limit: int = 0
    selects_all_columns: bool = False


def missing_denied_nodes() -> list[str]:
    """Names in the denylist that this sqlglot build does not expose.

    Surfaced by the test suite: if sqlglot renames a node class, the test fails
    loudly instead of the guard quietly weakening.
    """
    return [name for name in _DENIED_NODE_NAMES if not hasattr(exp, name)]


def _denied_node_types() -> tuple[type, ...]:
    return tuple(
        getattr(exp, name)
        for name in _DENIED_NODE_NAMES
        if isinstance(getattr(exp, name, None), type)
    )


def _normalise(name: str) -> str:
    return name.strip().strip('"').strip("`").strip("[]").lower()


def _collect_defined_aliases(tree: exp.Expression) -> set[str]:
    """Names the query itself introduces (column aliases, CTEs, table aliases)."""
    names: set[str] = set()
    for node in tree.walk():
        if isinstance(node, exp.Alias):
            alias = node.alias
            if alias:
                names.add(_normalise(alias))
        elif isinstance(node, exp.TableAlias):
            if node.name:
                names.add(_normalise(node.name))
            for col in node.columns:
                names.add(_normalise(col.name))
        elif isinstance(node, exp.CTE):
            if node.alias:
                names.add(_normalise(node.alias))
    return names


def validate_sql(
    sql: str,
    *,
    allowed_tables: Iterable[str],
    allowed_columns: Iterable[str],
    row_limit: int | None = None,
) -> SQLValidationResult:
    """Validate and rewrite `sql` so it is provably read-only and bounded."""
    if not sql or not sql.strip():
        raise UnsafeQueryError("Query is empty.")
    if len(sql) > _MAX_SQL_LENGTH:
        raise UnsafeQueryError(
            f"Query is {len(sql)} characters; the limit is {_MAX_SQL_LENGTH}."
        )

    for pattern, message in _FORBIDDEN_PATTERNS:
        if pattern.search(sql):
            raise UnsafeQueryError(message)

    try:
        statements = [s for s in sqlglot.parse(sql, read=_DIALECT) if s is not None]
    except Exception as exc:  # sqlglot raises ParseError and friends
        raise UnsafeQueryError(f"Query could not be parsed: {exc}") from exc

    if not statements:
        raise UnsafeQueryError("Query contains no executable statement.")
    if len(statements) > 1:
        raise UnsafeQueryError(
            "Only a single statement is allowed; statement chaining is blocked."
        )

    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery)):
        raise UnsafeQueryError(
            f"Only SELECT queries are permitted; got {type(tree).__name__.upper()}."
        )

    denied = _denied_node_types()
    for node in tree.walk():
        if isinstance(node, denied):
            raise UnsafeQueryError(
                f"{type(node).__name__.upper()} is not permitted. "
                "InsightOS executes read-only SELECT queries only."
            )
        if isinstance(node, (exp.Anonymous, exp.Func)):
            fname = _function_name(node)
            if fname and fname in _DENIED_FUNCTIONS:
                raise UnsafeQueryError(
                    f"Function '{fname}()' is not permitted — it can read files "
                    "or reach the network."
                )

    allowed_table_set = {_normalise(t) for t in allowed_tables}
    allowed_column_set = {_normalise(c) for c in allowed_columns}
    defined_aliases = _collect_defined_aliases(tree)

    referenced_tables: list[str] = []
    for table in tree.find_all(exp.Table):
        name = _normalise(table.name)
        if not name:
            continue
        if name in defined_aliases:  # a CTE, not a physical table
            continue
        if name not in allowed_table_set:
            raise UnsafeQueryError(
                f"Table '{table.name}' is not accessible. "
                f"Allowed: {', '.join(sorted(allowed_table_set)) or 'none'}."
            )
        if name not in referenced_tables:
            referenced_tables.append(name)

    if not referenced_tables:
        raise UnsafeQueryError(
            "Query does not reference an allowed table. "
            f"Use: {', '.join(sorted(allowed_table_set)) or 'none'}."
        )

    referenced_columns: list[str] = []
    for column in tree.find_all(exp.Column):
        name = _normalise(column.name)
        if not name or name == "*":
            continue
        if name in defined_aliases or name in allowed_column_set:
            if name in allowed_column_set and name not in referenced_columns:
                referenced_columns.append(name)
            continue
        raise UnsafeQueryError(
            f"Column '{column.name}' does not exist in this dataset."
        )

    selects_star = any(isinstance(s, exp.Star) for s in tree.find_all(exp.Star))

    limit = row_limit if row_limit is not None else settings.sql_row_limit
    limit = max(1, min(int(limit), settings.sql_row_limit))

    inner = tree.sql(dialect=_DIALECT)
    safe_sql = (
        f'SELECT * FROM ({inner}) AS "{_RESULT_ALIAS}" LIMIT {limit}'
    )

    return SQLValidationResult(
        safe_sql=safe_sql,
        tables=referenced_tables,
        columns=referenced_columns,
        row_limit=limit,
        selects_all_columns=selects_star,
    )


def _function_name(node: Any) -> str | None:
    name = getattr(node, "name", None)
    if isinstance(name, str) and name:
        return name.lower()
    sql_name = getattr(node, "sql_name", None)
    if callable(sql_name):
        try:
            return str(sql_name()).lower()
        except Exception:  # pragma: no cover - defensive
            return None
    return None
