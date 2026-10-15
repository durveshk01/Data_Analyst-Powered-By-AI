"""Async engine/session management.

Works unchanged against Supabase Postgres and local SQLite. On SQLite the
schema is created with `create_all` (dev/CI convenience); on Postgres the
migrations in `supabase/migrations/` are authoritative, and `create_all` is
only used when `INSIGHTOS_AUTO_MIGRATE=1` is set for a throwaway database.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config import settings
from app.db.models import Base
from app.obs.logging import get_logger

log = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _ensure_sqlite_dir() -> None:
    if not settings.uses_sqlite:
        return
    # sqlite+aiosqlite:///./.data/insightos.db  ->  ./.data/insightos.db
    _, _, path = settings.database_url.partition(":///")
    if not path or path == ":memory:":
        return
    from pathlib import Path

    target = Path(path)
    if not target.is_absolute():
        target = (Path(__file__).resolve().parents[2] / path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _ensure_sqlite_dir()
        if settings.uses_sqlite:
            _engine = create_async_engine(
                settings.database_url,
                echo=False,
                future=True,
                poolclass=NullPool,
                connect_args={"check_same_thread": False, "timeout": 30},
            )
        else:
            _engine = create_async_engine(
                settings.database_url,
                echo=False,
                future=True,
                pool_size=settings.db_pool_size,
                max_overflow=settings.db_max_overflow,
                pool_pre_ping=True,
                pool_recycle=1800,
                # Supabase's pooler does not support prepared statement caching.
                connect_args={"statement_cache_size": 0, "timeout": 30},
            )
        log.info(
            "database_engine_created",
            extra={"dialect": _engine.dialect.name, "sqlite": settings.uses_sqlite},
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope: commits on success, rolls back on failure."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency."""
    async with session_scope() as session:
        yield session


async def init_database() -> None:
    """Create tables when running on SQLite or an explicitly opted-in Postgres."""
    auto = settings.uses_sqlite or os.getenv("INSIGHTOS_AUTO_MIGRATE") == "1"
    if not auto:
        log.info("database_migrations_external", extra={"hint": "supabase/migrations"})
        return
    engine = get_engine()
    async with engine.begin() as conn:
        if settings.uses_sqlite:
            from sqlalchemy import text

            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA foreign_keys=ON"))
        await conn.run_sync(Base.metadata.create_all)
    log.info("database_schema_ready")


async def dispose_database() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


async def healthcheck() -> dict[str, object]:
    from sqlalchemy import text

    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"ok": True, "dialect": get_engine().dialect.name}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}
