"""SQLAlchemy ORM models.

Portability note: the same models run against Supabase Postgres (production)
and SQLite (offline dev + CI), so column types are chosen to work on both —
`String(36)` UUIDs generated in Python, and `JSON` with a `JSONB` variant on
Postgres. The canonical production schema, including indexes, constraints and
row-level-security policies, lives in `supabase/migrations/`; this module is
kept byte-for-byte compatible with it and is verified by
`tests/test_schema_parity.py`.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JSONType = JSON().with_variant(JSONB(), "postgresql")


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSONType, list[Any]: JSONType}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )


# ------------------------------------------------------------------- users ---
class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(200))
    # Null when authentication is delegated to Supabase Auth.
    password_hash: Mapped[str | None] = mapped_column(String(128))
    auth_provider: Mapped[str] = mapped_column(String(32), default="local", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    settings: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    projects: Mapped[list["Project"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )
    datasets: Mapped[list["Dataset"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("length(email) > 3", name="ck_users_email_len"),
    )


# ---------------------------------------------------------------- projects ---
class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    colour: Mapped[str] = mapped_column(String(16), default="indigo", nullable=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    owner: Mapped[User] = relationship(back_populates="projects")
    datasets: Mapped[list["Dataset"]] = relationship(back_populates="project")

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_projects_user_name"),
        Index("ix_projects_user_created", "user_id", "created_at"),
    )


# ---------------------------------------------------------------- datasets ---
class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), index=True
    )
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(300), nullable=False)
    file_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="pending", nullable=False, index=True)
    error_message: Mapped[str | None] = mapped_column(Text)
    quality_score: Mapped[float | None] = mapped_column(Float)
    processing_ms: Mapped[int | None] = mapped_column(Integer)
    storage_path: Mapped[str | None] = mapped_column(String(500))
    raw_storage_path: Mapped[str | None] = mapped_column(String(500))
    checksum: Mapped[str | None] = mapped_column(String(64), index=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    profile: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    semantics: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    quality: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)

    owner: Mapped[User] = relationship(back_populates="datasets")
    project: Mapped[Project | None] = relationship(back_populates="datasets")
    columns: Mapped[list["DatasetColumn"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    versions: Mapped[list["DatasetVersion"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','processing','ready','failed','deleted')",
            name="ck_datasets_status",
        ),
        CheckConstraint("row_count >= 0", name="ck_datasets_rows"),
        Index("ix_datasets_user_created", "user_id", "created_at"),
    )


class DatasetVersion(Base, TimestampMixin):
    """An immutable snapshot. v1 is always the untouched upload.

    Cleaning never mutates v1 — it writes a new version with a transform log,
    which is what makes "accept / reject / review" reversible.
    """

    __tablename__ = "dataset_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(120), default="original", nullable=False)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_original: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    transforms: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="versions")

    __table_args__ = (
        UniqueConstraint("dataset_id", "version", name="uq_dataset_versions"),
    )


class DatasetColumn(Base):
    __tablename__ = "dataset_columns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    dtype: Mapped[str] = mapped_column(String(32), nullable=False)
    semantic_type: Mapped[str] = mapped_column(String(32), nullable=False)
    role: Mapped[str] = mapped_column(String(24), nullable=False)
    null_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    null_ratio: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    unique_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    stats: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="columns")

    __table_args__ = (
        UniqueConstraint("dataset_id", "name", name="uq_dataset_columns"),
        CheckConstraint(
            "role IN ('measure','dimension','time','identifier','ignored')",
            name="ck_dataset_columns_role",
        ),
    )


# ---------------------------------------------------------------- analyses ---
class Analysis(Base, TimestampMixin):
    """A conversation thread scoped to one dataset."""

    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL")
    )
    title: Mapped[str] = mapped_column(String(300), default="New analysis", nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    messages: Mapped[list["AnalysisMessage"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_analyses_user_updated", "user_id", "updated_at"),)


class AnalysisMessage(Base):
    __tablename__ = "analysis_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # Which engine produced the answer: "deterministic" | "ollama" | "external"
    engine: Mapped[str | None] = mapped_column(String(32))
    tool_calls: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow, nullable=False
    )

    analysis: Mapped[Analysis] = relationship(back_populates="messages")

    __table_args__ = (
        CheckConstraint("role IN ('user','assistant','system','tool')", name="ck_msg_role"),
        Index("ix_analysis_messages_thread", "analysis_id", "created_at"),
    )


class AnalysisResult(Base):
    """A cached, replayable tool execution with its exact arguments."""

    __tablename__ = "analysis_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    analysis_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    cache_key: Mapped[str | None] = mapped_column(String(64), index=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow, nullable=False
    )


# ---------------------------------------------------------------- findings ---
class Insight(Base):
    __tablename__ = "insights"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, default="", nullable=False)
    severity: Mapped[str] = mapped_column(String(16), default="info", nullable=False)
    importance: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    metric: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    chart: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow, nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "severity IN ('info','positive','warning','critical')", name="ck_insight_sev"
        ),
        Index("ix_insights_dataset_importance", "dataset_id", "importance"),
    )


class Anomaly(Base):
    __tablename__ = "anomalies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    column_name: Mapped[str] = mapped_column(String(300), nullable=False)
    label: Mapped[str] = mapped_column(String(300), default="", nullable=False)
    observed: Mapped[float | None] = mapped_column(Float)
    expected: Mapped[float | None] = mapped_column(Float)
    deviation: Mapped[float | None] = mapped_column(Float)
    score: Mapped[float | None] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(16), default="warning", nullable=False)
    row_index: Mapped[int | None] = mapped_column(Integer)
    context: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow, nullable=False
    )


class Forecast(Base):
    __tablename__ = "forecasts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_column: Mapped[str] = mapped_column(String(300), nullable=False)
    time_column: Mapped[str] = mapped_column(String(300), nullable=False)
    grain: Mapped[str] = mapped_column(String(16), nullable=False)
    horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    candidates: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    history: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    points: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow, nullable=False
    )


class Chart(Base):
    __tablename__ = "charts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    analysis_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("analyses.id", ondelete="SET NULL")
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    chart_type: Mapped[str] = mapped_column(String(32), nullable=False)
    spec: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow, nullable=False
    )


class Report(Base, TimestampMixin):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL")
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    fmt: Mapped[str] = mapped_column(String(8), default="pdf", nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="ready", nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    storage_path: Mapped[str | None] = mapped_column(String(500))
    summary: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)

    __table_args__ = (
        CheckConstraint("fmt IN ('pdf','xlsx','csv','png','json')", name="ck_reports_fmt"),
        Index("ix_reports_user_created", "user_id", "created_at"),
    )


class AuditLog(Base):
    """Append-only trail of security-relevant actions."""

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    resource_type: Mapped[str | None] = mapped_column(String(48))
    resource_id: Mapped[str | None] = mapped_column(String(64))
    outcome: Mapped[str] = mapped_column(String(16), default="success", nullable=False)
    ip_hash: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(300))
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow, nullable=False
    )

    __table_args__ = (Index("ix_audit_logs_created", "created_at"),)


ALL_TABLES: tuple[str, ...] = (
    "users",
    "projects",
    "datasets",
    "dataset_versions",
    "dataset_columns",
    "analyses",
    "analysis_messages",
    "analysis_results",
    "insights",
    "anomalies",
    "forecasts",
    "charts",
    "reports",
    "audit_logs",
)
