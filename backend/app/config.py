"""Central configuration for InsightOS.

Every setting has a working default so the service boots with an empty
environment. Paid services are opt-in only: with no environment at all the
application runs on SQLite + local disk storage + deterministic analytics.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AuthMode = Literal["local", "supabase"]
StorageBackend = Literal["local", "supabase"]
AIProviderName = Literal["auto", "ollama", "external", "none"]

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_REPO_ROOT / ".env", Path(".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- core ---
    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    app_name: str = "InsightOS"
    app_version: str = "1.0.0"

    app_jwt_secret: str = "dev-only-insecure-secret-change-me-in-production"
    access_token_ttl_seconds: int = 3600
    refresh_token_ttl_seconds: int = 1_209_600
    upload_token_ttl_seconds: int = 300

    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # ---------------------------------------------------------------- auth ---
    auth_mode: AuthMode = "local"

    # ------------------------------------------------------------ database ---
    database_url: str = "sqlite+aiosqlite:///./.data/insightos.db"
    db_pool_size: int = 5
    db_max_overflow: int = 5

    # ------------------------------------------------------------ supabase ---
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    supabase_jwt_secret: str = ""
    supabase_storage_bucket: str = "datasets"

    # ------------------------------------------------------------- storage ---
    storage_backend: StorageBackend = "local"
    storage_local_dir: str = "./.data/storage"

    # ------------------------------------------------------------------ ai ---
    ai_provider: AIProviderName = "auto"
    ai_request_timeout_seconds: float = 60.0
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    external_ai_base_url: str = ""
    external_ai_api_key: str = ""
    external_ai_model: str = ""
    ai_redact_sample_values: bool = False

    # ------------------------------------------------------------- limits ----
    max_file_size_mb: int = 50
    max_rows: int = 250_000
    max_columns: int = 512
    max_datasets_per_user: int = 10
    max_reports_per_user: int = 20
    max_analysis_history: int = 100
    max_cell_length: int = 32_768
    query_timeout_seconds: int = 20
    sql_row_limit: int = 5_000

    # -------------------------------------------------------- rate limiting --
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    rate_limit_upload_requests: int = 10
    rate_limit_ai_requests: int = 30

    # ----------------------------------------------------------------- demo --
    demo_enabled: bool = True
    demo_rows: int = 120_000
    demo_seed: int = 20_260_101

    # ------------------------------------------------------------ validators --
    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        return v.upper()

    @field_validator("database_url")
    @classmethod
    def _normalise_db_url(cls, v: str) -> str:
        """Accept plain libpq URLs and upgrade them to async drivers.

        Supabase hands out `postgresql://...`; SQLAlchemy's async engine needs
        an explicit async driver, so rewrite it rather than making the operator
        remember the `+asyncpg` suffix.
        """
        if v.startswith("postgres://"):
            v = "postgresql://" + v[len("postgres://") :]
        if v.startswith("postgresql://"):
            v = "postgresql+asyncpg://" + v[len("postgresql://") :]
        if v.startswith("sqlite://") and "+aiosqlite" not in v:
            v = "sqlite+aiosqlite://" + v[len("sqlite://") :]
        return v

    # -------------------------------------------------------------- helpers --
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def storage_dir(self) -> Path:
        p = Path(self.storage_local_dir)
        if not p.is_absolute():
            p = _REPO_ROOT / "backend" / p
        return p.resolve()

    @property
    def uses_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def supabase_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_anon_key)

    @property
    def external_ai_configured(self) -> bool:
        return bool(self.external_ai_base_url and self.external_ai_model)

    def production_warnings(self) -> list[str]:
        """Misconfigurations that are tolerable locally but not in production."""
        problems: list[str] = []
        if not self.is_production:
            return problems
        if "change-me" in self.app_jwt_secret or len(self.app_jwt_secret) < 32:
            problems.append("APP_JWT_SECRET is weak or still the development default.")
        if self.auth_mode == "supabase" and not self.supabase_configured:
            problems.append("AUTH_MODE=supabase but SUPABASE_URL/ANON_KEY are missing.")
        if self.storage_backend == "supabase" and not self.supabase_service_role_key:
            problems.append(
                "STORAGE_BACKEND=supabase but SUPABASE_SERVICE_ROLE_KEY is missing."
            )
        if self.uses_sqlite:
            problems.append(
                "DATABASE_URL points at SQLite; use Supabase/Postgres in production."
            )
        if "*" in self.cors_origin_list:
            problems.append("CORS_ORIGINS contains a wildcard.")
        return problems


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
