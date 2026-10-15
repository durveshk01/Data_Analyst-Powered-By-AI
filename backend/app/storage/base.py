"""Object storage abstraction: local disk (default) or Supabase Storage.

Both backends are free. Local disk is used for offline development, CI and
single-container deployments; Supabase Storage is used when the deployment
wants durable, CDN-fronted objects on the free tier.

Keys are always of the form ``<user_id>/<dataset_id>/<name>`` so a traversal
attempt cannot escape a tenant prefix — see :func:`safe_key`.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from app.config import settings
from app.errors import NotFoundError, ServiceUnavailableError, ValidationError
from app.obs.logging import get_logger

log = get_logger(__name__)

_KEY_SEGMENT = re.compile(r"^[A-Za-z0-9._\-]{1,120}$")


def safe_key(*segments: str) -> str:
    """Build a storage key, rejecting anything that could escape the prefix."""
    cleaned: list[str] = []
    for seg in segments:
        if not seg:
            continue
        if not _KEY_SEGMENT.match(seg) or seg in {".", ".."}:
            raise ValidationError(f"Illegal storage key segment: {seg!r}")
        cleaned.append(seg)
    if not cleaned:
        raise ValidationError("Storage key cannot be empty.")
    return "/".join(cleaned)


class ObjectStore(ABC):
    @abstractmethod
    async def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str: ...

    @abstractmethod
    async def get(self, key: str) -> bytes: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def delete_prefix(self, prefix: str) -> int: ...

    @abstractmethod
    async def exists(self, key: str) -> bool: ...

    @abstractmethod
    def local_path(self, key: str) -> Path | None:
        """Return a filesystem path when one exists.

        DuckDB and pyarrow can read a Parquet file far more cheaply from disk
        than from bytes, so callers prefer this when it is available and fall
        back to :meth:`get` otherwise.
        """

    @property
    @abstractmethod
    def name(self) -> str: ...


# ----------------------------------------------------------------- local ------
class LocalObjectStore(ObjectStore):
    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or settings.storage_dir).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "local"

    def _resolve(self, key: str) -> Path:
        parts = [p for p in key.split("/") if p]
        safe_key(*parts)
        target = (self._root / Path(*parts)).resolve()
        if not str(target).startswith(str(self._root)):
            raise ValidationError("Resolved storage path escapes the storage root.")
        return target

    async def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(target)  # atomic within a filesystem
        return key

    async def get(self, key: str) -> bytes:
        target = self._resolve(key)
        if not target.is_file():
            raise NotFoundError(f"Stored object '{key}' no longer exists.")
        return target.read_bytes()

    async def delete(self, key: str) -> None:
        target = self._resolve(key)
        target.unlink(missing_ok=True)

    async def delete_prefix(self, prefix: str) -> int:
        target = self._resolve(prefix)
        if not target.exists():
            return 0
        removed = 0
        if target.is_file():
            target.unlink()
            return 1
        for path in sorted(target.rglob("*"), key=lambda p: -len(p.parts)):
            if path.is_file():
                path.unlink()
                removed += 1
            elif path.is_dir():
                path.rmdir()
        target.rmdir()
        return removed

    async def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def local_path(self, key: str) -> Path | None:
        target = self._resolve(key)
        return target if target.is_file() else None


# -------------------------------------------------------------- supabase ------
class SupabaseObjectStore(ObjectStore):
    """Supabase Storage via its REST API — no extra SDK dependency."""

    def __init__(self) -> None:
        if not (settings.supabase_url and settings.supabase_service_role_key):
            raise ServiceUnavailableError(
                "STORAGE_BACKEND=supabase requires SUPABASE_URL and "
                "SUPABASE_SERVICE_ROLE_KEY."
            )
        self._base = settings.supabase_url.rstrip("/") + "/storage/v1"
        self._bucket = settings.supabase_storage_bucket
        self._headers = {
            "Authorization": f"Bearer {settings.supabase_service_role_key}",
            "apikey": settings.supabase_service_role_key,
        }
        # Objects are cached locally after first read so repeat analyses on the
        # same dataset do not re-download it on every request.
        self._cache_dir = (settings.storage_dir / "_supabase_cache").resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "supabase"

    def _url(self, kind: str, key: str) -> str:
        return f"{self._base}/{kind}/{self._bucket}/{key}"

    def _cache_file(self, key: str) -> Path:
        parts = [p for p in key.split("/") if p]
        safe_key(*parts)
        return self._cache_dir / Path(*parts)

    async def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                self._url("object", key),
                content=data,
                headers={
                    **self._headers,
                    "Content-Type": content_type,
                    "x-upsert": "true",
                },
            )
        if resp.status_code >= 400:
            raise ServiceUnavailableError(
                f"Supabase Storage upload failed ({resp.status_code}): {resp.text[:200]}"
            )
        cache = self._cache_file(key)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
        return key

    async def get(self, key: str) -> bytes:
        cache = self._cache_file(key)
        if cache.is_file():
            return cache.read_bytes()
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(self._url("object", key), headers=self._headers)
        if resp.status_code == 404:
            raise NotFoundError(f"Stored object '{key}' no longer exists.")
        if resp.status_code >= 400:
            raise ServiceUnavailableError(
                f"Supabase Storage download failed ({resp.status_code})."
            )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(resp.content)
        return resp.content

    async def delete(self, key: str) -> None:
        self._cache_file(key).unlink(missing_ok=True)
        async with httpx.AsyncClient(timeout=60.0) as client:
            await client.delete(self._url("object", key), headers=self._headers)

    async def delete_prefix(self, prefix: str) -> int:
        async with httpx.AsyncClient(timeout=60.0) as client:
            listing = await client.post(
                f"{self._base}/object/list/{self._bucket}",
                headers=self._headers,
                json={"prefix": prefix, "limit": 1000},
            )
            if listing.status_code >= 400:
                return 0
            names = [f"{prefix}/{item['name']}" for item in listing.json()]
            if not names:
                return 0
            await client.request(
                "DELETE",
                f"{self._base}/object/{self._bucket}",
                headers=self._headers,
                json={"prefixes": names},
            )
        for name in names:
            self._cache_file(name).unlink(missing_ok=True)
        return len(names)

    async def exists(self, key: str) -> bool:
        if self._cache_file(key).is_file():
            return True
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.head(self._url("object", key), headers=self._headers)
        return resp.status_code < 400

    def local_path(self, key: str) -> Path | None:
        cache = self._cache_file(key)
        return cache if cache.is_file() else None


_store: ObjectStore | None = None


def get_store() -> ObjectStore:
    global _store
    if _store is None:
        if settings.storage_backend == "supabase":
            try:
                _store = SupabaseObjectStore()
            except ServiceUnavailableError as exc:
                log.warning(
                    "supabase_storage_unavailable_falling_back_to_local",
                    extra={"error": str(exc)},
                )
                _store = LocalObjectStore()
        else:
            _store = LocalObjectStore()
        log.info("object_store_ready", extra={"backend": _store.name})
    return _store


def reset_store() -> None:
    """Test hook."""
    global _store
    _store = None
