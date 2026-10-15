"""Typed application errors mapped to HTTP responses by a single handler.

Keeping the taxonomy here (rather than raising `HTTPException` everywhere) means
the engine modules stay framework-agnostic and unit-testable without FastAPI.
"""

from __future__ import annotations

from typing import Any


class InsightOSError(Exception):
    """Base class for every expected (non-bug) failure."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": {"code": self.code, "message": self.message}}
        if self.details is not None:
            payload["error"]["details"] = self.details
        return payload


class ValidationError(InsightOSError):
    status_code = 422
    code = "validation_error"


class BadRequestError(InsightOSError):
    status_code = 400
    code = "bad_request"


class AuthenticationError(InsightOSError):
    status_code = 401
    code = "unauthenticated"


class AuthorizationError(InsightOSError):
    status_code = 403
    code = "forbidden"


class NotFoundError(InsightOSError):
    status_code = 404
    code = "not_found"


class ConflictError(InsightOSError):
    status_code = 409
    code = "conflict"


class PayloadTooLargeError(InsightOSError):
    status_code = 413
    code = "payload_too_large"


class UnsupportedMediaTypeError(InsightOSError):
    status_code = 415
    code = "unsupported_media_type"


class QuotaExceededError(InsightOSError):
    """A free-tier guard rail was hit. Always actionable, never a bug."""

    status_code = 429
    code = "quota_exceeded"


class RateLimitedError(InsightOSError):
    status_code = 429
    code = "rate_limited"

    def __init__(self, message: str, *, retry_after: int = 60, details: Any = None):
        super().__init__(message, details=details)
        self.retry_after = retry_after


class UnsafeQueryError(InsightOSError):
    """Raised by the SQL guard when a statement is not provably read-only."""

    status_code = 400
    code = "unsafe_query"


class AnalysisError(InsightOSError):
    status_code = 422
    code = "analysis_error"


class ProviderUnavailableError(InsightOSError):
    """An optional AI provider could not be reached. Never fatal — callers fall
    back to the deterministic engine."""

    status_code = 503
    code = "provider_unavailable"


class ServiceUnavailableError(InsightOSError):
    status_code = 503
    code = "service_unavailable"
