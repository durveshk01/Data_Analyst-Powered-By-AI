"""Token issuing/verification and password hashing.

Two authentication modes share one request-side contract (`AuthPrincipal`):

* ``local``    — InsightOS mints HS256 JWTs signed with ``APP_JWT_SECRET`` and
                 stores bcrypt password hashes itself. Requires no external
                 service, which is what makes offline/CI operation possible.
* ``supabase`` — Sign-up/sign-in are delegated to Supabase Auth (free tier).
                 Access tokens are verified against the project JWT secret
                 (HS256) or, for projects using asymmetric signing keys, the
                 published JWKS.

Scoped upload tokens are always InsightOS-issued so that the browser never
needs to hold a long-lived credential in JavaScript.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import bcrypt
import jwt
from jwt import PyJWKClient

from app.config import settings
from app.errors import AuthenticationError
from app.obs.logging import get_logger

log = get_logger(__name__)

TokenKind = Literal["access", "refresh", "upload"]
_ALGORITHM = "HS256"
_ISSUER = "insightos"

# bcrypt refuses inputs longer than 72 bytes; pre-truncate explicitly so a long
# passphrase fails loudly at registration rather than silently at login.
_BCRYPT_MAX_BYTES = 72

_jwks_client: PyJWKClient | None = None


@dataclass(frozen=True, slots=True)
class AuthPrincipal:
    """The authenticated caller, normalised across both auth modes."""

    user_id: str
    email: str | None = None
    scope: TokenKind = "access"
    is_demo: bool = False

    @property
    def is_anonymous(self) -> bool:
        return self.is_demo


# --------------------------------------------------------------- passwords ---
def hash_password(password: str) -> str:
    raw = password.encode("utf-8")
    if len(raw) > _BCRYPT_MAX_BYTES:
        raise AuthenticationError("Password must be at most 72 bytes long.")
    return bcrypt.hashpw(raw, bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    raw = password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    try:
        return bcrypt.checkpw(raw, password_hash.encode("utf-8"))
    except ValueError:
        return False


# ------------------------------------------------------------------ tokens ---
def create_token(
    *,
    user_id: str,
    kind: TokenKind,
    email: str | None = None,
    ttl_seconds: int | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    now = int(time.time())
    ttl = ttl_seconds if ttl_seconds is not None else _default_ttl(kind)
    payload: dict[str, Any] = {
        "sub": user_id,
        "iss": _ISSUER,
        "iat": now,
        "nbf": now,
        "exp": now + ttl,
        "jti": uuid.uuid4().hex,
        "scope": kind,
    }
    if email:
        payload["email"] = email
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.app_jwt_secret, algorithm=_ALGORITHM)


def _default_ttl(kind: TokenKind) -> int:
    return {
        "access": settings.access_token_ttl_seconds,
        "refresh": settings.refresh_token_ttl_seconds,
        "upload": settings.upload_token_ttl_seconds,
    }[kind]


def decode_app_token(token: str, *, expected_scope: TokenKind | None = None) -> AuthPrincipal:
    """Verify an InsightOS-issued token."""
    try:
        claims = jwt.decode(
            token,
            settings.app_jwt_secret,
            algorithms=[_ALGORITHM],
            issuer=_ISSUER,
            options={"require": ["exp", "sub", "iss"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Invalid token.") from exc

    scope = claims.get("scope", "access")
    if expected_scope and scope != expected_scope:
        raise AuthenticationError(
            f"Token scope '{scope}' cannot be used for '{expected_scope}' operations."
        )
    return AuthPrincipal(
        user_id=str(claims["sub"]),
        email=claims.get("email"),
        scope=scope,
    )


def decode_supabase_token(token: str) -> AuthPrincipal:
    """Verify a Supabase Auth access token.

    Supabase projects historically sign with a shared HS256 secret; newer
    projects can use asymmetric keys published via JWKS. Both are supported so
    the deployment does not break when a project rotates to ECC keys.
    """
    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Malformed token.") from exc

    alg = header.get("alg", _ALGORITHM)
    options = {"require": ["exp", "sub"], "verify_aud": False}

    try:
        if alg == _ALGORITHM:
            if not settings.supabase_jwt_secret:
                raise AuthenticationError(
                    "SUPABASE_JWT_SECRET is not configured; cannot verify tokens."
                )
            claims = jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=[_ALGORITHM],
                options=options,
            )
        else:
            signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token, signing_key.key, algorithms=[alg], options=options
            )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Session has expired. Please sign in again.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Invalid session token.") from exc
    except AuthenticationError:
        raise
    except Exception as exc:  # JWKS fetch failures etc.
        log.warning("supabase_token_verification_failed", extra={"error": str(exc)})
        raise AuthenticationError("Could not verify session token.") from exc

    return AuthPrincipal(
        user_id=str(claims["sub"]),
        email=claims.get("email"),
        scope="access",
    )


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        if not settings.supabase_url:
            raise AuthenticationError("SUPABASE_URL is not configured.")
        url = settings.supabase_url.rstrip("/") + "/auth/v1/.well-known/jwks.json"
        _jwks_client = PyJWKClient(url, cache_keys=True, lifespan=3600)
    return _jwks_client


def decode_access_token(token: str) -> AuthPrincipal:
    """Verify a session token using whichever mode the deployment runs in.

    Upload tokens are always InsightOS-issued, so in Supabase mode we attempt
    the local verifier as a second step. The scope check keeps the two classes
    of token from being interchangeable.
    """
    if settings.auth_mode == "supabase":
        try:
            return decode_supabase_token(token)
        except AuthenticationError:
            return decode_app_token(token)
    return decode_app_token(token)
