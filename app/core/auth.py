"""
Authentication utilities — Feature #6.

The backend is the trust boundary. Every protected route depends on
`get_current_user_id`, which validates the caller's Supabase JWT against
Supabase Auth (GoTrue) and returns the verified user id. Admin routes
additionally check `users.role == 'admin'` server-side — the client's
claimed role is never trusted.

Tokens arrive via FastAPI's `HTTPBearer` security scheme, which is what makes
Swagger UI (/docs) render the global "Authorize" button. The scheme runs with
`auto_error=False` so this module keeps full control of the failure shape
(custom 401 + `WWW-Authenticate: Bearer`) and so the `AUTH_DEV_BYPASS` fallback
still applies when no token is present. A raw `Authorization` header is also
read (hidden from the schema) as a fallback, so a bare token with no "Bearer "
prefix — which HTTPBearer rejects — is still accepted.

Two dev escape hatches exist, both OFF by default and meant only for local
work / the existing pytest + CLI smoke scripts:
  * AUTH_DEV_BYPASS      — missing/invalid token falls back to TEST_USER_ID.
  * ENABLE_UNAUTHED_ADMIN — admin routes return TEST_USER_ID with no check.
"""
from logging import getLogger

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.core.database import get_supabase_client

logger = getLogger(__name__)

# Hardcoded dev user id — only ever returned when a dev bypass flag is on.
TEST_USER_ID = "024dd692-8b4a-44b7-968c-f6f3ddac3f4c"

# HTTPBearer is what makes Swagger show the "Authorize" button and attach a
# padlock to every secured operation. auto_error=False: a missing/blank/non-
# Bearer header yields None instead of HTTPBearer raising its own 403, so OUR
# 401 message and the AUTH_DEV_BYPASS fallback below stay in charge.
bearer_scheme = HTTPBearer(
    auto_error=False,
    description=(
        "Supabase JWT access token. The 'Bearer ' prefix is added for you. "
        "Mint one locally via POST /dev/login, then paste just the access_token."
    ),
)


def _strip_bearer(authorization: str | None) -> str | None:
    """
    Pull the raw JWT out of an `Authorization: Bearer <jwt>` header.

    Returns None for anything that doesn't carry credentials — a missing header,
    whitespace only, or a bare scheme word (`Bearer` with no token). A bare
    token with no scheme is accepted as-is. JWTs contain no spaces, so splitting
    on whitespace is safe.
    """
    if not authorization:
        return None
    parts = authorization.split()
    if not parts:
        return None
    if len(parts) >= 2 and parts[0].lower() == "bearer":
        return parts[1] or None
    if len(parts) == 1 and parts[0].lower() != "bearer":
        return parts[0]
    return None


def _validate_token(token: str | None) -> str | None:
    """
    Validate a raw JWT with Supabase Auth and return the user id, or None if
    the token is absent/invalid. Never raises — callers decide whether a None
    means 401 or a dev-bypass fallback.

    This is the CORE validation logic and is intentionally unchanged from the
    original implementation (Supabase GoTrue `auth.get_user`).
    """
    if not token:
        return None
    try:
        response = get_supabase_client().auth.get_user(token)
    except Exception as exc:  # network error, expired/invalid token, etc.
        logger.warning("Token validation failed: %s", exc)
        return None
    if response and response.user:
        return response.user.id
    return None


def _resolve_user_id(authorization: str | None) -> str | None:
    """
    Validate the bearer token from a raw `Authorization` header value and return
    the user id (or None). Kept as a thin wrapper over `_validate_token` for
    backward compatibility with `require_admin` and the existing unit tests that
    call it directly with a header string.
    """
    return _validate_token(_strip_bearer(authorization))


def _extract_token(
    authorization: str | None,
    credentials: HTTPAuthorizationCredentials | None,
) -> str | None:
    """
    Resolve the raw JWT from the two sources, preferring the HTTPBearer scheme.

    * `credentials` — produced by `bearer_scheme` from a well-formed
      `Authorization: Bearer <jwt>` request (this is what the Swagger Authorize
      button sends). It is only honoured when it is a real
      `HTTPAuthorizationCredentials`; direct (non-FastAPI) callers pass the
      `Depends` sentinel, which we ignore so the unit tests that invoke these
      dependencies as plain functions keep working.
    * `authorization` — the raw header, run through `_strip_bearer`. This is the
      fallback that also accepts a BARE token with no "Bearer " prefix, which
      HTTPBearer would reject (requirement: accept tokens with or without it).
    """
    if isinstance(credentials, HTTPAuthorizationCredentials):
        token = (credentials.credentials or "").strip()
        if token:
            return token
    return _strip_bearer(authorization)


def get_current_user_id(
    authorization: str | None = Header(default=None, include_in_schema=False),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    """
    FastAPI dependency: the verified user id for a protected endpoint.

    `credentials` (HTTPBearer) drives the Swagger Authorize button; `authorization`
    is the hidden raw-header fallback for bare tokens / direct callers. Rejects
    with 401 when the token is missing/invalid, unless AUTH_DEV_BYPASS is enabled
    (then it falls back to TEST_USER_ID).
    """
    user_id = _validate_token(_extract_token(authorization, credentials))
    if user_id:
        return user_id
    if settings.AUTH_DEV_BYPASS:
        return TEST_USER_ID
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid authentication token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user_id_optional(
    authorization: str | None = Header(default=None, include_in_schema=False),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str | None:
    """Like `get_current_user_id` but returns None instead of raising 401."""
    user_id = _validate_token(_extract_token(authorization, credentials))
    if user_id:
        return user_id
    if settings.AUTH_DEV_BYPASS:
        return TEST_USER_ID
    return None


def require_admin(
    authorization: str | None = Header(default=None, include_in_schema=False),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    """
    Admin gate. Validates the JWT, then looks up `users.role` server-side and
    403s unless it is 'admin'. ENABLE_UNAUTHED_ADMIN short-circuits to
    TEST_USER_ID for local dev (see config.py for why it's off by default).
    """
    if settings.ENABLE_UNAUTHED_ADMIN:
        return TEST_USER_ID

    user_id = _validate_token(_extract_token(authorization, credentials))
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # maybe_single() returns data=None for a missing profile row instead of
        # raising — so "no profile" falls through to the 403 below (a missing
        # profile is an authorization condition, not a server error). A genuine
        # DB/transport failure still raises and surfaces as 500.
        result = (
            get_supabase_client()
            .table("users")
            .select("role")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )
        role = (getattr(result, "data", None) or {}).get("role")
    except Exception as exc:
        logger.error("Admin role lookup failed for %s: %s", user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not verify admin privileges.",
        )

    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required.",
        )
    return user_id
