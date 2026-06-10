"""
Authentication utilities — Feature #6.

The backend is the trust boundary. Every protected route depends on
`get_current_user_id`, which validates the caller's Supabase JWT against
Supabase Auth (GoTrue) and returns the verified user id. Admin routes
additionally check `users.role == 'admin'` server-side — the client's
claimed role is never trusted.

Two dev escape hatches exist, both OFF by default and meant only for local
work / the existing pytest + CLI smoke scripts:
  * AUTH_DEV_BYPASS      — missing/invalid token falls back to TEST_USER_ID.
  * ENABLE_UNAUTHED_ADMIN — admin routes return TEST_USER_ID with no check.
"""
from logging import getLogger

from fastapi import Header, HTTPException, status

from app.core.config import settings
from app.core.database import get_supabase_client

logger = getLogger(__name__)

# Hardcoded dev user id — only ever returned when a dev bypass flag is on.
TEST_USER_ID = "024dd692-8b4a-44b7-968c-f6f3ddac3f4c"


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


def _resolve_user_id(authorization: str | None) -> str | None:
    """
    Validate the bearer token with Supabase Auth and return the user id,
    or None if the token is absent/invalid. Never raises — callers decide
    whether a None means 401 or a dev-bypass fallback.
    """
    token = _strip_bearer(authorization)
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


def get_current_user_id(authorization: str | None = Header(default=None)) -> str:
    """
    FastAPI dependency: the verified user id for a protected endpoint.

    Rejects with 401 when the token is missing/invalid, unless AUTH_DEV_BYPASS
    is enabled (then it falls back to TEST_USER_ID).
    """
    user_id = _resolve_user_id(authorization)
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
    authorization: str | None = Header(default=None),
) -> str | None:
    """Like `get_current_user_id` but returns None instead of raising 401."""
    user_id = _resolve_user_id(authorization)
    if user_id:
        return user_id
    if settings.AUTH_DEV_BYPASS:
        return TEST_USER_ID
    return None


def require_admin(authorization: str | None = Header(default=None)) -> str:
    """
    Admin gate. Validates the JWT, then looks up `users.role` server-side and
    403s unless it is 'admin'. ENABLE_UNAUTHED_ADMIN short-circuits to
    TEST_USER_ID for local dev (see config.py for why it's off by default).
    """
    if settings.ENABLE_UNAUTHED_ADMIN:
        return TEST_USER_ID

    user_id = _resolve_user_id(authorization)
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
