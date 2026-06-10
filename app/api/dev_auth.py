"""
DEV-ONLY auth helpers — NOT a production auth path.

Thin convenience proxies to Supabase GoTrue so you can mint a bearer token from
/docs (without curl), copy the access_token, and use it to call protected
endpoints. The real auth architecture is unchanged: Flutter still talks to
GoTrue directly and FastAPI only verifies JWTs — this router does not introduce
a new auth system.

This module is mounted ONLY when settings.ENABLE_DEV_AUTH is true (see
main.py). When the flag is off the routes do not exist at all (404).

Security notes:
  * Never log passwords or tokens (we log only the email + status code).
  * Returns the full token/user payload in the HTTP RESPONSE on purpose — that
    is the whole point (you copy the access_token). It is never written to logs.
"""
import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dev", tags=["Dev Auth (local only)"])


class DevLoginRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class DevRegisterRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)
    display_name: Optional[str] = None


def _gotrue_headers() -> dict:
    """Headers for GoTrue REST calls — uses the anon key, like the app does."""
    if not settings.SUPABASE_ANON_KEY:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_ANON_KEY is not set — required for dev auth.",
        )
    return {
        "apikey": settings.SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }


def _safe_error(resp: httpx.Response) -> str:
    """Extract GoTrue's error message (no secrets) for a clean passthrough."""
    try:
        data = resp.json()
        return (
            data.get("error_description")
            or data.get("msg")
            or data.get("message")
            or data.get("error")
            or "Supabase auth error"
        )
    except Exception:
        return f"Supabase auth error ({resp.status_code})"


@router.post("/login")
async def dev_login(payload: DevLoginRequest):
    """Password grant against Supabase GoTrue; returns the full token response."""
    url = f"{settings.SUPABASE_URL}/auth/v1/token?grant_type=password"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                url,
                headers=_gotrue_headers(),
                json={"email": payload.email, "password": payload.password},
            )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Supabase unreachable: {e}")

    if resp.status_code != 200:
        logger.warning("dev_login failed for %s: %s", payload.email, resp.status_code)
        raise HTTPException(status_code=resp.status_code, detail=_safe_error(resp))

    logger.info("dev_login ok for %s", payload.email)
    # { access_token, refresh_token, expires_in, token_type, user, ... }
    return resp.json()


@router.post("/register")
async def dev_register(payload: DevRegisterRequest):
    """Sign up via Supabase GoTrue; returns the created user (+ session if any).

    The existing `handle_new_user` trigger mirrors the new auth user into
    public.users — we do NOT touch the DB here, the trigger does its job.

    NOTE on email confirmation: if the Supabase project requires email
    confirmation, signup returns a user with NO session and /dev/login will
    return 400 ("Email not confirmed") until the address is confirmed. Disable
    "Confirm email" in Supabase Auth settings for frictionless local testing.
    """
    url = f"{settings.SUPABASE_URL}/auth/v1/signup"
    body: dict = {"email": payload.email, "password": payload.password}
    if payload.display_name:
        # Becomes user_metadata; handle_new_user reads display_name from it.
        body["data"] = {"display_name": payload.display_name}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, headers=_gotrue_headers(), json=body)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Supabase unreachable: {e}")

    if resp.status_code not in (200, 201):
        logger.warning("dev_register failed for %s: %s", payload.email, resp.status_code)
        raise HTTPException(status_code=resp.status_code, detail=_safe_error(resp))

    logger.info("dev_register ok for %s", payload.email)
    return resp.json()
