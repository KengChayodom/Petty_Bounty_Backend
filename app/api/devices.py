"""
Device-token registration for FCM push (SRS-FR-12).

The Flutter client registers its FCM token here after login and again on
token refresh. Upsert is keyed by the UNIQUE fcm_token, so a token that moves
to a different account reassigns cleanly instead of duplicating.
"""
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client
from app.schemas.common import StandardResponse

router = APIRouter(prefix="/devices", tags=["Devices"])


class DeviceRegisterRequest(BaseModel):
    fcm_token: str = Field(..., min_length=1, description="FCM registration token")
    platform: Literal["android", "ios"] = Field(..., description="Client platform")


class DeviceUnregisterRequest(BaseModel):
    fcm_token: str = Field(..., min_length=1, description="FCM registration token")


@router.post("/register", response_model=StandardResponse)
async def register_device(
    payload: DeviceRegisterRequest,
    supabase=Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    """Upsert the caller's FCM token (keyed on fcm_token)."""
    row = {
        "user_id": user_id,
        "fcm_token": payload.fcm_token,
        "platform": payload.platform,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        res = (
            supabase.table("device_tokens")
            .upsert(row, on_conflict="fcm_token")
            .execute()
        )
        if not res.data:
            raise ValueError("Token upsert returned no row")
        return StandardResponse(
            status="success",
            message="Device token registered.",
            data=res.data[0],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to register device token: {e}"
        )


@router.post("/unregister", response_model=StandardResponse)
async def unregister_device(
    payload: DeviceUnregisterRequest,
    supabase=Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    """Drop the caller's FCM token on logout (SRS-20).

    Scoped by BOTH user_id and fcm_token so a client can only delete its own
    token — never another account's. Deleting a token that is already gone
    (re-logout, token rotated) is a benign no-op, so the absence of a returned
    row is success, not a 404.
    """
    try:
        (
            supabase.table("device_tokens")
            .delete()
            .eq("user_id", user_id)
            .eq("fcm_token", payload.fcm_token)
            .execute()
        )
        return StandardResponse(
            status="success",
            message="Device token unregistered.",
            data=None,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to unregister device token: {e}"
        )
