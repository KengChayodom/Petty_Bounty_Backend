"""
Device-token registration for FCM push (SRS-FR-12).

The Flutter client registers its FCM token here after login and again on
token refresh. Upsert is keyed by the UNIQUE fcm_token, so a token that moves
to a different account reassigns cleanly instead of duplicating.
"""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_device_service
from app.core.auth import get_current_user_id
from app.schemas.common import StandardResponse
from app.services.device_service import DeviceService

# Every route here needs a signed-in caller, declared ONCE for the whole router so a route added later inherits it.
# The routes still ask for the id where they use it, which FastAPI resolves once per request.
router = APIRouter(
    prefix="/devices", tags=["Devices"],
    dependencies=[Depends(get_current_user_id)],
)


class DeviceRegisterRequest(BaseModel):
    fcm_token: str = Field(..., min_length=1, description="FCM registration token")
    platform: Literal["android", "ios"] = Field(..., description="Client platform")


class DeviceUnregisterRequest(BaseModel):
    fcm_token: str = Field(..., min_length=1, description="FCM registration token")


@router.post("/register", response_model=StandardResponse)
async def register_device(
    payload: DeviceRegisterRequest,
    service: DeviceService = Depends(get_device_service),
    user_id: str = Depends(get_current_user_id),
):
    """Upsert the caller's FCM token (keyed on fcm_token)."""
    try:
        created = await service.register(
            user_id, payload.fcm_token, payload.platform,
        )
        return StandardResponse(
            status="success",
            message="Device token registered.",
            data=created,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to register device token: {e}"
        )


@router.post("/unregister", response_model=StandardResponse)
async def unregister_device(
    payload: DeviceUnregisterRequest,
    service: DeviceService = Depends(get_device_service),
    user_id: str = Depends(get_current_user_id),
):
    """Drop the caller's FCM token on logout.

    Scoped by BOTH user_id and fcm_token so a client can only delete its own
    token — never another account's. Deleting a token that is already gone
    (re-logout, token rotated) is a benign no-op, so the absence of a returned
    row is success, not a 404.
    """
    try:
        await service.unregister(user_id, payload.fcm_token)
        return StandardResponse(
            status="success",
            message="Device token unregistered.",
            data=None,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to unregister device token: {e}"
        )
