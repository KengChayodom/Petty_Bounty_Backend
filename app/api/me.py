"""
Current-user presence — hunter location updates for geo-targeted push
(SRS-FR-12).

The client pushes the device's current position (from the existing geolocator
in home_map) so the backend can answer "which hunters are near this newly
reported pet" via the get_nearby_hunters RPC.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_user_service
from app.core.auth import get_current_user_id
from app.repositories.user_repository import UserProfileNotFound
from app.schemas.common import StandardResponse
from app.services.user_service import UserService

# Every route here needs a signed-in caller, declared ONCE for the whole router so a route added later inherits it.
# The routes still ask for the id where they use it, which FastAPI resolves once per request.
router = APIRouter(
    prefix="/me", tags=["Me"],
    dependencies=[Depends(get_current_user_id)],
)


class LocationUpdateRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)


@router.post("/location", response_model=StandardResponse)
async def update_my_location(
    payload: LocationUpdateRequest,
    service: UserService = Depends(get_user_service),
    user_id: str = Depends(get_current_user_id),
):
    """Write the caller's current location + timestamp onto their users row."""
    try:
        updated = await service.update_location(
            user_id, payload.latitude, payload.longitude,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update location: {e}"
        )
    if not updated:
        raise HTTPException(status_code=404, detail="User profile not found.")
    return StandardResponse(
        status="success", message="Location updated.", data=None
    )


class ProfileUpdateRequest(BaseModel):
    """Profile edit payload (username, phone and photo). Every field
    is optional so the client can save any one alone or all three together, but
    at least one must be present — an empty PATCH is rejected. `username`
    carries the username (stored in the users.username column);
    `photo_url` is the Object Storage address of a pre-uploaded picture;
    `phone` is the mobile number (users.phone).

    `phone` carries no format rule on purpose. The column is free text, the
    project has never specified one (the sign-up form does not check a format
    either), and inventing one here would reject numbers the same account could
    already have registered with.
    """
    username: str | None = None
    photo_url: str | None = None
    phone: str | None = None


@router.patch("", response_model=StandardResponse)
async def update_my_profile(
    payload: ProfileUpdateRequest,
    service: UserService = Depends(get_user_service),
    user_id: str = Depends(get_current_user_id),
):
    """Edit the caller's own profile — username, phone and/or photo.

    Caller identity comes solely from the JWT; the update is self-scoped to that
    row in `users`. Validation mirrors the use case: a blank username -> 400, an
    unsupported photo format -> 400, a missing profile row -> 404.
    """
    try:
        updated = await service.update_profile(
            user_id,
            username=payload.username,
            photo_url=payload.photo_url,
            phone=payload.phone,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except UserProfileNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update profile: {e}"
        )

    if updated is None:
        raise HTTPException(status_code=404, detail="User profile not found.")

    return StandardResponse(
        status="success",
        message="Profile updated successfully.",
        data=updated,
    )
