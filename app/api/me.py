"""
Current-user presence — hunter location updates for geo-targeted push
(SRS-FR-12).

The client pushes the device's current position (from the existing geolocator
in home_map) so the backend can answer "which hunters are near this newly
reported pet" via the get_nearby_hunters RPC.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client
from app.repositories.supabase_user_repository import SupabaseUserRepository
from app.repositories.user_repository import UserProfileNotFound, UserRepository
from app.schemas.common import StandardResponse
from app.utils.postgis import create_postgis_point

router = APIRouter(prefix="/me", tags=["Me"])

# Object-Storage photo formats accepted for the profile picture (MD-47).
_ALLOWED_PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_user_repository(
    supabase=Depends(get_supabase_client),
) -> UserRepository:
    return SupabaseUserRepository(supabase)


class LocationUpdateRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)


@router.post("/location", response_model=StandardResponse)
async def update_my_location(
    payload: LocationUpdateRequest,
    repo: UserRepository = Depends(get_user_repository),
    user_id: str = Depends(get_current_user_id),
):
    """Write the caller's current location + timestamp onto their users row."""
    location_point = create_postgis_point(payload.latitude, payload.longitude)
    try:
        updated = repo.update_last_location(user_id, location_point)
        if not updated:
            raise HTTPException(
                status_code=404, detail="User profile not found."
            )
        return StandardResponse(
            status="success", message="Location updated.", data=None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update location: {e}"
        )


class ProfileUpdateRequest(BaseModel):
    """Profile edit payload (MD-46 username + phone, MD-47 photo). Every field
    is optional so the client can save any one alone or all three together, but
    at least one must be present — an empty PATCH is rejected. `display_name`
    carries the username (stored in the users.display_name column, SRS-73);
    `photo_url` is the Object Storage address of a pre-uploaded picture (SRS-74);
    `phone` is the mobile number (users.phone, SRS-99).

    `phone` carries no format rule on purpose. The column is free text, the
    project has never specified one (the sign-up form does not check a format
    either), and inventing one here would reject numbers the same account could
    already have registered with.
    """
    display_name: str | None = None
    photo_url: str | None = None
    phone: str | None = None


@router.patch("", response_model=StandardResponse)
async def update_my_profile(
    payload: ProfileUpdateRequest,
    repo: UserRepository = Depends(get_user_repository),
    user_id: str = Depends(get_current_user_id),
):
    """Edit the caller's own profile — username and phone (MD-46) and/or photo
    (MD-47).

    Caller identity comes solely from the JWT; the update is self-scoped to that
    row in `users`. Validation mirrors UD-17: a blank username -> 400, an
    unsupported photo format -> 400, a missing profile row -> 404.
    """
    patch: dict = {}

    if payload.display_name is not None:
        display_name = payload.display_name.strip()
        if not display_name:
            raise HTTPException(
                status_code=400, detail="Username cannot be empty."
            )
        patch["display_name"] = display_name

    if payload.photo_url is not None:
        photo_url = payload.photo_url.strip()
        if not photo_url.lower().endswith(_ALLOWED_PHOTO_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload JPG, JPEG, or PNG.",
            )
        patch["profile_image_url"] = photo_url

    if payload.phone is not None:
        patch["phone"] = payload.phone.strip()

    if not patch:
        raise HTTPException(
            status_code=400,
            detail="Provide a username, a phone number, or a photo to update.",
        )

    try:
        updated = repo.update_profile(user_id, patch)
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
