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

# Object-Storage photo formats accepted for the profile picture (MD-42/SRS-70).
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
    """Profile edit payload (MD-41 username + MD-42 photo). Both fields are
    optional so the client can save either one alone or both together, but at
    least one must be present — an empty PATCH is rejected. `display_name`
    carries the username (stored in the users.display_name column, SRS-69);
    `photo_url` is the Object Storage address of a pre-uploaded picture (SRS-70).
    """
    display_name: str | None = None
    photo_url: str | None = None


@router.patch("", response_model=StandardResponse)
async def update_my_profile(
    payload: ProfileUpdateRequest,
    repo: UserRepository = Depends(get_user_repository),
    user_id: str = Depends(get_current_user_id),
):
    """Edit the caller's own profile — username (MD-41) and/or photo (MD-42).

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

    if not patch:
        raise HTTPException(
            status_code=400,
            detail="Provide a username or a photo to update.",
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
