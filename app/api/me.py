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
from app.repositories.user_repository import UserRepository
from app.schemas.common import StandardResponse
from app.utils.postgis import create_postgis_point

router = APIRouter(prefix="/me", tags=["Me"])


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
