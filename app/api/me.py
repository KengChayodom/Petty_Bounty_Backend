"""
Current-user presence — hunter location updates for geo-targeted push
(SRS-FR-12).

The client pushes the device's current position (from the existing geolocator
in home_map) so the backend can answer "which hunters are near this newly
reported pet" via the get_nearby_hunters RPC.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client
from app.schemas.common import StandardResponse
from app.utils.postgis import create_postgis_point

router = APIRouter(prefix="/me", tags=["Me"])


class LocationUpdateRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)


@router.post("/location", response_model=StandardResponse)
async def update_my_location(
    payload: LocationUpdateRequest,
    supabase=Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    """Write the caller's current location + timestamp onto their users row."""
    location_point = create_postgis_point(payload.latitude, payload.longitude)
    try:
        res = (
            supabase.table("users")
            .update(
                {
                    "last_location": location_point,
                    "last_location_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            .eq("id", user_id)
            .execute()
        )
        if not res.data:
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
