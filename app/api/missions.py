from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from app.schemas.common import StandardResponse
from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client
from datetime import datetime

router = APIRouter(prefix="/missions", tags=["Missions"])


@router.post("/accept/{sighting_id}", response_model=StandardResponse)
async def accept_mission(
    sighting_id: str,
    supabase = Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    try:
        # Fetch the sighting first
        sighting_res = supabase.table("sightings")\
            .select("*")\
            .eq("id", sighting_id)\
            .execute()

        if not sighting_res.data:
            raise HTTPException(status_code=404, detail=f"Sighting {sighting_id} not found")

        sighting = sighting_res.data[0]

        # Check if already accepted
        if sighting.get("sighting_status") == "Confirmed":
            return StandardResponse(
                status="success",
                message="Mission already accepted.",
                data=sighting
            )

        # Update sighting status to Confirmed
        update_res = supabase.table("sightings")\
            .update({
                "sighting_status": "Confirmed",
                "updated_at": datetime.utcnow().isoformat()
            })\
            .eq("id", sighting_id)\
            .execute()

        if not update_res.data:
            raise HTTPException(status_code=500, detail="Failed to update sighting status")

        # Create activity log entry
        activity_data = {
            "user_id": user_id,
            "sighting_id": sighting_id,
            "action_type": "Mission_Accepted",
            "metadata": {
                "accepted_at": datetime.utcnow().isoformat()
            }
        }

        try:
            supabase.table("activity_logs").insert(activity_data).execute()
        except Exception:
            # Log error but don't fail the request
            pass

        return StandardResponse(
            status="success",
            message="Mission accepted successfully.",
            data=update_res.data[0]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to accept mission: {str(e)}")


@router.get("/active", response_model=StandardResponse)
async def get_active_missions(
    supabase = Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    try:
        # Confirmed sightings for the authenticated hunter.
        response = supabase.table("sightings")\
            .select("*")\
            .eq("hunter_id", user_id)\
            .eq("sighting_status", "Confirmed")\
            .order("created_at", desc=True)\
            .execute()

        return StandardResponse(
            status="success",
            message=f"Found {len(response.data)} active missions.",
            data=response.data
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch active missions: {str(e)}")


@router.patch("/{sighting_id}/status", response_model=StandardResponse)
async def update_mission_status(
    sighting_id: str,
    status: str,
    notes: Optional[str] = None,
    supabase = Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    valid_statuses = ["Confirmed", "Closed"]

    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {valid_statuses}"
        )

    try:
        # Update sighting status
        update_data = {
            "sighting_status": status,
            "updated_at": datetime.utcnow().isoformat()
        }

        if notes:
            update_data["notes"] = notes

        response = supabase.table("sightings")\
            .update(update_data)\
            .eq("id", sighting_id)\
            .eq("hunter_id", user_id)\
            .execute()

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail=f"Sighting {sighting_id} not found or not owned by user"
            )

        return StandardResponse(
            status="success",
            message="Mission status updated successfully.",
            data=response.data[0]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update mission status: {str(e)}")


@router.get("/history", response_model=StandardResponse)
async def get_mission_history(
    limit: int = 20,
    supabase = Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    try:
        response = supabase.table("sightings")\
            .select("*")\
            .eq("hunter_id", user_id)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()

        return StandardResponse(
            status="success",
            message=f"Found {len(response.data)} missions in history.",
            data=response.data
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch mission history: {str(e)}")
