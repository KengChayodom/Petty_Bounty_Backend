from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client
from app.repositories.supabase_sighting_repository import (
    SupabaseSightingRepository,
)
from app.schemas.common import StandardResponse
from app.services.ai_service import AIManager
from app.services.sighting_service import SightingService

router = APIRouter(prefix="/hunters", tags=["Hunters"])


def get_sighting_service(
    supabase=Depends(get_supabase_client),
) -> SightingService:
    return SightingService(
        repo=SupabaseSightingRepository(supabase), ai_manager=AIManager
    )


@router.get("/me/score", response_model=StandardResponse)
async def get_my_score(
    service: SightingService = Depends(get_sighting_service),
    user_id: str = Depends(get_current_user_id),
):
    """Cumulative scoring + activity stats for the current Bounty Hunter."""
    try:
        stats = await service.get_hunter_stats(user_id)
        return StandardResponse(
            status="success",
            message="Hunter stats retrieved successfully.",
            data=stats,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve hunter stats: {e}"
        )
