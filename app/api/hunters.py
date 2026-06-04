from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import TEST_USER_ID
from app.core.database import get_supabase_client
from app.schemas.common import StandardResponse
from app.services.ai_service import AIManager
from app.services.sighting_service import SightingService

router = APIRouter(prefix="/hunters", tags=["Hunters"])


def get_sighting_service(
    supabase=Depends(get_supabase_client),
) -> SightingService:
    return SightingService(db_client=supabase, ai_manager=AIManager)


@router.get("/me/score", response_model=StandardResponse)
async def get_my_score(
    hunter_id: str | None = Query(
        None,
        description="Override for the auth-bypass period. Remove once "
                    "Feature #6 lands and the JWT identifies the caller.",
    ),
    service: SightingService = Depends(get_sighting_service),
):
    """Cumulative scoring + activity stats for the current Bounty Hunter."""
    try:
        stats = await service.get_hunter_stats(hunter_id or TEST_USER_ID)
        return StandardResponse(
            status="success",
            message="Hunter stats retrieved successfully.",
            data=stats,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve hunter stats: {e}"
        )
