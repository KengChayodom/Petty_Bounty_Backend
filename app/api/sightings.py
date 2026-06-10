from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client
from app.schemas.common import StandardResponse
from app.schemas.sightings import AnalyzeRequest, SightingCreate
from app.services.ai_service import AIManager
from app.services.sighting_service import SightingService

router = APIRouter(prefix="/sightings", tags=["Sightings"])


def get_sighting_service(
    supabase=Depends(get_supabase_client),
) -> SightingService:
    return SightingService(db_client=supabase, ai_manager=AIManager)


@router.post("/analyze", response_model=StandardResponse)
async def analyze_image(
    request: AnalyzeRequest,
    service: SightingService = Depends(get_sighting_service),
    user_id: str = Depends(get_current_user_id),
):
    """
    Heavy step. Runs YOLO-seg + mask + CLIP and caches the feature vector
    so the follow-up POST /sightings/ only has to write the row.

    Authenticated (Feature #6): this is the most expensive call in the app,
    so it must not be reachable without a valid session.
    """
    try:
        result = await service.analyze_sighting_image(request.image_url)
        return StandardResponse(
            status=result["status"],
            message=result["message"],
            data=result.get("data"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Analysis Error: {e}")


@router.post("/", response_model=StandardResponse)
async def report_sighting(
    sighting: SightingCreate,
    service: SightingService = Depends(get_sighting_service),
    user_id: str = Depends(get_current_user_id),
):
    """
    Hot step. Pulls the cached feature vector, INSERTs the sighting with
    the USER-CONFIRMED species (may override YOLO's guess), and runs the
    pgvector match RPC. Returns {sighting, matches} so the client doesn't
    need a separate /matches call.
    """
    # Identity comes from the verified JWT, never from the request body.
    sighting.hunter_id = user_id
    try:
        result = await service.process_and_save_sighting(sighting)
        return StandardResponse(
            status="success",
            message=f"Sighting saved with {len(result['matches'])} matches.",
            data=result,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process sighting: {e}")


@router.get("/me", response_model=StandardResponse)
async def get_my_activity(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    service: SightingService = Depends(get_sighting_service),
    user_id: str = Depends(get_current_user_id),
):
    """Bounty Hunter activity log — own sightings + matches + earned points."""
    try:
        data = await service.get_hunter_activity(
            user_id, limit=limit, offset=offset,
        )
        return StandardResponse(
            status="success",
            message=f"Retrieved {len(data['sightings'])} sightings.",
            data=data,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve activity: {e}"
        )


@router.get("/{sighting_id}/matches")
async def get_ranking(
    sighting_id: str,
    limit: int = 5,
    threshold: float = 0.0,
    service: SightingService = Depends(get_sighting_service),
):
    """Re-query matches for a previously stored sighting."""
    try:
        matches = await service.get_matches(sighting_id, limit, threshold)
        return {"status": "success", "matches": matches}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking query failed: {e}")


@router.get("/{sighting_id}", response_model=StandardResponse)
async def get_sighting(
    sighting_id: str,
    service: SightingService = Depends(get_sighting_service),
):
    """Fetch a single sighting record (feature_vector stripped)."""
    try:
        sighting = await service.get_sighting_by_id(sighting_id)
        if not sighting:
            raise HTTPException(
                status_code=404, detail=f"Sighting {sighting_id} not found"
            )
        return StandardResponse(
            status="success",
            message="Sighting retrieved successfully.",
            data=sighting,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve sighting: {e}"
        )
