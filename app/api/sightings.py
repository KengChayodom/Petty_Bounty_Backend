from fastapi import APIRouter, Depends, HTTPException

from app.core.database import get_supabase_client
from app.schemas.common import StandardResponse
from app.schemas.sightings import AnalyzeRequest, SightingCreate
from app.services.ai_service import AIManager
from app.services.sighting_service import SightingService

router = APIRouter(prefix="/sightings", tags=["Sightings"])

# Hardcoded test user ID — bypasses auth until real auth is wired up.
TEST_USER_ID = "024dd692-8b4a-44b7-968c-f6f3ddac3f4c"


def get_sighting_service(
    supabase=Depends(get_supabase_client),
) -> SightingService:
    return SightingService(db_client=supabase, ai_manager=AIManager)


@router.post("/analyze", response_model=StandardResponse)
async def analyze_image(
    request: AnalyzeRequest,
    service: SightingService = Depends(get_sighting_service),
):
    """
    Heavy step. Runs YOLO-seg + mask + CLIP and caches the feature vector
    so the follow-up POST /sightings/ only has to write the row.
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
):
    """
    Hot step. Pulls the cached feature vector, INSERTs the sighting with
    the USER-CONFIRMED species (may override YOLO's guess), and runs the
    pgvector match RPC. Returns {sighting, matches} so the client doesn't
    need a separate /matches call.
    """
    sighting.hunter_id = TEST_USER_ID
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
