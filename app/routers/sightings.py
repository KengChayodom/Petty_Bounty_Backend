from fastapi import APIRouter, HTTPException, Depends
from app.schemas.sightings import SightingCreate, AnalyzeRequest
from app.schemas.common import StandardResponse
from app.core.database import get_supabase_client
from app.services.sighting_service import SightingService
from app.services.ai_service import AIManager

router = APIRouter(prefix="/sightings", tags=["Sightings"])

def get_sighting_service(supabase = Depends(get_supabase_client)) -> SightingService:
    
    return SightingService(db_client=supabase, ai_manager=AIManager)

@router.post("/analyze", response_model=StandardResponse)
async def analyze_image(
    request: AnalyzeRequest,
    service: SightingService = Depends(get_sighting_service)
):
    try:
        result = await service.analyze_sighting_image(request.image_url)
        return StandardResponse(
            status=result["status"],
            message=result["message"],
            data=result.get("data")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Analysis Error: {str(e)}")

@router.post("/", response_model=StandardResponse)
async def report_sighting(
    sighting: SightingCreate,
    service: SightingService = Depends(get_sighting_service)
):
    try:
 
        data = await service.process_and_save_sighting(sighting)
        return StandardResponse(
            status="success", 
            message="Sighting reported successfully.",
            data=data
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process sighting: {str(e)}")
    
@router.get("/{sighting_id}/matches")
async def get_ranking(
    sighting_id: str, 
    limit: int = 5,
    service: SightingService = Depends(get_sighting_service)
):
    try:

        matches = await service.get_matches(sighting_id, limit)
        return {
            "status": "success",
            "matches": matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking query failed: {str(e)}")