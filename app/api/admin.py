from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import require_admin
from app.core.database import get_supabase_client
from app.schemas.admin import ResolveMissingPetRequest, VerifySightingRequest
from app.schemas.common import StandardResponse
from app.services.admin_service import AdminService

router = APIRouter(prefix="/admin", tags=["Admin"])


def get_admin_service(supabase=Depends(get_supabase_client)) -> AdminService:
    return AdminService(db_client=supabase)


@router.patch(
    "/sightings/{sighting_id}/verification",
    response_model=StandardResponse,
)
async def verify_sighting(
    sighting_id: str,
    payload: VerifySightingRequest,
    service: AdminService = Depends(get_admin_service),
    admin_id: str = Depends(require_admin),
):
    """Admin sets a sighting's verification_status to 'Verified' or 'Dismissed'."""
    try:
        row = await service.verify_sighting(
            sighting_id, payload.verification_status,
        )
        return StandardResponse(
            status="success",
            message=f"Sighting marked {payload.verification_status}.",
            data=row,
        )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to verify sighting: {e}"
        )


@router.get(
    "/missing-pets/{pet_id}/sighting-timeline",
    response_model=StandardResponse,
)
async def get_sighting_timeline(
    pet_id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    service: AdminService = Depends(get_admin_service),
    admin_id: str = Depends(require_admin),
):
    """Full sighting timeline for a pet — includes Dismissed entries."""
    try:
        data = await service.get_sighting_timeline(
            pet_id, limit=limit, offset=offset,
        )
        return StandardResponse(
            status="success",
            message=f"Retrieved {len(data)} timeline entries.",
            data=data,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load timeline: {e}"
        )


@router.post(
    "/missing-pets/{pet_id}/resolve",
    response_model=StandardResponse,
)
async def resolve_missing_pet(
    pet_id: str,
    payload: ResolveMissingPetRequest,
    service: AdminService = Depends(get_admin_service),
    admin_id: str = Depends(require_admin),
):
    """
    Atomic resolution: pays the full bounty to the final hunter and
    distributes F1 ranking points (25/15/10/5/5/…) to every other hunter
    who submitted a Verified sighting for this pet. Sets pet.status =
    'Resolved'. Idempotent — second call returns 400.
    """
    try:
        result = await service.resolve_missing_pet(
            pet_id=pet_id,
            final_sighting_id=payload.final_sighting_id,
            slip_image_url=payload.slip_image_url,
            reference_no=payload.reference_no,
            verified_by=admin_id,
        )
        return StandardResponse(
            status="success",
            message="Missing pet resolved; bounty paid and clue scores distributed.",
            data=result,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to resolve pet: {e}"
        )
