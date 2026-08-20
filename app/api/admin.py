from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import require_admin
from app.core.database import get_supabase_client
from app.repositories.missing_pet_repository import MissingPetRepository
from app.repositories.report_repository import (
    ReportAlreadyModerated,
    ReportNotFound,
)
from app.repositories.supabase_admin_repository import SupabaseAdminRepository
from app.repositories.supabase_missing_pet_repository import (
    SupabaseMissingPetRepository,
)
from app.repositories.supabase_report_repository import SupabaseReportRepository
from app.schemas.admin import (
    ResolveMissingPetRequest,
    ReviewReportRequest,
    VerifySightingRequest,
)
from app.schemas.common import StandardResponse
from app.services.admin_service import AdminService
from app.services.pet_service import PetService

router = APIRouter(prefix="/admin", tags=["Admin"])


def get_admin_service(supabase=Depends(get_supabase_client)) -> AdminService:
    return AdminService(
        repo=SupabaseAdminRepository(supabase),
        report_repo=SupabaseReportRepository(supabase),
    )


def get_missing_pet_repository(
    supabase=Depends(get_supabase_client),
) -> MissingPetRepository:
    return SupabaseMissingPetRepository(supabase)


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
    """Admin sets a sighting's verification_status to 'Verified' or 'Dismissed'.

    'Dismissed' is the live half: it withdraws a sighting from every owner
    timeline and from scoring, and it is what upholding a flag writes. 'Verified'
    no longer gates anything — the owner's confirmation took over both the
    scoring (2026-08-21) and the bounty eligibility — and is kept only so an
    administrator can undo a dismissal.
    """
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


@router.get("/missing-pets", response_model=StandardResponse)
async def list_all_missing_pets(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: str | None = Query(
        None,
        description="Optional filter: Searching, Spotted, Found, or Resolved.",
    ),
    repo: MissingPetRepository = Depends(get_missing_pet_repository),
    admin_id: str = Depends(require_admin),
):
    """MD-37 / SRS-64 — browse every missing-pet report for moderation."""
    try:
        pets = await PetService.list_all_missing_pets(
            repo, limit=limit, offset=offset, status=status,
        )
        return StandardResponse(
            status="success",
            message=f"Retrieved {len(pets)} reports.",
            data=pets,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list missing pets: {e}"
        )


@router.delete("/missing-pets/{pet_id}", response_model=StandardResponse)
async def remove_missing_pet(
    pet_id: str,
    repo: MissingPetRepository = Depends(get_missing_pet_repository),
    admin_id: str = Depends(require_admin),
):
    """
    MD-38 / SRS-66 — remove a report that violates the platform guidelines.

    UD-14's postcondition is "removed from the database and the search map", so
    the row is deleted outright; the moderation action is recorded in the log.
    """
    try:
        removed = await PetService.remove_missing_pet(repo, pet_id, admin_id)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to remove missing pet: {e}"
        )
    return StandardResponse(
        status="success",
        message="Missing pet report removed.",
        data=removed,
    )


@router.get("/reports", response_model=StandardResponse)
async def list_reports(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: str | None = Query(
        None,
        description="Optional filter: Pending, Reviewed_Penalty, or "
                    "Dismissed. Omit for every status.",
    ),
    service: AdminService = Depends(get_admin_service),
    admin_id: str = Depends(require_admin),
):
    """
    MD-51 — browse the moderation flag queue.

    The listing `PATCH /admin/reports/{report_id}` acts from: without it an
    administrator can only review a flag whose identifier they already hold,
    which no screen could supply.
    """
    try:
        flags = await service.list_reports(
            status=status, limit=limit, offset=offset,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list reports: {e}"
        )
    return StandardResponse(
        status="success",
        message=f"Retrieved {len(flags)} flags.",
        data=flags,
    )


@router.patch("/reports/{report_id}", response_model=StandardResponse)
async def review_report(
    report_id: str,
    payload: ReviewReportRequest,
    service: AdminService = Depends(get_admin_service),
    admin_id: str = Depends(require_admin),
):
    """
    MD-40 / SRS-68 — dismiss a flag, or uphold it and deduct the hunter's score.

    Upholding sets the flagged sighting to Dismissed and subtracts score from
    the hunter who submitted it (UD-14 steps 4-5). **No account is banned or
    suspended** — that sanction was replaced by the deduction on 2026-08-20; see
    the `admin_service` module docstring for why. `penalty_points` overrides the
    per-reason default. A flag another admin has already decided returns 409
    rather than being overwritten (UD-14 [E1]).
    """
    try:
        result = await service.review_report(
            report_id, payload.decision, admin_id,
            penalty_points=payload.penalty_points,
        )
    except ReportNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ReportAlreadyModerated as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to review report: {e}"
        )
    return StandardResponse(
        status="success",
        message="Moderation decision recorded.",
        data=result,
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
    Settle the bounty on a case the OWNER has already closed.

    Records the transfer against the sighting the owner confirmed as the catch
    and moves the report from 'Found' to 'Resolved'. Since 2026-08-21 it does
    NOT award any points: the clue scores were distributed when the owner
    confirmed the rescue, possibly days earlier (see
    `SightingService.decide_match`), and awarding here as well would pay every
    hunter twice.

    400 when the pet was never recovered ('Found' is the precondition), when
    the nominated sighting is not the owner-confirmed catch, or when the bounty
    was already settled.
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
