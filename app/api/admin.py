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
from app.repositories.supabase_user_repository import SupabaseUserRepository
from app.repositories.user_repository import (
    RoleAssignmentRefused,
    UserAccountNotFound,
)
from app.schemas.admin import (
    AssignRoleRequest,
    ResolveMissingPetRequest,
    ReviewReportRequest,
    VerifySightingRequest,
)
from app.schemas.common import PaginatedData, StandardResponse
from app.services.admin_service import AdminService
from app.services.pet_service import PetService

router = APIRouter(prefix="/admin", tags=["Admin"])


def get_admin_service(supabase=Depends(get_supabase_client)) -> AdminService:
    return AdminService(
        repo=SupabaseAdminRepository(supabase),
        report_repo=SupabaseReportRepository(supabase),
        user_repo=SupabaseUserRepository(supabase),
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
    species: str | None = Query(
        None,
        description="Optional filter: Cat, Dog, Bird, or Other.",
    ),
    repo: MissingPetRepository = Depends(get_missing_pet_repository),
    admin_id: str = Depends(require_admin),
):
    """MD-41 / SRS-71 — browse every missing-pet report for moderation.

    Returns `{items, total, limit, offset}`. `total` counts every report
    matching `status`, not just this page, so the console can draw numbered
    pages instead of guessing whether another page exists.
    """
    try:
        page = await PetService.list_all_missing_pets(
            repo, limit=limit, offset=offset, status=status, species=species,
        )
        return StandardResponse(
            status="success",
            message=f"Retrieved {len(page.items)} of {page.total} reports.",
            data=PaginatedData(
                items=page.items, total=page.total,
                limit=limit, offset=offset,
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list missing pets: {e}"
        )

@router.get("/missing-pets/{pet_id}")
async def get_missing_pet(
    pet_id: str,
    repo: MissingPetRepository = Depends(get_missing_pet_repository),
    admin_id: str = Depends(require_admin),
):
    """Get a specific missing pet for admin details view."""
    pet = await PetService.get_missing_pet_by_id(repo, pet_id)
    if not pet:
        raise HTTPException(status_code=404, detail="Pet not found")
    return pet

@router.delete("/missing-pets/{pet_id}", response_model=StandardResponse)
async def remove_missing_pet(
    pet_id: str,
    repo: MissingPetRepository = Depends(get_missing_pet_repository),
    admin_id: str = Depends(require_admin),
):
    """
    MD-42 / SRS-70 — remove a report that violates the platform guidelines.

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
    MD-52 — browse the moderation flag queue.

    The listing `PATCH /admin/reports/{report_id}` acts from: without it an
    administrator can only review a flag whose identifier they already hold,
    which no screen could supply.

    Returns `{items, total, limit, offset}`; `total` is the depth of the queue
    for the requested status, which is the number a moderator works against.
    """
    try:
        page = await service.list_reports(
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
        message=f"Retrieved {len(page.items)} of {page.total} flags.",
        data=PaginatedData(
            items=page.items, total=page.total, limit=limit, offset=offset,
        ),
    )


@router.patch("/reports/{report_id}", response_model=StandardResponse)
async def review_report(
    report_id: str,
    payload: ReviewReportRequest,
    service: AdminService = Depends(get_admin_service),
    admin_id: str = Depends(require_admin),
):
    """
    MD-44 / SRS-73 — dismiss a flag, or uphold it and deduct the hunter's score.

    Upholding sets the flagged sighting to Dismissed and subtracts score from
    the hunter who submitted it (UD-16 steps 4-5). **No account is banned or
    suspended** — that sanction was replaced by the deduction on 2026-08-20; see
    the `admin_service` module docstring for why. `penalty_points` overrides the
    per-reason default. A flag another admin has already decided returns 409
    rather than being overwritten (UD-16 [E1]).
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


# ---------------------------------------------------------------------- #
# Role assignment — UD-23 / SRS-94..98. Access control, not moderation:
# nothing below can suspend, deactivate, or delete an account. Listing is
# limited to the current admin set (always small); there is no route that
# lists or searches all user accounts.
# ---------------------------------------------------------------------- #
@router.get("/users/lookup", response_model=StandardResponse)
async def find_user_by_email(
    email: str = Query(
        ...,
        description="The account's exact email address. Matched in full and "
                    "case-insensitively — this is not a search.",
    ),
    service: AdminService = Depends(get_admin_service),
    admin_id: str = Depends(require_admin),
):
    """
    MD-57 / SRS-94 — resolve ONE account from its exact email address.

    The lookup `PATCH /admin/users/{id}/role` acts from: without it an
    administrator can only change the role of an account whose identifier they
    already hold, which no screen could supply.

    Returns one account or 404. It is not a listing and never will be: an
    address that nearly matches discloses nothing, because the account browse
    was struck on 2026-08-21 and returning a page here would rebuild it under
    another name.
    """
    try:
        account = await service.find_user_by_email(email)
    except UserAccountNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to look up account: {e}"
        )
    return StandardResponse(
        status="success",
        message="Account found.",
        data=account,
    )


@router.get("/users/admins", response_model=StandardResponse)
async def list_admins(
    service: AdminService = Depends(get_admin_service),
    admin_id: str = Depends(require_admin),
):
    """Every account currently holding the admin role.

    Not paginated: the admin set is a handful of people by design. This is
    NOT the account listing struck on 2026-08-21 — it shows only the people
    who can sign into this console, which is a property of the console, not
    of the user base.
    """
    try:
        admins = await service.list_admins()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list administrators: {e}"
        )
    return StandardResponse(
        status="success",
        message=f"{len(admins)} administrator(s).",
        data=admins,
    )


@router.patch("/users/{target_user_id}/role", response_model=StandardResponse)
async def assign_user_role(
    target_user_id: str,
    payload: AssignRoleRequest,
    service: AdminService = Depends(get_admin_service),
    admin_id: str = Depends(require_admin),
):
    """
    MD-58 / SRS-95-98 — grant administrator access to an account, or withdraw it.

    **This is the only thing an administrator does to an account.** Setting
    'user' withdraws console access and leaves everything else alone — the
    account keeps its username, photograph, score, sightings and reports, and
    goes on using the product exactly as before. Nobody is suspended or banned;
    that sanction was struck on 2026-08-21 and the sanction for an abusive
    hunter remains the score deduction of `PATCH /admin/reports/{id}`.

    Two changes are refused with 409: withdrawing your own access (ask another
    administrator), and any change that would leave no administrator at all.
    Both are enforced inside the database procedure, so they hold when two
    administrators act at the same moment.

    Assigning a role the account already holds returns `changed: false` and
    writes no audit row.

    The withdrawal takes effect on the affected account's NEXT REQUEST — their
    session is not ended, because `require_admin` re-reads the role from the
    database every time rather than trusting the token.
    """
    try:
        result = await service.assign_user_role(
            target_user_id, payload.role, admin_id,
        )
    except UserAccountNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RoleAssignmentRefused as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to assign role: {e}"
        )
    changed = bool(result.get("changed"))
    return StandardResponse(
        status="success",
        message=(
            f"Role set to {result.get('role_after')}." if changed
            else f"Account already holds the role {result.get('role_after')}. "
                 "Nothing changed."
        ),
        data=result,
    )


@router.get("/role-changes", response_model=StandardResponse)
async def list_role_changes(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    target_user_id: str | None = Query(
        None,
        description="Optional: narrow the history to one account's changes.",
    ),
    service: AdminService = Depends(get_admin_service),
    admin_id: str = Depends(require_admin),
):
    """
    MD-59 / SRS-97 — the record of every role grant and withdrawal, newest first.

    Append-only: nothing edits or deletes a row, so an account's access history
    stays complete. Returns `{items, total, limit, offset}`.
    """
    try:
        page = await service.list_role_changes(
            limit=limit, offset=offset, target_user_id=target_user_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list role changes: {e}"
        )
    return StandardResponse(
        status="success",
        message=f"Retrieved {len(page.items)} of {page.total} role changes.",
        data=PaginatedData(
            items=page.items, total=page.total, limit=limit, offset=offset,
        ),
    )
