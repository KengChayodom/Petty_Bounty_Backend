# app/api/missing_pets.py
import logging

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field

from app.schemas.missing_pets import MissingPetCreate, MissingPetUpdate
from app.schemas.common import StandardResponse
from app.services.ai_service import AIManager
from app.services.pet_service import PetService
from app.services.sighting_service import SightingService
from app.repositories.supabase_missing_pet_repository import (
    SupabaseMissingPetRepository,
)
from app.repositories.sighting_repository import OwnerDecisionRefused
from app.repositories.supabase_sighting_repository import (
    SupabaseSightingRepository,
)
from app.services.notification_service import notify_nearby_hunters
from app.core.auth import get_current_user_id
from app.core.config import settings
from app.core.database import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/missing-pets", tags=["Missing Pets"])

# Statuses an owner can PATCH that mean "the search is over". `Resolved` is not
# here because the schema does not let an owner write it: it means "the bounty
# has been paid", which only the administrator's settlement writes, and by then
# the search is already Found and its sightings already closed — either by this
# route (the owner ended the search themselves) or by owner_decide_sighting
# (the owner confirmed a rescue).
CLOSED_PET_STATUSES = frozenset({"Found"})

@router.post("/", response_model=StandardResponse)
async def create_missing_pet(
    pet: MissingPetCreate,
    background_tasks: BackgroundTasks,
    supabase = Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    try:
        # Owner identity comes from the verified JWT, not the request body.
        pet.owner_id = user_id
        data = await PetService.register_missing_pet(
            SupabaseMissingPetRepository(supabase), pet
        )

        # Fan out a geolocation push to nearby hunters AFTER the insert, in a
        # background task so the owner's response is never blocked (SRS-FR-12).
        # Guarded internally by is_firebase_ready() — no-op without creds.
        background_tasks.add_task(
            notify_nearby_hunters,
            supabase,
            data["id"],
            pet.latitude,
            pet.longitude,
            user_id,
            settings.DEFAULT_SEARCH_RADIUS_KM,
            pet.pet_name,
            pet.species,
        )

        return StandardResponse(
            status="success",
            message="Pet registered successfully.",
            data=data
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register pet: {str(e)}")
    

@router.get("/nearby", response_model=StandardResponse)
async def get_nearby_missing_pets(
    latitude: float = Query(..., ge=-90.0, le=90.0, description="Center point latitude"),
    longitude: float = Query(..., ge=-180.0, le=180.0, description="Center point longitude"),
    radius_km: float = Query(10.0, ge=0.1, le=100.0, description="Search radius in kilometers"),
    supabase = Depends(get_supabase_client)
):
    try:
        pets = await PetService.get_nearby_missing_pets(
            SupabaseMissingPetRepository(supabase),
            latitude, longitude, radius_km, limit=20
        )
        return StandardResponse(
            status="success",
            message=f"Found {len(pets)} nearby missing pets.",
            data=pets
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find nearby pets: {str(e)}")


@router.get("/me", response_model=StandardResponse)
async def get_my_missing_pets(
    supabase = Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    """
    MD-38 / SRS-67 — the owner's "My Reports" list, newest first.

    ⚠️  Declared BEFORE `/{pet_id}`: FastAPI matches routes in declaration
    order, so with the parameterised route first, "me" would be captured as a
    pet id and this endpoint would never run.

    Owner identity comes from the JWT, never from a query parameter, so one
    owner cannot list another's reports.
    """
    try:
        pets = await PetService.get_my_missing_pets(
            SupabaseMissingPetRepository(supabase), user_id
        )
        return StandardResponse(
            status="success",
            message=f"Retrieved {len(pets)} reports.",
            data=pets,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch your reports: {str(e)}"
        )


@router.get("/{pet_id}/sightings", response_model=StandardResponse)
async def get_sightings_for_pet(
    pet_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    supabase = Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    """
    Owner-facing list of sightings reported against this missing pet.
    Includes AI-matched sightings AND any sighting whose hunter explicitly
    tagged this pet via `initial_target_pet_id` (per product requirement).
    Newest first.

    Requires authentication (Feature #6); the sighting timeline reveals who
    spotted a pet and where, so it is not exposed anonymously.
    """
    try:
        data = await PetService.get_sightings_for_pet(
            SupabaseMissingPetRepository(supabase),
            pet_id, limit=limit, offset=offset,
        )
        return StandardResponse(
            status="success",
            message=f"Retrieved {len(data)} sightings.",
            data=data,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch sightings for pet: {str(e)}",
        )


class MatchDecisionRequest(BaseModel):
    """Body of PATCH /missing-pets/{pet_id}/sightings/{sighting_id}.

    `decision` is a plain string rather than a Literal so an unrecognised value
    comes back as the documented 400 (raised by
    `sighting_logic.normalize_owner_decision`, which names the permitted set)
    instead of FastAPI's 422.
    """

    decision: str = Field(
        ...,
        description="'Confirmed' (that is my pet) or 'Rejected' (it isn't).",
    )


@router.patch(
    "/{pet_id}/sightings/{sighting_id}", response_model=StandardResponse
)
async def decide_sighting_match(
    pet_id: str,
    sighting_id: str,
    payload: MatchDecisionRequest,
    supabase = Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    """
    The owner's verdict on one sighting matched to their pet.

    The pet id leads the path because that is what the ownership check is
    against: an owner is ruling on an entry in their own pet's timeline. A pet
    the caller does not own answers 404, exactly as the owner-scoped PATCH
    above does, so the endpoint never reveals that someone else's pet exists.

    Confirming a sighting whose `action_type` is 'Caught' does considerably
    more than record a verdict: it ends the search and distributes every clue
    score for the pet (see `SightingService.decide_match`). The response says
    which happened — `search_closed` and `awards` — so the client can react
    without a second round-trip.

    The queue is ordered, so this can also fail as a **409**: the card was
    already decided, an older card is still undecided, or the search is over.
    Those handlers must precede the generic ValueError -> 400, because they
    subclass it.
    """
    service = SightingService(
        repo=SupabaseSightingRepository(supabase), ai_manager=AIManager
    )
    try:
        result = await service.decide_match(
            pet_id, sighting_id, user_id, payload.decision,
        )
    except LookupError as le:
        raise HTTPException(status_code=404, detail=str(le))
    except OwnerDecisionRefused as ode:
        raise HTTPException(status_code=409, detail=str(ode))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to record decision: {e}"
        )
    return StandardResponse(
        status="success",
        message=(
            "Search closed; scores awarded."
            if result.get("search_closed") else "Decision recorded."
        ),
        data=result,
    )


@router.get("/{pet_id}", response_model=StandardResponse)
async def get_missing_pet(
    pet_id: str,
    supabase = Depends(get_supabase_client)
):
    try:
        pet = await PetService.get_missing_pet_by_id(
            SupabaseMissingPetRepository(supabase), pet_id
        )
        if not pet:
            raise HTTPException(status_code=404, detail=f"Missing pet {pet_id} not found")
        return StandardResponse(
            status="success",
            message="Pet retrieved successfully.",
            data=pet
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve pet: {str(e)}")

@router.patch("/{pet_id}", response_model=StandardResponse)
async def update_missing_pet(
    pet_id: str,
    update: MissingPetUpdate,
    supabase = Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    """
    General-purpose PATCH for missing pet fields (pet_name, status, bounty_amount, ฯลฯ).
    รับแค่ field ที่ client ส่งมาจริงๆ (exclude_unset) — field ที่ไม่ส่งจะไม่ถูก overwrite.

    ⚠️  NOTE: endpoint นี้ call Supabase โดยตรง ไม่ผ่าน PetService.update_missing_pet_status()
    ผลคือ valid_statuses guard ใน service ไม่ถูก enforce ที่นี่
    → status validation ต้องทำที่ MissingPetUpdate schema แทน (ดู schemas/missing_pets.py)

    Security: owner-scoped — .eq("owner_id", user_id) ทำให้แก้ได้แค่ pet ของตัวเอง
    pet ที่ไม่มีหรือเป็นของคนอื่นจะ return 0 row → 404 (ไม่บอกว่ามี pet นั้นอยู่ไหม)
    """
    try:
        # Build update payload with only provided fields
        payload = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}

        if not payload:
            raise HTTPException(status_code=400, detail="No fields to update")

        # Owner-scoped: the update only matches a row the caller owns. A pet
        # that doesn't exist OR isn't theirs both yield no rows -> 404 (we don't
        # leak existence of other owners' reports).
        repo = SupabaseMissingPetRepository(supabase)
        updated = repo.update_missing_pet_owned(pet_id, user_id, payload)

        if not updated:
            raise HTTPException(
                status_code=404,
                detail=f"Missing pet {pet_id} not found or not owned by you",
            )

        # Ending the search ends its sightings too: they stop being live leads
        # the moment the pet is home. Isolated in its own try/except — the
        # owner's report IS already updated, so failing the request here would
        # report a failure that did not happen (same reasoning as
        # SightingService._persist_matches).
        if payload.get("status") in CLOSED_PET_STATUSES:
            try:
                closed = repo.close_sightings_for_pet(pet_id)
                logger.info(
                    "Pet %s closed by owner — %d sighting(s) marked Closed.",
                    pet_id, closed,
                )
            except Exception as e:
                logger.error(
                    "Pet %s closed but its sightings could NOT be closed: %s",
                    pet_id, e, exc_info=True,
                )

        return StandardResponse(
            status="success",
            message="Pet updated successfully.",
            data=updated
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update pet: {str(e)}")

