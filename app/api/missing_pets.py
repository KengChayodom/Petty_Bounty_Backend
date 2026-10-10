# app/api/missing_pets.py
import logging

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field

from app.api.dependencies import (
    get_notification_repository,
    get_pet_service,
    get_sighting_service,
)
from app.core.auth import get_current_user_id
from app.core.config import settings
from app.repositories.notification_repository import NotificationRepository
from app.repositories.sighting_repository import OwnerDecisionRefused
from app.schemas.common import StandardResponse
from app.schemas.missing_pets import MissingPetCreate, MissingPetUpdate
from app.services.notification_service import notify_nearby_hunters
from app.services.pet_logic import OwnerEditRefused
from app.services.pet_service import PetService
from app.services.sighting_service import SightingService

logger = logging.getLogger(__name__)

# Every route here needs a signed-in caller, declared ONCE for the whole router so a route added later inherits it.
# The routes still ask for the id where they use it, which FastAPI resolves once per request.
router = APIRouter(
    prefix="/missing-pets", tags=["Missing Pets"],
    dependencies=[Depends(get_current_user_id)],
)

# The map's route. Open on purpose: it returns no contact details (only the pet
# and its `owner_id`), and the map loads before anything else. It lives on its
# own router so that `router` above can require a login for everything else.
# main.py includes this one FIRST, so `/nearby` is not read as a `{pet_id}`.
public_router = APIRouter(prefix="/missing-pets", tags=["Missing Pets"])


@router.post("/", response_model=StandardResponse)
async def create_missing_pet(
    pet: MissingPetCreate,
    background_tasks: BackgroundTasks,
    service: PetService = Depends(get_pet_service),
    notif_repo: NotificationRepository = Depends(get_notification_repository),
    user_id: str = Depends(get_current_user_id),
):
    try:
        # Owner identity comes from the verified JWT, not the request body.
        pet.owner_id = user_id
        data = await service.register_missing_pet(pet)

        # Fan out a geolocation push to nearby hunters AFTER the insert, in a
        # background task so the owner's response is never blocked (SRS-FR-12).
        # Guarded internally by is_firebase_ready() — no-op without creds.
        background_tasks.add_task(
            notify_nearby_hunters,
            notif_repo,
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
    

@public_router.get("/nearby", response_model=StandardResponse)
async def get_nearby_missing_pets(
    latitude: float = Query(..., ge=-90.0, le=90.0, description="Center point latitude"),
    longitude: float = Query(..., ge=-180.0, le=180.0, description="Center point longitude"),
    radius_km: float = Query(10.0, ge=0.1, le=100.0, description="Search radius in kilometers"),
    service: PetService = Depends(get_pet_service),
):
    try:
        pets = await service.get_nearby_missing_pets(
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
    service: PetService = Depends(get_pet_service),
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
        pets = await service.get_my_missing_pets(user_id)
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
    service: PetService = Depends(get_pet_service),
    user_id: str = Depends(get_current_user_id),
):
    """
    Owner-facing list of sightings reported against this missing pet.
    Includes AI-matched sightings AND any sighting whose hunter explicitly
    tagged this pet via `initial_target_pet_id` (per product requirement).
    Newest first.

    Scoped to the caller's OWN report (2026-09-09). Being signed in was the only
    check here until then, so any authenticated account could read any owner's
    timeline — and these rows carry where a pet was seen plus the hunter's name
    and telephone number, which is a third party's contact details. A report the
    caller does not own answers 404 rather than 403, exactly as the owner-scoped
    PATCH does, so the endpoint never reveals that someone else's report exists.
    """
    try:
        data = await service.get_sightings_for_pet(
            pet_id, limit=limit, offset=offset, owner_id=user_id,
        )
        return StandardResponse(
            status="success",
            message=f"Retrieved {len(data)} sightings.",
            data=data,
        )
    except LookupError as le:
        # No such report, or it is not the caller's. Deliberately the same
        # answer for both. Must precede the generic handler below.
        raise HTTPException(status_code=404, detail=str(le))
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
    service: SightingService = Depends(get_sighting_service),
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
    service: PetService = Depends(get_pet_service),
    user_id: str = Depends(get_current_user_id),
):
    """One missing-pet report, for a signed-in caller.

    The detail RPC joins in the owner's username, telephone number and photo, so
    this needs a token: without one, anyone could list pet ids from `/nearby`
    and read a telephone number for each without an account. The app already
    sends every signed-out user to the login screen, so a request without a
    token is not one it makes.
    """
    try:
        pet = await service.get_missing_pet_by_id(pet_id)
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
    service: PetService = Depends(get_pet_service),
    user_id: str = Depends(get_current_user_id),
):
    """
    Partial update of the caller's own report (name, status, bounty,
    characteristics, colour). Only the fields the client actually sent are
    written (`exclude_unset`), so an unsent field is never overwritten.

    Owner-scoped: a report that does not exist and one owned by somebody else
    both answer 404, so the endpoint never reveals that a report exists.

    Feature 6 (SRS-100, SRS-102, SRS-103): the bounty may be raised but never
    lowered, and neither it nor the status may change once the report is
    "Found". Closing by hand is refused while a "Caught" sighting awaits the
    owner's decision. Otherwise it closes straight to "Resolved", there being
    nobody to pay. A refused edit answers 409. The rules themselves live in
    `PetService.update_owned_missing_pet` and `pet_logic.apply_owner_edit_rules`.
    """
    payload = {
        k: v for k, v in update.model_dump(exclude_unset=True).items()
        if v is not None
    }
    if not payload:
        raise HTTPException(status_code=400, detail="No fields to update")

    try:
        updated = await service.update_owned_missing_pet(pet_id, user_id, payload)
    except LookupError as le:
        raise HTTPException(status_code=404, detail=str(le))
    except OwnerEditRefused as oe:
        raise HTTPException(status_code=409, detail=str(oe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update pet: {str(e)}")

    return StandardResponse(
        status="success",
        message="Pet updated successfully.",
        data=updated,
    )
