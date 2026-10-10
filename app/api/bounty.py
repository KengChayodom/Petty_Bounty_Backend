"""Feature 6 — the owner pays the returning hunter, and the hunter answers.

Routes for the two parties of a closed report. The Administrator's routes are
in app/api/admin.py.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_bounty_service
from app.core.auth import get_current_user_id
from app.repositories.bounty_repository import BountyConflict, BountyNotFound
from app.schemas.common import StandardResponse
from app.services.bounty_service import BountyService

# Every route here needs a signed-in caller, declared ONCE for the whole router so a route added later inherits it.
# The routes still ask for the id where they use it, which FastAPI resolves once per request.
router = APIRouter(
    prefix="/bounty", tags=["Bounty"],
    dependencies=[Depends(get_current_user_id)],
)


class PaymentSubmission(BaseModel):
    slip_image_url: str = Field(..., min_length=1, description="URL of the transfer slip image")
    reference_no: str | None = Field(None, max_length=255, description="Transfer reference number")


class HunterAnswer(BaseModel):
    answer: str = Field(..., description="'Confirm' (the money arrived) or 'Dispute' (it did not)")


async def _run(coro):
    """Map the service's exceptions to HTTP, the same way for every route."""
    try:
        return await coro
    except BountyNotFound as nf:
        raise HTTPException(status_code=404, detail=str(nf))
    except BountyConflict as bc:
        raise HTTPException(status_code=409, detail=str(bc))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bounty request failed: {e}")


@router.get("/mine", response_model=StandardResponse)
async def my_bounties(
    service: BountyService = Depends(get_bounty_service),
    user_id: str = Depends(get_current_user_id),
):
    """The reports the caller returned the pet for, with each one's payment
    state, newest first. Declared before /{pet_id} so "mine" is not read as a
    pet id. It is the hunter's way into the confirm, dispute and unpaid routes."""
    rows = await _run(service.bounties_for_hunter(user_id))
    return StandardResponse(status="success", message=f"{len(rows)} bounty case(s).", data=rows)


@router.get("/{pet_id}", response_model=StandardResponse)
async def get_bounty(
    pet_id: str,
    service: BountyService = Depends(get_bounty_service),
    user_id: str = Depends(get_current_user_id),
):
    """The payment state of one closed report, for its owner or its returning
    hunter; anyone else gets 404.

    SRS-104, SRS-105, SRS-107: for an owner who still owes the bounty, the
    response carries the hunter's profile phone as their PromptPay number when
    it is a Thai mobile number, else null, so the app can show the QR code or
    say the hunter's payout details are awaited.
    """
    data = await _run(service.get_view(pet_id, user_id))
    return StandardResponse(status="success", message="Bounty retrieved.", data=data)


@router.post("/{pet_id}/payment", response_model=StandardResponse)
async def submit_payment(
    pet_id: str,
    payload: PaymentSubmission,
    service: BountyService = Depends(get_bounty_service),
    user_id: str = Depends(get_current_user_id),
):
    """SRS-106: the owner submits a payment for an unpaid report, the slip
    image and an optional reference number, recorded as awaiting confirmation.
    404 when the report is not the caller's; 409 when it is not awaiting a
    payment or already has one awaiting confirmation or confirmed."""
    data = await _run(service.submit_payment(
        pet_id, user_id, payload.slip_image_url, payload.reference_no))
    return StandardResponse(status="success", message="Payment submitted.", data=data)


@router.post("/payments/{payment_id}/answer", response_model=StandardResponse)
async def answer_payment(
    payment_id: str,
    payload: HunterAnswer,
    service: BountyService = Depends(get_bounty_service),
    user_id: str = Depends(get_current_user_id),
):
    """SRS-108: the returning hunter, and no one else, confirms a payment at
    any time or disputes it within its three-day dispute window. Confirming
    resolves the report; a dispute sends it to the Administrator's queue."""
    data = await _run(service.hunter_answer(payment_id, user_id, payload.answer))
    return StandardResponse(status="success", message="Answer recorded.", data=data)


@router.post("/{pet_id}/unpaid", response_model=StandardResponse)
async def refer_unpaid(
    pet_id: str,
    service: BountyService = Depends(get_bounty_service),
    user_id: str = Depends(get_current_user_id),
):
    """SRS-109: the returning hunter refers an unpaid report to the
    Administrator, within three days of it becoming "Found"."""
    data = await _run(service.refer_unpaid(pet_id, user_id))
    return StandardResponse(status="success", message="Referred as unpaid.", data=data)
