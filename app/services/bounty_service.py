"""Feature 6 bounty settlement: the owner pays the returning hunter directly by
PromptPay and submits the slip, the hunter confirms or disputes it, and an
Administrator rules on what the two could not settle.

The rules live in the SQL functions (migrations/2026_10_08_bounty_settlement.sql)
because a payment and the report's status must change together. This service
translates their RAISE messages into the exceptions the routes map to HTTP.
"""
import asyncio
import logging

from app.repositories.bounty_repository import (
    BountyConflict,
    BountyNotFound,
    BountyRepository,
)
from app.services.promptpay import is_thai_mobile

logger = logging.getLogger(__name__)

HUNTER_ANSWERS = ("Confirm", "Dispute")
ADMIN_DECISIONS = ("Confirm", "Reject")


def translate_db_error(exc: Exception) -> Exception:
    """Map a SQL function's RAISE message to the exception the routes expect.

    Returns the exception to raise. Anything unrecognised is returned as is,
    so the route reports it as a 500 rather than hiding a real fault.
    """
    msg = str(exc)
    low = msg.lower()
    if "not found" in low:
        return BountyNotFound(msg)
    if ("already" in low or "is not awaiting" in low or "is not unpaid" in low
            or "window" in low):
        return BountyConflict(msg)
    if "without a slip" in low or "must be" in low:
        return ValueError(msg)
    return exc


class BountyService:
    def __init__(self, repo: BountyRepository):
        self.repo = repo

    async def _call(self, fn, *args):
        try:
            return await asyncio.to_thread(fn, *args)
        except Exception as e:  # noqa: BLE001 - re-raised, translated
            raise translate_db_error(e) from e

    # ------------------------------------------------------------------ #
    # The owner's and the returning hunter's view of one report's payment.
    # ------------------------------------------------------------------ #
    async def get_view(self, pet_id: str, user_id: str) -> dict:
        pet = await asyncio.to_thread(self.repo.get_pet, pet_id)
        if not pet:
            raise BountyNotFound(f"Missing pet {pet_id} not found")
        final = await asyncio.to_thread(self.repo.get_final_sighting, pet_id)
        hunter_id = (final or {}).get("hunter_id")

        if str(user_id) == str(pet["owner_id"]):
            role = "owner"
        elif hunter_id and str(user_id) == str(hunter_id):
            role = "hunter"
        else:
            raise BountyNotFound(f"Missing pet {pet_id} not found")

        payment = await asyncio.to_thread(self.repo.get_latest_payment, pet_id)
        live = payment if payment and payment.get("status") in (
            "Pending_Verification", "Verified") else None
        # A returning hunter is required: a report closed with nobody to pay
        # owes nothing, whatever its status says.
        unpaid = (pet.get("status") == "Found"
                  and float(pet.get("bounty_amount") or 0) > 0
                  and live is None
                  and hunter_id is not None)

        view = {
            "pet_id": pet_id,
            "role": role,
            "pet_status": pet.get("status"),
            "bounty_amount": pet.get("bounty_amount"),
            "unpaid": unpaid,
            "found_at": pet.get("found_at"),
            "unpaid_referred_at": pet.get("unpaid_referred_at"),
            "bounty_unpaid": pet.get("bounty_unpaid"),
            "hunter_id": hunter_id,
            "hunter_username": None,
            "hunter_promptpay": None,
            "payment": payment,
        }
        if hunter_id:
            view["hunter_username"] = await asyncio.to_thread(
                self.repo.get_username, hunter_id)
        # The hunter's number is shown only to an owner who owes them, and only
        # when it is a Thai mobile number that PromptPay can take.
        if role == "owner" and unpaid and hunter_id:
            phone = await asyncio.to_thread(self.repo.get_user_phone, hunter_id)
            view["hunter_promptpay"] = phone if is_thai_mobile(phone) else None
        return view

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    async def submit_payment(self, pet_id: str, owner_id: str, slip_url: str,
                             reference_no: str | None) -> dict:
        return await self._call(self.repo.submit_payment, pet_id, owner_id,
                                slip_url, reference_no)

    async def hunter_answer(self, payment_id: str, hunter_id: str, answer: str) -> dict:
        normalized = (answer or "").strip().capitalize()
        if normalized not in HUNTER_ANSWERS:
            raise ValueError(f"answer must be one of {', '.join(HUNTER_ANSWERS)}")
        return await self._call(self.repo.hunter_answer, payment_id, hunter_id,
                                normalized)

    async def refer_unpaid(self, pet_id: str, hunter_id: str) -> dict:
        return await self._call(self.repo.refer_unpaid, pet_id, hunter_id)

    async def admin_rule(self, payment_id: str, admin_id: str, decision: str) -> dict:
        normalized = (decision or "").strip().capitalize()
        if normalized not in ADMIN_DECISIONS:
            raise ValueError(f"decision must be one of {', '.join(ADMIN_DECISIONS)}")
        return await self._call(self.repo.admin_rule, payment_id, admin_id,
                                normalized)

    async def admin_close_unpaid(self, pet_id: str, admin_id: str) -> dict:
        return await self._call(self.repo.admin_close_unpaid, pet_id, admin_id)

    async def ruling_queue(self) -> list[dict]:
        return await self._call(self.repo.ruling_queue)

    async def bounties_for_hunter(self, hunter_id: str) -> list[dict]:
        return await self._call(self.repo.bounties_for_hunter, hunter_id)
