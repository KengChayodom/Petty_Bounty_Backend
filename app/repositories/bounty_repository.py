"""Feature 6 bounty settlement: the repository port.

One method per database operation. The SQL functions of
migrations/2026_10_08_bounty_settlement.sql do the checking; their RAISE
messages are left to propagate so BountyService can map them to HTTP statuses.
"""
from typing import Protocol


class BountyNotFound(LookupError):
    """No such report or payment, or not one the caller may act on. The two
    read the same so the routes do not reveal which reports exist."""


class BountyConflict(ValueError):
    """The request is well formed but the report or payment is in a state that
    does not allow it: already paid, already decided, window closed."""


class BountyRepository(Protocol):
    def get_pet(self, pet_id: str) -> dict | None:
        """id, owner_id, pet_name, status, bounty_amount, found_at,
        unpaid_referred_at and bounty_unpaid of a report, or None."""
        ...

    def get_final_sighting(self, pet_id: str) -> dict | None:
        """{sighting_id, hunter_id} of the confirmed Caught sighting that closed
        the report, or None."""
        ...

    def get_latest_payment(self, pet_id: str) -> dict | None: ...
    def get_user_phone(self, user_id: str) -> str | None: ...
    def get_username(self, user_id: str) -> str | None: ...

    def submit_payment(
        self, pet_id: str, owner_id: str, slip_url: str, reference_no: str | None
    ) -> dict: ...
    def hunter_answer(self, payment_id: str, hunter_id: str, answer: str) -> dict: ...
    def refer_unpaid(self, pet_id: str, hunter_id: str) -> dict: ...
    def admin_rule(self, payment_id: str, admin_id: str, decision: str) -> dict: ...
    def admin_close_unpaid(self, pet_id: str, admin_id: str) -> dict: ...
    def ruling_queue(self) -> list[dict]: ...
    def bounties_for_hunter(self, hunter_id: str) -> list[dict]: ...
