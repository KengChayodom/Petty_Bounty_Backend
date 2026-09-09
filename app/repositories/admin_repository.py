"""The admin aggregate's repository port (verification / timeline / resolution).

One method per real DB op in admin_service.py. The resolve RPC's vendor error
is intentionally NOT translated here — the service inspects the message to
decide 400 vs 500, so the exception is left to propagate to it.
"""
from typing import Protocol


class SightingNotFound(ValueError):
    """Raised when a sighting identifier matches no row.

    Subclasses ValueError so callers written before it existed, and the unit
    cases that assert `ValueError`, keep working. The route catches this first
    to answer 404, and the plain ValueError behind it to answer 400 — the two
    were indistinguishable until 2026-09-09 and a mistyped verification state
    was reported as a sighting that does not exist.
    """

    def __init__(self, sighting_id: str):
        self.sighting_id = sighting_id
        super().__init__(f"Sighting {sighting_id} not found")


class AdminRepository(Protocol):
    def update_sighting_verification(
        self, sighting_id: str, verification_status: str
    ) -> dict | None: ...
    def get_sighting_timeline(
        self, pet_id: str, limit: int, offset: int
    ) -> list[dict]: ...
    def resolve_missing_pet(
        self,
        pet_id: str,
        final_sighting_id: str,
        slip_image_url: str,
        reference_no: str | None,
        verified_by: str,
    ) -> object: ...
    def apply_score_penalty(
        self,
        user_id: str,
        sighting_id: str | None,
        report_id: str,
        points: int,
        reason: str | None,
        penalised_by: str,
    ) -> dict: ...
