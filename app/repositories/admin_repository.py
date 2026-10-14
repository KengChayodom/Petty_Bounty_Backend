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
    ) -> dict | None:
        """Set the status, returning the row ONLY if this call changed it.

        None means nothing was written, and that covers two different things:
        the sighting does not exist, or it already held that status. The write
        is guarded in the database so two administrators acting at once cannot
        both count as the one who changed it. Callers that must tell the two
        apart follow up with `get_sighting`.
        """
        ...

    def get_sighting(self, sighting_id: str) -> dict | None:
        """The sighting row, or None when there is no such sighting."""
        ...

    def has_penalty_for_sighting(self, sighting_id: str) -> bool:
        """Whether a score deduction has already been recorded against this
        sighting, by any flag."""
        ...

    def get_sighting_timeline(
        self, pet_id: str, limit: int, offset: int
    ) -> list[dict]: ...
    def apply_score_penalty(
        self,
        user_id: str,
        sighting_id: str | None,
        report_id: str,
        points: int,
        reason: str | None,
        penalised_by: str,
    ) -> dict: ...
