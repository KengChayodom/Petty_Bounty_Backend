"""The admin aggregate's repository port (verification / timeline / resolution).

One method per real DB op in admin_service.py. The resolve RPC's vendor error
is intentionally NOT translated here — the service inspects the message to
decide 400 vs 500, so the exception is left to propagate to it.
"""
from typing import Protocol


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
