"""The missing-pet aggregate's repository port.

Owned by this codebase; PetService depends on it, tests double it. One method
per real DB operation, derived from the 5 call sites in pet_service.py. No
vendor type crosses the boundary; methods are sync (supabase-py blocks).
"""
from typing import Protocol


class MissingPetNotSaved(ValueError):
    """Raised by the adapter when the missing_pets INSERT returns no row.

    Subclasses ValueError so the service's existing `except ValueError: raise`
    contract and the tests asserting ValueError stay intact.
    """

    def __init__(self, payload: dict):
        super().__init__("Database insertion failed - no data returned")
        self.payload = payload


class MissingPetRepository(Protocol):
    def insert_missing_pet(self, payload: dict) -> dict: ...
    def get_missing_pet_by_id(self, pet_id: str) -> dict | None: ...
    def update_missing_pet_owned(
        self, pet_id: str, owner_id: str, patch: dict
    ) -> dict | None: ...
    def sightings_for_pet(
        self, pet_id: str, limit: int, offset: int, include_dismissed: bool
    ) -> list[dict]: ...
    def get_nearby_missing_pets(
        self, center_wkt: str, radius_meters: float, limit: int
    ) -> list[dict]: ...
