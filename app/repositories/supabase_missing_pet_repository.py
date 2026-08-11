"""Supabase adapter for MissingPetRepository.

The ONLY place the missing-pet flow touches supabase-py. Each method is the
`.table(...)/.rpc(...).execute()` chain moved verbatim out of PetService, plus
the `.data` unwrap and the one INSERT error-translation (MissingPetNotSaved).
"""
from app.repositories.missing_pet_repository import MissingPetNotSaved


class SupabaseMissingPetRepository:
    def __init__(self, db):
        self._db = db

    def insert_missing_pet(self, payload: dict) -> dict:
        res = self._db.table("missing_pets").insert(payload).execute()
        if not res.data:
            raise MissingPetNotSaved(payload)
        return res.data[0]

    def get_missing_pet_by_id(self, pet_id: str) -> dict | None:
        res = self._db.rpc(
            "get_missing_pet_by_id", {"p_pet_id": pet_id}
        ).execute()
        return res.data[0] if res.data else None

    def update_missing_pet_owned(
        self, pet_id: str, owner_id: str, patch: dict
    ) -> dict | None:
        # Owner-scoped: matches a row only if it belongs to owner_id. A pet that
        # doesn't exist OR isn't theirs both yield no rows (caller -> 404).
        res = (self._db.table("missing_pets")
                       .update(patch)
                       .eq("id", pet_id)
                       .eq("owner_id", owner_id)
                       .execute())
        return res.data[0] if res.data else None

    def sightings_for_pet(
        self, pet_id: str, limit: int, offset: int, include_dismissed: bool
    ) -> list[dict]:
        res = self._db.rpc("sightings_for_pet", {
            "p_pet_id":            pet_id,
            "p_limit":             limit,
            "p_offset":            offset,
            "p_include_dismissed": include_dismissed,
        }).execute()
        return res.data or []

    def get_nearby_missing_pets(
        self, center_wkt: str, radius_meters: float, limit: int
    ) -> list[dict]:
        res = self._db.rpc('get_nearby_missing_pets', {
            'center_location': center_wkt,
            'radius_meters': radius_meters,
            'limit': limit,
        }).execute()
        return res.data if res.data else []
