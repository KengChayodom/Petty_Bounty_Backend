"""Supabase adapter for SightingRepository.

The ONLY place the sighting flow touches supabase-py. Each method is the
`.table(...)/.rpc(...).execute()` chain moved verbatim out of the service, plus
the `.data`/`.count` unwrap and the one vendor->domain error translation the
port contract requires (SightingNotSaved on an empty INSERT). No logic beyond
that lives here — filtering/mapping/assembly are pure functions in the service
layer.
"""
from app.repositories.sighting_repository import SightingNotSaved


class SupabaseSightingRepository:
    def __init__(self, db):
        self._db = db

    # --- writes ---------------------------------------------------------- #
    def insert_sighting(self, payload: dict) -> dict:
        res = self._db.table("sightings").insert(payload).execute()
        if not res.data:
            raise SightingNotSaved(payload)
        return res.data[0]

    def upsert_sighting_matches(self, rows: list[dict]) -> None:
        (self._db.table("sighting_matches")
                 .upsert(rows, on_conflict="sighting_id,missing_pet_id")
                 .execute())

    def set_sighting_status(self, sighting_id: str, status: str) -> dict | None:
        res = (self._db.table("sightings")
                       .update({"sighting_status": status})
                       .eq("id", sighting_id)
                       .execute())
        return res.data[0] if res.data else None

    def update_match_owner_status(
        self, sighting_id: str, pet_id: str, status: str
    ) -> dict | None:
        # Scoped to the ONE (sighting, pet) pair: a sighting can match several
        # pets, and each owner decides only about their own.
        res = (self._db.table("sighting_matches")
                       .update({"owner_status": status})
                       .eq("sighting_id", sighting_id)
                       .eq("missing_pet_id", pet_id)
                       .execute())
        return res.data[0] if res.data else None

    # --- owner side of the loop ------------------------------------------ #
    def get_pet_owners(self, pet_ids: list[str]) -> dict[str, str]:
        """pet id -> owner id, for the pets a sighting was matched to.

        A separate read because the `match_missing_pets` RPC does not return
        `owner_id` — reading it here keeps the push fan-out working without a
        migration to change that function's return type.
        """
        if not pet_ids:
            return {}
        res = (self._db.table("missing_pets")
                       .select("id, owner_id")
                       .in_("id", pet_ids)
                       .execute())
        return {
            row["id"]: row["owner_id"]
            for row in (res.data or [])
            if row.get("id") and row.get("owner_id")
        }

    # --- discovery / match reads ---------------------------------------- #
    def get_sighting_for_match(self, sighting_id: str) -> dict | None:
        res = (self._db.table("sightings")
                       .select("id, feature_vector, detected_species, "
                               "sighted_location, primary_color_hex")
                       .eq("id", sighting_id)
                       .execute())
        return res.data[0] if res.data else None

    def match_missing_pets(self, sighting_id: str, limit: int) -> list[dict]:
        res = self._db.rpc("match_missing_pets", {
            "p_sighting_id": sighting_id,
            "match_limit": limit,
        }).execute()
        return res.data or []

    def get_sighting(self, sighting_id: str) -> dict | None:
        res = (self._db.table("sightings")
                       .select("*")
                       .eq("id", sighting_id)
                       .execute())
        return res.data[0] if res.data else None

    # --- hunter activity / stats reads ---------------------------------- #
    def count_sightings_for_hunter(self, hunter_id: str) -> int:
        res = (self._db.table("sightings")
                       .select("id", count="exact")
                       .eq("hunter_id", hunter_id)
                       .execute())
        return res.count or 0

    def list_sightings_for_hunter(
        self, hunter_id: str, limit: int, offset: int
    ) -> list[dict]:
        res = (self._db.table("sightings")
                       .select("id, image_url, detected_species, "
                               "action_type, sighting_status, "
                               "verification_status, sighted_location, "
                               "initial_target_pet_id, created_at")
                       .eq("hunter_id", hunter_id)
                       .order("created_at", desc=True)
                       .range(offset, offset + limit - 1)
                       .execute())
        return res.data or []

    def get_matches_for_sightings(self, sighting_ids: list[str]) -> list[dict]:
        res = (self._db.table("sighting_matches")
                       .select("sighting_id, missing_pet_id, "
                               "similarity_score, owner_status")
                       .in_("sighting_id", sighting_ids)
                       .execute())
        return res.data or []

    def get_awards_for_hunter(self, hunter_id: str) -> list[dict]:
        res = (self._db.table("score_awards")
                       .select("sighting_id, missing_pet_id, "
                               "points, rank, awarded_at")
                       .eq("user_id", hunter_id)
                       .execute())
        return res.data or []

    def get_user(self, hunter_id: str) -> dict | None:
        res = (self._db.table("users")
                       .select("total_score")
                       .eq("id", hunter_id)
                       .execute())
        return res.data[0] if res.data else None

    def count_verified_sightings_for_hunter(self, hunter_id: str) -> int:
        res = (self._db.table("sightings")
                       .select("id", count="exact")
                       .eq("hunter_id", hunter_id)
                       .eq("verification_status", "Verified")
                       .execute())
        return res.count or 0

    def count_contributions_for_hunter(self, hunter_id: str) -> int:
        res = (self._db.table("score_awards")
                       .select("missing_pet_id", count="exact")
                       .eq("user_id", hunter_id)
                       .execute())
        return res.count or 0
