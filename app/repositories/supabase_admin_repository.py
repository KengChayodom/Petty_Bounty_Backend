"""Supabase adapter for AdminRepository."""


class SupabaseAdminRepository:
    def __init__(self, db):
        self._db = db

    def update_sighting_verification(
        self, sighting_id: str, verification_status: str
    ) -> dict | None:
        res = (self._db.table("sightings")
                       .update({"verification_status": verification_status})
                       .eq("id", sighting_id)
                       .execute())
        return res.data[0] if res.data else None

    def get_sighting_timeline(
        self, pet_id: str, limit: int, offset: int
    ) -> list[dict]:
        res = self._db.rpc("sightings_for_pet", {
            "p_pet_id":            pet_id,
            "p_limit":             limit,
            "p_offset":            offset,
            "p_include_dismissed": True,  # admins must see Dismissed entries
        }).execute()
        return res.data or []

    def resolve_missing_pet(
        self,
        pet_id: str,
        final_sighting_id: str,
        slip_image_url: str,
        reference_no: str | None,
        verified_by: str,
    ) -> object:
        res = self._db.rpc("resolve_missing_pet", {
            "p_pet_id":            pet_id,
            "p_final_sighting_id": final_sighting_id,
            "p_slip_image_url":    slip_image_url,
            "p_reference_no":      reference_no,
            "p_verified_by":       verified_by,
        }).execute()
        return res.data
