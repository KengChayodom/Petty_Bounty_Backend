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
                       .neq("verification_status", verification_status)
                       .execute())
        return res.data[0] if res.data else None

    def get_sighting(self, sighting_id: str) -> dict | None:
        res = (self._db.table("sightings")
                       .select("*")
                       .eq("id", sighting_id)
                       .execute())
        return res.data[0] if res.data else None

    def has_penalty_for_sighting(self, sighting_id: str) -> bool:
        res = (self._db.table("score_penalties")
                       .select("id")
                       .eq("sighting_id", sighting_id)
                       .limit(1)
                       .execute())
        return bool(res.data)

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

    def apply_score_penalty(
        self,
        user_id: str,
        sighting_id: str | None,
        report_id: str,
        points: int,
        reason: str | None,
        penalised_by: str,
    ) -> dict:
        # One RPC, not two writes: the audit row and the balance change must
        # land together. The function is idempotent on report_id, so a retried
        # review does not deduct twice — it returns already_applied=True.
        res = self._db.rpc("apply_score_penalty", {
            "p_user_id":      user_id,
            "p_sighting_id":  sighting_id,
            "p_report_id":    report_id,
            "p_points":       points,
            "p_reason":       reason,
            "p_penalised_by": penalised_by,
        }).execute()
        return res.data or {}
