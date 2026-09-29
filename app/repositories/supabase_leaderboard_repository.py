"""Supabase adapter for LeaderboardRepository.

The chains are the ones that used to sit in the route, moved verbatim, plus the
`.data` / `.count` unwrap.
"""

# A search that has ended. Kept as a list because PostgREST's `not in` takes one.
_CLOSED_PET_STATUSES = ["Found", "Resolved"]


class SupabaseLeaderboardRepository:
    def __init__(self, db):
        self._db = db

    def list_users_page(self, limit: int, offset: int) -> list[dict]:
        res = (
            self._db.table("users")
            .select("id, username, profile_image_url, total_score")
            .order("total_score", desc=True)
            .order("id")
            .range(offset, offset + limit - 1)
            .execute()
        )
        return res.data or []

    def count_users_above(self, score: float) -> int:
        res = (
            self._db.table("users")
            .select("id", count="exact")
            .gt("total_score", score)
            .execute()
        )
        return res.count or 0

    def get_user_standing(self, user_id: str) -> dict | None:
        res = (
            self._db.table("users")
            .select("username, profile_image_url, total_score")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        return res.data[0] if res.data else None

    def list_open_bounties_page(self, limit: int, offset: int) -> list[dict]:
        res = (
            self._db.table("missing_pets")
            .select("id, pet_name, image_url, bounty_amount")
            .not_.in_("status", _CLOSED_PET_STATUSES)
            .order("bounty_amount", desc=True)
            .order("id")
            .range(offset, offset + limit - 1)
            .execute()
        )
        return res.data or []

    def count_open_bounties_above(self, amount: float) -> int:
        res = (
            self._db.table("missing_pets")
            .select("id", count="exact")
            .not_.in_("status", _CLOSED_PET_STATUSES)
            .gt("bounty_amount", amount)
            .execute()
        )
        return res.count or 0
