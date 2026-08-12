"""Supabase adapter for UserRepository — public.users reads + location write."""
from datetime import datetime, timezone

from app.repositories.user_repository import UserProfileNotFound


class SupabaseUserRepository:
    def __init__(self, db):
        self._db = db

    def get_user_profile(self, user_id: str) -> dict:
        try:
            res = (
                self._db.table("users")
                .select("id, display_name, phone, role, total_score, profile_image_url, created_at")
                .eq("id", user_id)
                .single()
                .execute()
            )
        except Exception as exc:
            # `.single()` raises when there isn't exactly one row (e.g. the
            # profile trigger missed). Any failure here => "profile not found",
            # matching the route's pre-seam broad-except behaviour.
            raise UserProfileNotFound(user_id) from exc
        return res.data

    def get_user_role(self, user_id: str) -> str | None:
        # maybe_single() returns data=None for a missing row instead of raising,
        # so "no profile" yields role None (caller 403s); a genuine transport
        # failure still raises and the caller surfaces 500.
        res = (
            self._db.table("users")
            .select("role")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )
        return (getattr(res, "data", None) or {}).get("role")

    def update_last_location(
        self, user_id: str, location_point: str
    ) -> dict | None:
        res = (
            self._db.table("users")
            .update({
                "last_location": location_point,
                "last_location_at": datetime.now(timezone.utc).isoformat(),
            })
            .eq("id", user_id)
            .execute()
        )
        return res.data[0] if res.data else None
