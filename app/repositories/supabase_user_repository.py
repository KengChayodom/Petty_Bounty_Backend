"""Supabase adapter for UserRepository — public.users reads, location write,
and the administrator role assignment of MD-58 to MD-60."""
from datetime import datetime, timezone

from app.repositories.pagination import Page
from app.repositories.user_repository import (
    RoleAssignmentRefused,
    UserAccountNotFound,
    UserProfileNotFound,
)


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

    def update_profile(self, user_id: str, patch: dict) -> dict | None:
        # Self-scoped: the update matches only the caller's own row. A missing
        # profile row yields no rows updated (caller -> 404). On success we
        # re-read through the same projection as get_user_profile so the
        # response body matches the GET /auth/me contract exactly (no leaking
        # of last_location / last_location_at that a bare update would return).
        res = (
            self._db.table("users")
            .update(patch)
            .eq("id", user_id)
            .execute()
        )
        if not res.data:
            return None
        return self.get_user_profile(user_id)

    # ------------------------------------------------------------------ #
    # Role assignment (MD-58 to MD-60)
    # ------------------------------------------------------------------ #
    def find_by_email(self, email: str) -> dict | None:
        # An RPC and not a table read: the address lives in `auth.users`, which
        # PostgREST does not expose and the API key has no grant on. The
        # SECURITY DEFINER function is the narrow window onto it — one exact
        # address in, one row or none out, no way to enumerate.
        #
        # The alternative, walking `auth.admin.list_users()` and filtering in
        # Python, would pull every account's address into this process on every
        # lookup, which is the account list this feature is specified not to be.
        res = self._db.rpc("find_user_by_email", {"p_email": email}).execute()
        return getattr(res, "data", None) or None

    def assign_user_role(
        self, target_user_id: str, role: str, changed_by: str
    ) -> dict:
        # One RPC, not two writes: the guards, the role change and the audit row
        # must land together. Assigning a role the account already holds comes
        # back as changed=False rather than an error, so a retry is harmless.
        try:
            res = self._db.rpc("assign_user_role", {
                "p_target_user_id": target_user_id,
                "p_role":           role,
                "p_changed_by":     changed_by,
            }).execute()
        except Exception as exc:
            # The procedure RAISEs with these texts. Translating them here
            # keeps the service free of database error strings, the same way
            # SupabaseReportRepository raises ReportNotFound.
            msg = str(exc)
            if "not found" in msg:
                raise UserAccountNotFound(target_user_id) from exc
            if ("own administrator access" in msg
                    or "last administrator" in msg):
                raise RoleAssignmentRefused(_guard_message(msg)) from exc
            raise
        return getattr(res, "data", None) or {}

    def list_role_changes(
        self, limit: int, offset: int, target_user_id: str | None
    ) -> Page:
        # Newest first, and count="exact" so the console can draw numbered
        # pages — the same shape as the moderation queue of MD-52.
        query = self._db.table("role_changes").select("*", count="exact")
        if target_user_id is not None:
            query = query.eq("target_user_id", target_user_id)
        res = (query.order("created_at", desc=True)
                    .range(offset, offset + limit - 1)
                    .execute())
        rows = res.data or []
        total = getattr(res, "count", None)
        return Page(rows, len(rows) if total is None else total)

    def list_admins(self) -> list[dict]:
        # The admin set is always small (a handful of people), so no
        # pagination is needed and count=exact is not requested.
        res = (
            self._db.table("users")
            .select("id, display_name, role")
            .eq("role", "admin")
            .order("display_name")
            .execute()
        )
        return res.data or []


def _guard_message(raw: str) -> str:
    """The refusal the procedure raised, without the driver's framing.

    The message reaches the administrator as the 409 body, so it should read as
    the reason the change was refused and not as a database error.
    """
    if "own administrator access" in raw:
        return (
            "You cannot withdraw your own administrator access. "
            "Ask another administrator to do it."
        )
    return (
        "This is the only administrator. "
        "Grant the role to another account first."
    )
