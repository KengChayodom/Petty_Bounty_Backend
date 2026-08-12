"""The user aggregate's repository port (profile reads).

Covers the two `public.users` reads outside the service layer: the caller's own
profile (GET /auth/me) and the server-side admin role check (require_admin).

NOTE: JWT validation via Supabase Auth/GoTrue (`client.auth.get_user`) is a
SEPARATE identity boundary, not a DB read — it is intentionally NOT part of this
repository.
"""
from typing import Protocol


class UserProfileNotFound(Exception):
    """Raised by the adapter when a user's profile row can't be read as exactly
    one row (missing profile, or any read failure the `.single()` terminal
    surfaces). Callers map it to 404. Preserves the route's pre-seam behaviour
    of treating any get-profile failure as 'profile not found'.
    """

    def __init__(self, user_id: str):
        super().__init__(f"No profile row for user {user_id}")
        self.user_id = user_id


class UserRepository(Protocol):
    def get_user_profile(self, user_id: str) -> dict: ...
    def get_user_role(self, user_id: str) -> str | None: ...
    def update_last_location(
        self, user_id: str, location_point: str
    ) -> dict | None: ...
    def update_profile(
        self, user_id: str, patch: dict
    ) -> dict | None: ...
