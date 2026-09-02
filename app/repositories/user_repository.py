"""The user aggregate's repository port (profile reads + role assignment).

Covers the `public.users` reads outside the service layer — the caller's own
profile (GET /auth/me) and the server-side admin role check (require_admin) —
and, since 2026-09-02, the administrator's assignment of the role that gate
reads (MD-57 to MD-59).

NOTE: JWT validation via Supabase Auth/GoTrue (`client.auth.get_user`) is a
SEPARATE identity boundary, not a DB read — it is intentionally NOT part of this
repository.

**What the role methods are not.** `find_by_email` resolves ONE account from a
full address and returns None otherwise. It is not `search_accounts`, and there
is deliberately no `list_accounts`, `suspend`, or `deactivate` here: those were
struck on 2026-08-21 and stay struck. Granting console access and sanctioning an
account are different subjects (see `admin_service`'s module docstring).
"""
from typing import Protocol

from app.repositories.pagination import Page


class UserProfileNotFound(Exception):
    """Raised by the adapter when a user's profile row can't be read as exactly
    one row (missing profile, or any read failure the `.single()` terminal
    surfaces). Callers map it to 404. Preserves the route's pre-seam behaviour
    of treating any get-profile failure as 'profile not found'.
    """

    def __init__(self, user_id: str):
        super().__init__(f"No profile row for user {user_id}")
        self.user_id = user_id


class UserAccountNotFound(ValueError):
    """No account matched — the 404 of MD-57 (by address) and MD-58 (by id).

    Distinct from `UserProfileNotFound`, which means the CALLER's own profile
    could not be read. This one is about the account being looked up or acted
    upon, so `identifier` is whichever handle the caller used.
    """

    def __init__(self, identifier: str):
        super().__init__(f"No account matching {identifier}")
        self.identifier = identifier


class RoleAssignmentRefused(ValueError):
    """A guard of SRS-96 refused the change — MD-58's 409.

    Two cases, both raised by the `assign_user_role` procedure and translated
    here: an administrator withdrawing their own access, and a change that would
    leave the platform with no administrator. Neither is a server fault and
    neither is retryable as written, so they are 409 rather than 400 or 500.
    """

    def __init__(self, message: str):
        super().__init__(message)


class UserRepository(Protocol):
    def get_user_profile(self, user_id: str) -> dict: ...
    def get_user_role(self, user_id: str) -> str | None: ...
    def update_last_location(
        self, user_id: str, location_point: str
    ) -> dict | None: ...
    def update_profile(
        self, user_id: str, patch: dict
    ) -> dict | None: ...

    # -- role assignment (MD-57 to MD-59) ------------------------------- #
    def find_by_email(self, email: str) -> dict | None: ...
    def assign_user_role(
        self, target_user_id: str, role: str, changed_by: str
    ) -> dict: ...
    def list_role_changes(
        self, limit: int, offset: int, target_user_id: str | None
    ) -> Page: ...
    def list_admins(self) -> list[dict]: ...
