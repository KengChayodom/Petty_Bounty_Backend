"""
The signed-in caller's own account: read the profile, edit it, and record the
device's location.

Every method is self-scoped. The user id comes from the verified JWT at the
route, never from a request body, and each repository call matches on it.
DB access goes through the UserRepository port, run off the event loop because
supabase-py is blocking (see the threading note in sighting_service.py).
"""
import asyncio

from app.repositories.user_repository import UserRepository
from app.utils.postgis import create_postgis_point

# Object-Storage photo formats accepted for the profile picture (MD-46).
ALLOWED_PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png")


def build_profile_patch(
    username: str | None, photo_url: str | None, phone: str | None,
) -> dict:
    """Turn a profile edit into the columns to write.

    A field left as None is not touched. `phone` carries no format rule on
    purpose (see ProfileUpdateRequest in app/api/me.py).

    Raises:
        ValueError: a blank username, an unsupported photo format, or nothing
            to update (the API answers 400).
    """
    patch: dict = {}

    if username is not None:
        username = username.strip()
        if not username:
            raise ValueError("Username cannot be empty.")
        patch["username"] = username

    if photo_url is not None:
        photo_url = photo_url.strip()
        if not photo_url.lower().endswith(ALLOWED_PHOTO_EXTENSIONS):
            raise ValueError(
                "Unsupported file format. Please upload JPG, JPEG, or PNG."
            )
        patch["profile_image_url"] = photo_url

    if phone is not None:
        patch["phone"] = phone.strip()

    if not patch:
        raise ValueError(
            "Provide a username, a phone number, or a photo to update."
        )
    return patch


class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

    async def get_profile(self, user_id: str) -> dict:
        """MD-08 — the caller's `users` row.

        Raises:
            UserProfileNotFound: no profile row for this account.
        """
        return await asyncio.to_thread(self.repo.get_user_profile, user_id)

    async def update_location(
        self, user_id: str, latitude: float, longitude: float,
    ) -> dict | None:
        """MD-14 — store the device's position and timestamp.

        Returns the updated row, or None when the account has no profile row.
        """
        point = create_postgis_point(latitude, longitude)
        return await asyncio.to_thread(
            self.repo.update_last_location, user_id, point,
        )

    async def update_profile(
        self,
        user_id: str,
        *,
        username: str | None = None,
        photo_url: str | None = None,
        phone: str | None = None,
    ) -> dict | None:
        """MD-46 — edit the caller's username, phone and/or photo.

        Returns the updated row, or None when the account has no profile row.

        Raises:
            ValueError: the edit failed validation. Nothing is written.
            UserProfileNotFound: the adapter could not find the profile row.
        """
        patch = build_profile_patch(username, photo_url, phone)
        return await asyncio.to_thread(self.repo.update_profile, user_id, patch)
