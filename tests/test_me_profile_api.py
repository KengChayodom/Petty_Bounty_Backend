"""
Route unit tests for PATCH /me — profile edit (UTC-43/44, MD-46/47, SRS-74/75).

The spec (`progress_2/method_specification.md`) maps BOTH MD-46 (username) and
MD-47 (photo) to a single `PATCH /me`, so the two test-plan blocks exercise one
route (`me.update_my_profile`) through its two fields rather than two functions.

Boundary rule (matches the reconciled Progress-2 plan): the auth dependency and
the `UserRepository` port are the seams, replaced via FastAPI
dependency_overrides with `MagicMock(spec=UserRepository)`. We assert on the
HTTP status, on whether the port was called at all (validation must short-circuit
BEFORE any write), and on the exact `(user_id, patch)` handed to `update_profile`
— the call args ARE the "written to the caller's own row, scoped to the JWT id"
behaviour the state-based test-plan cells describe.
"""
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import me as me_api
from app.core.auth import get_current_user_id
from app.repositories.user_repository import UserProfileNotFound, UserRepository


def _client(repo, user_id="u1"):
    app = FastAPI()
    app.include_router(me_api.router)
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[me_api.get_user_repository] = lambda: repo
    return TestClient(app)


def _repo(profile=None):
    """A UserRepository double whose update_profile returns `profile` (the
    re-read projection on success) or None (missing row)."""
    repo = MagicMock(spec=UserRepository)
    repo.update_profile.return_value = profile
    return repo


# --------------------------------------------------------------------------- #
# UTC-43: update the profile username (MD-46, SRS-74)
# --------------------------------------------------------------------------- #
class TestUpdateProfileName:
    def test_empty_name_yields_400_and_repo_unchanged(self):
        """UTC-43-TC-01 — blank username is rejected before any write."""
        repo = _repo()
        r = _client(repo).patch("/me", json={"display_name": ""})

        assert r.status_code == 400
        repo.update_profile.assert_not_called()

    def test_writes_display_name_scoped_to_self(self):
        """UTC-43-TC-02 — the username is written to the caller's own row."""
        updated = {"id": "u1", "display_name": "Kus"}
        repo = _repo(profile=updated)
        r = _client(repo, user_id="u1").patch("/me", json={"display_name": "Kus"})

        assert r.status_code == 200
        assert r.json()["data"] == updated
        # Scoping is structural: user_id comes from the JWT, and the patch
        # carries the new username on the display_name column.
        repo.update_profile.assert_called_once_with("u1", {"display_name": "Kus"})

    def test_missing_profile_yields_404(self):
        """UTC-43-TC-03 — no such row -> 404."""
        repo = _repo(profile=None)
        r = _client(repo, user_id="ghost").patch(
            "/me", json={"display_name": "Kus"}
        )

        assert r.status_code == 404

    def test_database_error_yields_500(self):
        """UTC-43-TC-04 — an unexpected repo failure surfaces as 500."""
        repo = _repo()
        repo.update_profile.side_effect = Exception("connection reset")
        r = _client(repo).patch("/me", json={"display_name": "Kus"})

        assert r.status_code == 500


# --------------------------------------------------------------------------- #
# UTC-44: update the profile photograph (MD-47, SRS-75)
# --------------------------------------------------------------------------- #
class TestUpdateProfilePhone:
    """UTC-43-TC-05 to TC-08 — the phone half of MD-46 (SRS-99).

    It shipped with the username field and had no test of any kind until
    2026-09-02, which is how the requirement it realises (SRS-99) came to be
    written in the use-case document and nowhere else.
    """

    def test_writes_phone_scoped_to_self(self):
        """UTC-43-TC-05 — the number is written to the caller's own row."""
        updated = {"id": "u1", "phone": "0812345678"}
        repo = _repo(profile=updated)
        r = _client(repo, user_id="u1").patch("/me", json={"phone": "0812345678"})

        assert r.status_code == 200
        repo.update_profile.assert_called_once_with("u1", {"phone": "0812345678"})

    def test_phone_alone_is_a_valid_edit(self):
        """UTC-43-TC-06 — a phone-only PATCH is not the empty PATCH.

        The three fields are independent, so saving the number without touching
        the username or the photo must reach the write rather than fall into the
        "nothing supplied" 400.
        """
        repo = _repo(profile={"id": "u1", "phone": "0899999999"})
        r = _client(repo).patch("/me", json={"phone": "0899999999"})

        assert r.status_code == 200
        patch = repo.update_profile.call_args[0][1]
        assert set(patch) == {"phone"}

    def test_phone_is_trimmed_and_not_format_checked(self):
        """UTC-43-TC-07 — surrounding space is stripped, the number itself is
        taken as given.

        `users.phone` is free text and no requirement specifies a format, so the
        route deliberately applies none. Pinning that here means a format rule
        added later has to be a decision, not a silent regression.
        """
        repo = _repo(profile={"id": "u1"})
        r = _client(repo).patch("/me", json={"phone": "  +66 81 234 5678  "})

        assert r.status_code == 200
        repo.update_profile.assert_called_once_with(
            "u1", {"phone": "+66 81 234 5678"}
        )

    def test_all_three_fields_travel_in_one_patch(self):
        """UTC-43-TC-08 — username, phone and photo are one write, not three.

        The edit dialog saves them together, so the route has to fold them into
        a single `update_profile` call on the three real columns.
        """
        repo = _repo(profile={"id": "u1"})
        r = _client(repo).patch(
            "/me",
            json={
                "display_name": "Kus",
                "phone": "0812345678",
                "photo_url": "https://storage.test/u1.jpg",
            },
        )

        assert r.status_code == 200
        repo.update_profile.assert_called_once_with(
            "u1",
            {
                "display_name": "Kus",
                "profile_image_url": "https://storage.test/u1.jpg",
                "phone": "0812345678",
            },
        )


class TestUpdateProfilePhoto:
    def test_missing_or_invalid_url_yields_400_and_repo_unchanged(self):
        """UTC-44-TC-01 — empty / unsupported photo URL is rejected pre-write."""
        repo = _repo()
        r = _client(repo).patch("/me", json={"photo_url": ""})

        assert r.status_code == 400
        repo.update_profile.assert_not_called()

    def test_non_image_extension_yields_400(self):
        """UTC-44-TC-01 (format arc) — a non JPG/JPEG/PNG URL is rejected."""
        repo = _repo()
        r = _client(repo).patch(
            "/me", json={"photo_url": "http://x/a.gif"}
        )

        assert r.status_code == 400
        repo.update_profile.assert_not_called()

    def test_writes_photo_url_scoped_to_self(self):
        """UTC-44-TC-02 — the photo URL is written to the caller's own row."""
        updated = {"id": "u1", "profile_image_url": "http://x/a.jpg"}
        repo = _repo(profile=updated)
        r = _client(repo, user_id="u1").patch(
            "/me", json={"photo_url": "http://x/a.jpg"}
        )

        assert r.status_code == 200
        assert r.json()["data"] == updated
        # Request field is `photo_url`; it lands on the profile_image_url column.
        repo.update_profile.assert_called_once_with(
            "u1", {"profile_image_url": "http://x/a.jpg"}
        )

    def test_missing_profile_yields_404(self):
        """UTC-44-TC-03 — no such row -> 404 (both None and the port's own
        UserProfileNotFound map to 404)."""
        repo = _repo()
        repo.update_profile.side_effect = UserProfileNotFound("ghost")
        r = _client(repo, user_id="ghost").patch(
            "/me", json={"photo_url": "http://x/a.jpg"}
        )

        assert r.status_code == 404

    def test_database_error_yields_500(self):
        """UTC-44-TC-04 — an unexpected repo failure surfaces as 500."""
        repo = _repo()
        repo.update_profile.side_effect = Exception("connection reset")
        r = _client(repo).patch("/me", json={"photo_url": "http://x/a.jpg"})

        assert r.status_code == 500


# --------------------------------------------------------------------------- #
# Empty patch — neither field supplied is a 400 (no-op writes are refused).
# --------------------------------------------------------------------------- #
def test_empty_patch_yields_400():
    repo = _repo()
    r = _client(repo).patch("/me", json={})

    assert r.status_code == 400
    repo.update_profile.assert_not_called()
