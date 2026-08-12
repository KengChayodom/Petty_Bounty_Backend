"""
Route unit tests for PATCH /me — profile edit (UTC-42/43, MD-41/42, SRS-69/70).

The spec (`progress_2/method_specification.md`) maps BOTH MD-41 (username) and
MD-42 (photo) to a single `PATCH /me`, so the two test-plan blocks exercise one
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
# UTC-42: update the profile username (MD-41, SRS-69)
# --------------------------------------------------------------------------- #
class TestUpdateProfileName:
    def test_empty_name_yields_400_and_repo_unchanged(self):
        """UTC-42-TC-01 — blank username is rejected before any write."""
        repo = _repo()
        r = _client(repo).patch("/me", json={"display_name": ""})

        assert r.status_code == 400
        repo.update_profile.assert_not_called()

    def test_writes_display_name_scoped_to_self(self):
        """UTC-42-TC-02 — the username is written to the caller's own row."""
        updated = {"id": "u1", "display_name": "Kus"}
        repo = _repo(profile=updated)
        r = _client(repo, user_id="u1").patch("/me", json={"display_name": "Kus"})

        assert r.status_code == 200
        assert r.json()["data"] == updated
        # Scoping is structural: user_id comes from the JWT, and the patch
        # carries the new username on the display_name column.
        repo.update_profile.assert_called_once_with("u1", {"display_name": "Kus"})

    def test_missing_profile_yields_404(self):
        """UTC-42-TC-03 — no such row -> 404."""
        repo = _repo(profile=None)
        r = _client(repo, user_id="ghost").patch(
            "/me", json={"display_name": "Kus"}
        )

        assert r.status_code == 404

    def test_database_error_yields_500(self):
        """UTC-42-TC-04 — an unexpected repo failure surfaces as 500."""
        repo = _repo()
        repo.update_profile.side_effect = Exception("connection reset")
        r = _client(repo).patch("/me", json={"display_name": "Kus"})

        assert r.status_code == 500


# --------------------------------------------------------------------------- #
# UTC-43: update the profile photograph (MD-42, SRS-70)
# --------------------------------------------------------------------------- #
class TestUpdateProfilePhoto:
    def test_missing_or_invalid_url_yields_400_and_repo_unchanged(self):
        """UTC-43-TC-01 — empty / unsupported photo URL is rejected pre-write."""
        repo = _repo()
        r = _client(repo).patch("/me", json={"photo_url": ""})

        assert r.status_code == 400
        repo.update_profile.assert_not_called()

    def test_non_image_extension_yields_400(self):
        """UTC-43-TC-01 (format arc) — a non JPG/JPEG/PNG URL is rejected."""
        repo = _repo()
        r = _client(repo).patch(
            "/me", json={"photo_url": "http://x/a.gif"}
        )

        assert r.status_code == 400
        repo.update_profile.assert_not_called()

    def test_writes_photo_url_scoped_to_self(self):
        """UTC-43-TC-02 — the photo URL is written to the caller's own row."""
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
        """UTC-43-TC-03 — no such row -> 404 (both None and the port's own
        UserProfileNotFound map to 404)."""
        repo = _repo()
        repo.update_profile.side_effect = UserProfileNotFound("ghost")
        r = _client(repo, user_id="ghost").patch(
            "/me", json={"photo_url": "http://x/a.jpg"}
        )

        assert r.status_code == 404

    def test_database_error_yields_500(self):
        """UTC-43-TC-04 — an unexpected repo failure surfaces as 500."""
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
