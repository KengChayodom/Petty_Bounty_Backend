"""
Route unit tests for PATCH /me — profile edit (UTC-43, MD-46,
SRS-73 username, SRS-74 photograph, SRS-99 phone).

The spec (`progress_2/method_specification.md`) maps all three requirements onto
MD-46, a single `PATCH /me`, and UTC-43 is the one test-plan block for it. The
classes below group its twelve cases by the field of the payload they exercise.
UTC-44 held the photograph half until 2026-09-11 and is retired.

Boundary rule (matches the reconciled Progress-2 plan): the auth dependency and
the `UserRepository` port are the seams, replaced via FastAPI
dependency_overrides with `MagicMock(spec=UserRepository)`. We assert on the
HTTP status, on whether the port was called at all (validation must short-circuit
BEFORE any write), and on the exact `(user_id, patch)` handed to `update_profile`
— the call args ARE the "written to the caller's own row, scoped to the JWT id"
behaviour the state-based test-plan cells describe.
"""
from unittest.mock import MagicMock

import pytest
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
# UTC-43-TC-01 to TC-04 and TC-06: the username (MD-46, SRS-73)
# --------------------------------------------------------------------------- #
class TestUpdateProfileName:
    def test_empty_name_yields_400_and_repo_unchanged(self):
        """UTC-43-TC-01 — blank username is rejected before any write."""
        repo = _repo()
        r = _client(repo).patch("/me", json={"username": ""})

        assert r.status_code == 400
        repo.update_profile.assert_not_called()

    def test_writes_username_scoped_to_self(self):
        """UTC-43-TC-02 — the username is written to the caller's own row."""
        updated = {"id": "u1", "username": "Kus"}
        repo = _repo(profile=updated)
        r = _client(repo, user_id="u1").patch("/me", json={"username": "Kus"})

        assert r.status_code == 200
        assert r.json()["data"] == updated
        # Scoping is structural: user_id comes from the JWT, and the patch
        # carries the new username on the username column.
        repo.update_profile.assert_called_once_with("u1", {"username": "Kus"})

    def test_the_username_is_trimmed(self):
        """UTC-43-TC-06 — surrounding space is stripped before the write, the
        same rule the phone number is held to. Unframed until 2026-09-07,
        which left the strip that decides whether a name is blank asserted on
        the refusal path only."""
        repo = _repo(profile={"id": "u1"})
        r = _client(repo).patch("/me", json={"username": "  Kus  "})

        assert r.status_code == 200
        repo.update_profile.assert_called_once_with("u1", {"username": "Kus"})

    def test_missing_profile_yields_404(self):
        """UTC-43-TC-03 — no such row -> 404."""
        repo = _repo(profile=None)
        r = _client(repo, user_id="ghost").patch(
            "/me", json={"username": "Kus"}
        )

        assert r.status_code == 404

    def test_database_error_yields_500(self):
        """UTC-43-TC-04 — an unexpected repo failure surfaces as 500."""
        repo = _repo()
        repo.update_profile.side_effect = Exception("connection reset")
        r = _client(repo).patch("/me", json={"username": "Kus"})

        assert r.status_code == 500


# --------------------------------------------------------------------------- #
# UTC-43-TC-05 to TC-07: the phone number and the whole payload (MD-46, SRS-99)
# --------------------------------------------------------------------------- #
class TestUpdateProfilePhone:
    """UTC-43-TC-05 to TC-07 — the phone half of MD-46 (SRS-99).

    It shipped with the username field and had no test of any kind until
    2026-09-02, which is how the requirement it realises (SRS-99) came to be
    written in the use-case document and nowhere else.
    """

    def test_writes_phone_scoped_to_self(self):
        """UTC-43-TC-05 — the number is written to the caller's own row, and
        a phone-only edit is not the empty edit.

        Absorbed a struck duplicate on 2026-09-07: that case sent this identical
        body, and the exact-call assertion below already establishes both that
        the write happened and that the patch carries the phone alone."""
        updated = {"id": "u1", "phone": "0812345678"}
        repo = _repo(profile=updated)
        r = _client(repo, user_id="u1").patch("/me", json={"phone": "0812345678"})

        assert r.status_code == 200
        repo.update_profile.assert_called_once_with("u1", {"phone": "0812345678"})

    def test_phone_is_trimmed_and_not_format_checked(self):
        """UTC-43-TC-06 — surrounding space is stripped, the number itself is
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
        """UTC-43-TC-07 — username, phone and photo are one write, not three.

        The edit dialog saves them together, so the route has to fold them into
        a single `update_profile` call on the three real columns.
        """
        repo = _repo(profile={"id": "u1"})
        r = _client(repo).patch(
            "/me",
            json={
                "username": "Kus",
                "phone": "0812345678",
                "photo_url": "https://storage.test/u1.jpg",
            },
        )

        assert r.status_code == 200
        repo.update_profile.assert_called_once_with(
            "u1",
            {
                "username": "Kus",
                "profile_image_url": "https://storage.test/u1.jpg",
                "phone": "0812345678",
            },
        )


class TestUpdateProfilePhoto:
    @pytest.mark.parametrize("address", ["", "   ", "http://x/a.gif"])
    def test_an_address_outside_the_accepted_formats_is_refused(self, address):
        """UTC-43-TC-09 [error] — one check decides this, so an empty address
        and an address ending in an unaccepted extension are one choice and
        take one frame between them. The values are checked together rather
        than in separate cases.

        Absorbed a struck duplicate on 2026-09-07, which framed the same choice
        a second time."""
        repo = _repo()
        r = _client(repo).patch("/me", json={"photo_url": address})

        assert r.status_code == 400
        repo.update_profile.assert_not_called()

    def test_writes_photo_url_scoped_to_self(self):
        """UTC-43-TC-10 — the photo URL is written to the caller's own row."""
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
        """UTC-43-TC-11 — no such row means 404 (both None and the port's own
        UserProfileNotFound map to 404)."""
        repo = _repo()
        repo.update_profile.side_effect = UserProfileNotFound("ghost")
        r = _client(repo, user_id="ghost").patch(
            "/me", json={"photo_url": "http://x/a.jpg"}
        )

        assert r.status_code == 404

    # The database-failure path is framed once, by UTC-43-TC-04 above. It never
    # reads which field was sent, so a second case here asserted the same thing
    # with a different body. Struck on 2026-09-11 with the merge of UTC-44.

    def test_an_accepted_extension_is_matched_whatever_its_casing(self):
        """UTC-43-TC-12 [single] — the boundary of the format check. The
        comparison lower-cases the address first, so a camera that names its
        files .JPG is accepted. Unframed until 2026-09-07, which left the
        lower-casing free to be removed without a test noticing."""
        repo = _repo(profile={"id": "u1"})
        r = _client(repo).patch("/me", json={"photo_url": "http://x/A.JPG"})

        assert r.status_code == 200
        repo.update_profile.assert_called_once_with(
            "u1", {"profile_image_url": "http://x/A.JPG"}
        )


# --------------------------------------------------------------------------- #
# Empty patch — neither field supplied is a 400 (no-op writes are refused).
# --------------------------------------------------------------------------- #
def test_empty_patch_yields_400():
    """UTC-43-TC-08 — a PATCH supplying none of the three fields is refused
    before any write, so a no-op edit cannot reach the database."""
    repo = _repo()
    r = _client(repo).patch("/me", json={})

    assert r.status_code == 400
    repo.update_profile.assert_not_called()
