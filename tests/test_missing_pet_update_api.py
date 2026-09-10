"""
Route unit tests for PATCH /missing-pets/{pet_id} - the owner's report edit
and closure (UTC-35, MD-39, SRS-65 editable fields, SRS-66 closure).

Case selection. The cases below are the test frames of the category-partition
table in progress_2/test_plan.md, one test method per frame. Each category of
the handler's input is partitioned into choices, one case is written per
choice, an [error] choice takes exactly one frame and is never combined with
another, and a boundary marked [single] takes exactly one frame of its own.
The closure category is annotated [if CLOSING] because it exists only when the
status choice is the one that ends the search.

Boundary rule (matches every other route block): the auth dependency and the
`MissingPetRepository` port are the seams. The Supabase client is overridden
with an inert double only so the route can build something; the repository the
route would have built is replaced at the module boundary, and every assertion
is on the HTTP status or on the exact `(pet_id, owner_id, patch)` handed to
`update_missing_pet_owned`. Those call args ARE the behaviour the plan
describes: the patch carries only what the client sent, and the owner_id
scoping it comes from the JWT.

The closure fan-out (status Found also closes the pet's sightings) is part of
this handler and so is tested here. It lived in
tests/test_owner_loop_api.py::TestEndSearchClosesSightings until 2026-09-07,
where three of its five tests repeated frames this file already had.
"""
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import missing_pets as pets_api
from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client
from app.repositories.missing_pet_repository import MissingPetRepository

JWT_OWNER = "owner-1"


@pytest.fixture
def repo(monkeypatch):
    """The repository port the route builds inline, replaced at the module
    boundary. Returns the edited row by default, as a matched write does."""
    r = MagicMock(spec=MissingPetRepository)
    r.update_missing_pet_owned.return_value = {"id": "p1"}
    r.close_sightings_for_pet.return_value = 2
    monkeypatch.setattr(pets_api, "SupabaseMissingPetRepository", lambda db: r)
    return r


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(pets_api.router)
    app.dependency_overrides[get_current_user_id] = lambda: JWT_OWNER
    app.dependency_overrides[get_supabase_client] = lambda: MagicMock()
    return TestClient(app)


def _patch_sent(repo):
    """The patch dict the route handed the repository."""
    return repo.update_missing_pet_owned.call_args.args[2]


# --------------------------------------------------------------------------- #
# Category: how many fields the patch carries
# --------------------------------------------------------------------------- #
class TestPatchContent:
    def test_empty_body_yields_400_and_no_write(self, client, repo):
        """UTC-35-TC-01 [error] - an edit carrying no field is a 400, and the
        row is never touched. Reaching the database with an empty patch would
        either error there or write nothing while reporting success."""
        r = client.patch("/missing-pets/p1", json={})

        assert r.status_code == 400
        repo.update_missing_pet_owned.assert_not_called()

    def test_all_null_body_yields_400_and_no_write(self, client, repo):
        """UTC-35-TC-02 [error] - a body whose every field is null is the same
        empty edit reached through the other clause of the filter. Null means
        "not supplied" on this route, so it must not be written over the
        stored value."""
        r = client.patch(
            "/missing-pets/p1",
            json={"pet_name": None, "status": None, "bounty_amount": None},
        )

        assert r.status_code == 400
        repo.update_missing_pet_owned.assert_not_called()

    def test_editable_fields_are_written_scoped_to_the_jwt_owner(
        self, client, repo
    ):
        """UTC-35-TC-03 [single] - the upper boundary of the field count:
        every editable field at once reaches the row unchanged, and the write
        is scoped to the report id and the caller's own id. This is also the
        frame for the valid lower-case colour, which the schema upper-cases.
        """
        updated = {
            "id": "p1",
            "pet_name": "Mochi",
            "bounty_amount": 500.0,
            "characteristics": {"color": "White"},
            "primary_color_hex": "#FFFFFF",
        }
        repo.update_missing_pet_owned.return_value = updated

        r = client.patch(
            "/missing-pets/p1",
            json={
                "pet_name": "Mochi",
                "bounty_amount": 500.0,
                "characteristics": {"color": "White"},
                "primary_color_hex": "#ffffff",
            },
        )

        assert r.status_code == 200
        assert r.json()["data"] == updated
        repo.update_missing_pet_owned.assert_called_once_with(
            "p1",
            JWT_OWNER,
            {
                "pet_name": "Mochi",
                "bounty_amount": 500.0,
                "characteristics": {"color": "White"},
                # normalised to upper case by the schema validator
                "primary_color_hex": "#FFFFFF",
            },
        )

    def test_unsent_fields_are_absent_from_the_patch(self, client, repo):
        """UTC-35-TC-04 - the lower boundary of the field count, and the frame
        for the status category's "absent" choice. A partial edit patches only
        what was sent, so an edit of the name cannot blank the bounty, and with
        no status on the patch no closure is attempted.

        Absorbed the closure half of a struck duplicate on 2026-09-07: that case
        sent this identical body and so was this same frame."""
        r = client.patch("/missing-pets/p1", json={"pet_name": "Mochi"})

        assert r.status_code == 200
        assert _patch_sent(repo) == {"pet_name": "Mochi"}
        repo.close_sightings_for_pet.assert_not_called()

    def test_an_owner_id_in_the_body_is_ignored(self, client, repo):
        """UTC-35-TC-06 - the caller cannot re-scope the write. An owner_id
        sent in the body reaches neither the scoping argument nor the patch;
        the identity comes from the verified token."""
        client.patch(
            "/missing-pets/p1",
            json={"pet_name": "Mochi", "owner_id": "somebody-else"},
        )

        pet_id, owner_id, patch = repo.update_missing_pet_owned.call_args.args
        assert (pet_id, owner_id) == ("p1", JWT_OWNER)
        assert "owner_id" not in patch


# --------------------------------------------------------------------------- #
# Category: bounty_amount
# --------------------------------------------------------------------------- #
class TestBountyAmount:
    def test_a_bounty_of_zero_is_written_not_dropped(self, client, repo):
        """UTC-35-TC-05 [single] - the lower boundary of the permitted range,
        and the value the payload filter is most likely to lose. Withdrawing
        the reward is an edit to zero, which is falsy; the filter tests for
        null, not for truthiness, so zero survives it."""
        client.patch("/missing-pets/p1", json={"bounty_amount": 0})

        assert _patch_sent(repo) == {"bounty_amount": 0.0}

    def test_a_negative_bounty_is_refused(self, client, repo):
        """UTC-35-TC-08 [error] - just outside the lower boundary. The schema
        declares ge=0, so a negative reward is refused before any write rather
        than being stored as a debt against the finder."""
        r = client.patch("/missing-pets/p1", json={"bounty_amount": -1})

        assert r.status_code == 422
        repo.update_missing_pet_owned.assert_not_called()


# --------------------------------------------------------------------------- #
# Category: pet_name
# --------------------------------------------------------------------------- #
class TestPetName:
    def test_an_empty_name_is_refused(self, client, repo):
        """UTC-35-TC-08 [error] - just outside the lower length boundary. The
        schema declares min_length=1, so an owner who clears the field is
        refused rather than left with a nameless report on the map."""
        r = client.patch("/missing-pets/p1", json={"pet_name": ""})

        assert r.status_code == 422
        repo.update_missing_pet_owned.assert_not_called()

    def test_a_name_beyond_the_length_limit_is_refused(self, client, repo):
        """UTC-35-TC-08 [error] - just outside the upper length boundary. The
        schema declares max_length=255, so a name one character longer is
        refused here rather than by the column at write time."""
        r = client.patch("/missing-pets/p1", json={"pet_name": "x" * 256})

        assert r.status_code == 422
        repo.update_missing_pet_owned.assert_not_called()


# --------------------------------------------------------------------------- #
# Category: primary_color_hex
# --------------------------------------------------------------------------- #
class TestPrimaryColour:
    def test_an_unparseable_colour_is_rejected(self, client, repo):
        """UTC-63-TC-01 [error] - a colour that is not #RRGGBB is refused
        before any write, by the same rule the create path enforces. An
        unparseable hex in the column makes the colour re-ranking drop the
        report from its own owner's matches."""
        r = client.patch(
            "/missing-pets/p1", json={"primary_color_hex": "not-a-colour"},
        )

        assert r.status_code == 422
        repo.update_missing_pet_owned.assert_not_called()


# --------------------------------------------------------------------------- #
# Category: characteristics
# --------------------------------------------------------------------------- #
class TestCharacteristics:
    def test_an_empty_characteristics_object_is_refused(self, client, repo):
        """UTC-62-TC-01 [error] - an empty object is not a description. The
        create path has always refused one; the edit path did not until
        2026-09-07, so an owner could blank the coat description of their own
        lost pet with a request the schema called valid."""
        r = client.patch("/missing-pets/p1", json={"characteristics": {}})

        assert r.status_code == 422
        repo.update_missing_pet_owned.assert_not_called()


# --------------------------------------------------------------------------- #
# Category: status, and the closure it triggers  (SRS-66)
# --------------------------------------------------------------------------- #
class TestStatusAndClosure:
    def test_found_is_written_and_closes_the_pets_sightings(
        self, client, repo
    ):
        """UTC-35-TC-07 [property CLOSING] - closing the search is an edit of
        the status column like any other, so the closed status is what reaches
        the row, and the sightings of that report stop being live leads the
        moment the pet is home.

        Absorbed a struck duplicate on 2026-09-07: that case sent this identical
        body and so was this same frame, asserting only the second half."""
        repo.update_missing_pet_owned.return_value = {
            "id": "p1",
            "status": "Found",
        }

        r = client.patch("/missing-pets/p1", json={"status": "Found"})

        assert r.status_code == 200
        assert _patch_sent(repo) == {"status": "Found"}
        repo.close_sightings_for_pet.assert_called_once_with("p1")

    def test_still_searching_leaves_sightings_alone(self, client, repo):
        """UTC-35-TC-11 - re-opening a search is not the end of one. The
        status choice differs from TC-04's absent status, which is why both
        frames exist even though both reach the same branch."""
        r = client.patch("/missing-pets/p1", json={"status": "Searching"})

        assert r.status_code == 200
        assert _patch_sent(repo) == {"status": "Searching"}
        repo.close_sightings_for_pet.assert_not_called()

    def test_a_permitted_status_is_accepted_in_any_casing(self, client, repo):
        """UTC-35-TC-13 [single] - the schema capitalises the value before
        checking it, so the casing a person types never reaches the database
        enumeration. The normalised value is also what the closure guard
        reads, so a lower-case close really does close."""
        repo.update_missing_pet_owned.return_value = {
            "id": "p1",
            "status": "Found",
        }

        r = client.patch("/missing-pets/p1", json={"status": "found"})

        assert r.status_code == 200
        assert _patch_sent(repo) == {"status": "Found"}
        repo.close_sightings_for_pet.assert_called_once_with("p1")

    def test_a_status_an_owner_may_not_write_is_rejected(self, client, repo):
        """UTC-35-TC-08 [error] - 'Resolved' means the bounty was settled,
        which only the administrator writes. It is refused by the schema, so
        no write is attempted."""
        r = client.patch("/missing-pets/p1", json={"status": "Resolved"})

        assert r.status_code == 422
        repo.update_missing_pet_owned.assert_not_called()

    def test_closure_failure_does_not_fail_the_request(self, client, repo):
        """UTC-35-TC-12 [if CLOSING] [error] - the pet IS already marked Found
        by the time the closure runs. Returning 500 would tell the owner their
        closure failed when it did not."""
        repo.update_missing_pet_owned.return_value = {
            "id": "p1",
            "status": "Found",
        }
        repo.close_sightings_for_pet.side_effect = RuntimeError("db down")

        r = client.patch("/missing-pets/p1", json={"status": "Found"})

        assert r.status_code == 200
        assert r.json()["data"]["status"] == "Found"


# --------------------------------------------------------------------------- #
# Category: what the scoped write returns
# --------------------------------------------------------------------------- #
class TestWriteOutcome:
    def test_no_row_matched_yields_404_and_closes_nothing(self, client, repo):
        """UTC-35-TC-09 [error] - a report that does not exist and one
        belonging to somebody else both match zero rows and both answer 404,
        so the reply never discloses that another owner's report exists. The
        frame carries the closing status because that is the combination in
        which no row matched has a second consequence: a stranger must not be
        able to close a report's sightings by editing it.

        Absorbed a struck duplicate on 2026-09-07, which was this same [error]
        choice framed a second time."""
        repo.update_missing_pet_owned.return_value = None

        r = client.patch("/missing-pets/p1", json={"status": "Found"})

        assert r.status_code == 404
        # One message for both cases: the caller cannot tell them apart.
        assert "not found or not owned by you" in r.json()["detail"]
        repo.close_sightings_for_pet.assert_not_called()

    def test_a_repository_failure_becomes_500(self, client, repo):
        """UTC-35-TC-10 [error] - a failed write is reported as a failure,
        never as a silent success on a row that did not change."""
        repo.update_missing_pet_owned.side_effect = RuntimeError("db down")

        r = client.patch("/missing-pets/p1", json={"pet_name": "Mochi"})

        assert r.status_code == 500
