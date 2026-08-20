"""
Unit tests for app/services/pet_service.py:

  * UTC-12  get_nearby_missing_pets (MD-15, SRS-24) — Home Map proximity query:
    WKT centre in POINT(lng lat) form, km->m conversion, limit passthrough,
    empty-result normalisation, transport-error re-raise.
  * UTC-13  register_missing_pet (MD-11, SRS-47, trigger of SRS-21) — owner
    report runs the same mask-isolate + CLIP path as the live sighting save
    (full-frame fallback on a YOLO miss), builds the PostGIS point, inserts with
    status "Searching" and the feature vector; a no-row insert raises.

Boundary rule (per db-testing-seams): the DB is reached only through the
MissingPetRepository port, so we double THAT with MagicMock(spec=...). The AI
pipeline is mocked at the AIManager class boundary. We assert on the payload the
service hands the repo (the insert contract) and on the semantic arguments to
the nearby query — the calls that ARE the behaviour — plus the returned rows.
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.repositories.missing_pet_repository import (
    MissingPetNotSaved,
    MissingPetRepository,
)
from app.schemas.missing_pets import MissingPetCreate
from app.services.ai_service import AIManager
from app.services.pet_service import PetService


def run(coro):
    return asyncio.run(coro)


def _repo():
    return MagicMock(spec=MissingPetRepository)


# --------------------------------------------------------------------------- #
# UTC-12  get_nearby_missing_pets
# --------------------------------------------------------------------------- #
class TestGetNearbyMissingPets:
    def test_builds_wkt_centre_metre_radius_and_limit(self):
        repo = _repo()
        repo.get_nearby_missing_pets.return_value = [{"id": "pet-1"}]

        rows = run(PetService.get_nearby_missing_pets(
            repo, latitude=13.7563, longitude=100.5018, radius_km=5.0, limit=7,
        ))

        assert rows == [{"id": "pet-1"}]
        # WKT is POINT(lng lat); km -> metres; limit passed through verbatim.
        repo.get_nearby_missing_pets.assert_called_once_with(
            "POINT(100.5018 13.7563)", 5000.0, 7
        )

    def test_defaults_radius_and_limit(self):
        repo = _repo()
        repo.get_nearby_missing_pets.return_value = []

        run(PetService.get_nearby_missing_pets(repo, latitude=0.0, longitude=0.0))

        repo.get_nearby_missing_pets.assert_called_once_with(
            "POINT(0.0 0.0)", 10000.0, 20
        )

    def test_no_rows_returns_empty_list(self):
        repo = _repo()
        repo.get_nearby_missing_pets.return_value = []
        assert run(PetService.get_nearby_missing_pets(repo, 1.0, 2.0)) == []

    def test_rpc_error_is_reraised(self):
        repo = _repo()
        repo.get_nearby_missing_pets.side_effect = RuntimeError("RPC transport error")
        with pytest.raises(RuntimeError):
            run(PetService.get_nearby_missing_pets(repo, 1.0, 2.0))


# --------------------------------------------------------------------------- #
# UTC-13  register_missing_pet
# --------------------------------------------------------------------------- #
def _make_pet(**overrides):
    data = {
        "owner_id": "owner-1",
        "pet_name": "Luna",
        "species": "Dog",
        "characteristics": {"color": "brown"},
        "bounty_amount": 1500,
        "latitude": 13.7563,
        "longitude": 100.5018,
        "last_seen_time": datetime(2026, 6, 1, 12, 0, 0),
        "image_url": "https://img.example/pet.jpg",
    }
    data.update(overrides)
    return MissingPetCreate(**data)


def _patch_ai(monkeypatch, *, isolate_return, vector):
    """Mock the AIManager pipeline at the class boundary.
    download/yolo/clip are awaited; isolate_subject is synchronous."""
    monkeypatch.setattr(AIManager, "download_image", AsyncMock(return_value="SRC_IMG"))
    monkeypatch.setattr(AIManager, "run_yolo_seg", AsyncMock(return_value="YOLO_RESULTS"))
    monkeypatch.setattr(AIManager, "isolate_subject", MagicMock(return_value=isolate_return))
    monkeypatch.setattr(AIManager, "clip_encode", AsyncMock(return_value=vector))


class TestRegisterMissingPet:
    def test_success_inserts_isolated_vector(self, monkeypatch):
        vector = [0.1, 0.2, 0.3]
        _patch_ai(
            monkeypatch,
            isolate_return=("CROP_IMG", "Dog", 0.9, [1.0, 2.0, 3.0, 4.0]),
            vector=vector,
        )
        repo = _repo()
        repo.insert_missing_pet.return_value = {"id": "pet-xyz"}

        created = run(PetService.register_missing_pet(repo, _make_pet(species="Dog")))

        assert created == {"id": "pet-xyz"}
        repo.insert_missing_pet.assert_called_once()
        payload = repo.insert_missing_pet.call_args.args[0]
        assert payload["feature_vector"] == vector
        assert payload["status"] == "Searching"
        assert payload["species"] == "Dog"
        assert payload["last_seen_location"] == "POINT(100.5018 13.7563)"
        # isolate constrained to the user-confirmed species; CLIP encodes the crop
        AIManager.isolate_subject.assert_called_once_with(
            "SRC_IMG", "YOLO_RESULTS", expected_species="Dog"
        )
        AIManager.clip_encode.assert_awaited_once_with("CROP_IMG")

    def test_yolo_miss_falls_back_to_full_frame(self, monkeypatch):
        vector = [0.4, 0.5]
        _patch_ai(monkeypatch, isolate_return=None, vector=vector)
        repo = _repo()
        repo.insert_missing_pet.return_value = {"id": "pet-xyz"}

        run(PetService.register_missing_pet(repo, _make_pet()))

        # On a YOLO miss CLIP encodes the original downloaded frame, and that
        # vector is what gets inserted.
        AIManager.clip_encode.assert_awaited_once_with("SRC_IMG")
        assert repo.insert_missing_pet.call_args.args[0]["feature_vector"] == vector

    def test_insert_returning_no_row_raises_valueerror(self, monkeypatch):
        _patch_ai(
            monkeypatch,
            isolate_return=("CROP_IMG", "Dog", 0.9, [1.0, 2.0, 3.0, 4.0]),
            vector=[0.1],
        )
        repo = _repo()
        # The adapter raises MissingPetNotSaved (a ValueError) on an empty insert.
        repo.insert_missing_pet.side_effect = MissingPetNotSaved({})

        with pytest.raises(ValueError):
            run(PetService.register_missing_pet(repo, _make_pet()))

    def test_ai_pipeline_error_is_wrapped_with_context(self, monkeypatch):
        # A NON-ValueError from the AI pipeline is caught and re-raised as a
        # generic Exception carrying context (distinct from the ValueError path).
        monkeypatch.setattr(
            AIManager, "download_image", AsyncMock(side_effect=RuntimeError("net")))
        repo = _repo()

        with pytest.raises(Exception) as ei:
            run(PetService.register_missing_pet(repo, _make_pet()))

        assert "Failed to register missing pet" in str(ei.value)
        repo.insert_missing_pet.assert_not_called()


# --------------------------------------------------------------------------- #
# get_missing_pet_by_id — the row PLUS the derived badge the list endpoint
# already attaches, so the Status Tracker reads the rule instead of owning a
# second copy of it. A repo error must still propagate.
# --------------------------------------------------------------------------- #
class TestGetMissingPetById:
    def test_returns_repo_row_with_the_derived_badge(self):
        repo = _repo()
        repo.get_missing_pet_by_id.return_value = {
            "id": "pet-1", "latitude": 13.7, "status": "Searching",
        }
        repo.get_sighting_links_for_pets.return_value = []

        assert run(PetService.get_missing_pet_by_id(repo, "pet-1")) == {
            "id": "pet-1",
            "latitude": 13.7,
            "status": "Searching",
            "sighting_count": 0,
            "post_status": "Pending",
        }
        repo.get_sighting_links_for_pets.assert_called_once_with(["pet-1"])

    def test_badge_counts_the_pets_sightings(self):
        """The count is the product rule (de-duplicated, rejected matches
        dropped), not len(rows) — one sighting reaching the pet from BOTH
        sources is one sighting."""
        repo = _repo()
        repo.get_missing_pet_by_id.return_value = {
            "id": "pet-1", "status": "Searching",
        }
        repo.get_sighting_links_for_pets.return_value = [
            {"pet_id": "pet-1", "sighting_id": "s1", "owner_status": "Pending"},
            {"pet_id": "pet-1", "sighting_id": "s1", "owner_status": None},
            {"pet_id": "pet-1", "sighting_id": "s2", "owner_status": "Rejected"},
        ]

        out = run(PetService.get_missing_pet_by_id(repo, "pet-1"))
        assert out["sighting_count"] == 1
        assert out["post_status"] == "Spotted"

    def test_a_settled_case_reads_rescued(self):
        """'Resolved' (the bounty was paid) closes a search exactly as 'Found'
        does. The client used to test for 'Found' alone and reopened the case
        on screen the moment the money moved."""
        repo = _repo()
        repo.get_missing_pet_by_id.return_value = {
            "id": "pet-1", "status": "Resolved",
        }
        repo.get_sighting_links_for_pets.return_value = []

        assert run(
            PetService.get_missing_pet_by_id(repo, "pet-1")
        )["post_status"] == "Rescued"

    def test_missing_pet_returns_none_without_a_second_query(self):
        repo = _repo()
        repo.get_missing_pet_by_id.return_value = None

        assert run(PetService.get_missing_pet_by_id(repo, "nope")) is None
        repo.get_sighting_links_for_pets.assert_not_called()

    def test_error_is_reraised(self):
        repo = _repo()
        repo.get_missing_pet_by_id.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            run(PetService.get_missing_pet_by_id(repo, "pet-1"))


# --------------------------------------------------------------------------- #
# get_sightings_for_pet — owner timeline; forces include_dismissed=False and
# propagates repo errors.
# --------------------------------------------------------------------------- #
class TestGetSightingsForPet:
    def test_returns_rows_and_forces_include_dismissed_false(self):
        repo = _repo()
        repo.sightings_for_pet.return_value = [{"id": "s1"}]
        out = run(PetService.get_sightings_for_pet(repo, "pet-1", limit=10, offset=5))
        assert out == [{"id": "s1"}]
        # the owner never sees Dismissed reports
        repo.sightings_for_pet.assert_called_once_with(
            "pet-1", 10, 5, include_dismissed=False
        )

    def test_error_is_reraised(self):
        repo = _repo()
        repo.sightings_for_pet.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            run(PetService.get_sightings_for_pet(repo, "pet-1"))


# --------------------------------------------------------------------------- #
# UTC-29  get_my_missing_pets (MD-34, SRS-65) — the owner's "My Reports" list.
#
# As-built note: the test plan writes this as `PetService(pet_repo)
# .get_my_missing_pets(owner_id)`, but PetService is a static-method service
# whose repo is the first argument (as it is for every other method here), so
# the call is `PetService.get_my_missing_pets(repo, owner_id)`.
#
# Owner scoping is STRUCTURAL: the port takes an owner_id, so there is no shape
# this service could pass that would return another owner's rows. TC-01
# therefore asserts the owner_id actually forwarded — the defect it catches is
# a service that reads the id from somewhere other than the verified caller.
# --------------------------------------------------------------------------- #
class TestGetMyMissingPets:
    def test_returns_only_the_callers_reports(self):
        """UTC-29-TC-01 — the caller's own id is what reaches the repo."""
        repo = _repo()
        repo.get_by_owner.side_effect = lambda owner_id: {
            "u1": [{"id": "pet-1", "owner_id": "u1", "status": "Searching"}],
            "u2": [{"id": "pet-2", "owner_id": "u2", "status": "Searching"}],
        }[owner_id]
        repo.get_sighting_links_for_pets.return_value = []

        out = run(PetService.get_my_missing_pets(repo, "u1"))

        assert [p["id"] for p in out] == ["pet-1"]
        repo.get_by_owner.assert_called_once_with("u1")

    def test_empty_when_owner_has_none(self):
        """UTC-29-TC-02 — an owner with no reports gets [], not None, and no
        count query is fired for an empty list of ids."""
        repo = _repo()
        repo.get_by_owner.return_value = []

        assert run(PetService.get_my_missing_pets(repo, "u1")) == []
        repo.get_sighting_links_for_pets.assert_not_called()

    def test_error_is_reraised(self):
        """UTC-29-TC-03 — DB failure propagates (API maps it to 500)."""
        repo = _repo()
        repo.get_by_owner.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            run(PetService.get_my_missing_pets(repo, "u1"))

    # --- derived post status (2026-08-17) --------------------------------- #
    def test_report_with_no_sightings_is_pending(self):
        repo = _repo()
        repo.get_by_owner.return_value = [{"id": "p1", "status": "Searching"}]
        repo.get_sighting_links_for_pets.return_value = []

        out = run(PetService.get_my_missing_pets(repo, "u1"))

        assert out[0]["post_status"] == "Pending"
        assert out[0]["sighting_count"] == 0
        repo.get_sighting_links_for_pets.assert_called_once_with(["p1"])

    def test_report_with_sightings_is_spotted_without_anyone_approving_it(self):
        """The whole point of the new model: a hunter's report alone moves the
        badge — no admin verification step in between."""
        repo = _repo()
        repo.get_by_owner.return_value = [{"id": "p1", "status": "Searching"}]
        repo.get_sighting_links_for_pets.return_value = [
            {"pet_id": "p1", "sighting_id": "s1", "owner_status": None},
            {"pet_id": "p1", "sighting_id": "s2", "owner_status": "Confirmed"},
        ]

        out = run(PetService.get_my_missing_pets(repo, "u1"))

        assert out[0]["post_status"] == "Spotted"
        assert out[0]["sighting_count"] == 2

    def test_status_falls_back_when_the_last_sighting_is_removed(self):
        """The status is derived, so an admin deleting a bogus sighting takes
        the badge back to Pending on the next read. A stored badge would be
        stuck on Spotted and keep telling the owner someone had seen their pet."""
        repo = _repo()
        repo.get_by_owner.return_value = [{"id": "p1", "status": "Searching"}]

        repo.get_sighting_links_for_pets.return_value = [
            {"pet_id": "p1", "sighting_id": "s1", "owner_status": None},
        ]
        assert run(
            PetService.get_my_missing_pets(repo, "u1")
        )[0]["post_status"] == "Spotted"

        repo.get_sighting_links_for_pets.return_value = []   # sighting deleted
        assert run(
            PetService.get_my_missing_pets(repo, "u1")
        )[0]["post_status"] == "Pending"

    def test_closed_search_reads_rescued_even_with_sightings(self):
        repo = _repo()
        repo.get_by_owner.return_value = [{"id": "p1", "status": "Found"}]
        repo.get_sighting_links_for_pets.return_value = [
            {"pet_id": "p1", "sighting_id": f"s{i}", "owner_status": None}
            for i in range(4)
        ]

        out = run(PetService.get_my_missing_pets(repo, "u1"))

        assert out[0]["post_status"] == "Rescued"

    def test_count_failure_propagates(self):
        repo = _repo()
        repo.get_by_owner.return_value = [{"id": "p1", "status": "Searching"}]
        repo.get_sighting_links_for_pets.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            run(PetService.get_my_missing_pets(repo, "u1"))


# --------------------------------------------------------------------------- #
# UTC-31  list_all_missing_pets (MD-37, SRS-64) — admin browse.
#
# The distinction that matters is "no filter" vs "filter on None": passing
# status=None must mean every status, never `status IS NULL`. Both TC-01 and
# TC-02 assert the exact argument handed to the port, because that argument IS
# the filtering behaviour at this layer.
# --------------------------------------------------------------------------- #
class TestListAllMissingPets:
    def test_applies_status_filter_when_given(self):
        """UTC-31-TC-01 — a supplied status is forwarded verbatim."""
        repo = _repo()
        repo.list_all.return_value = [{"id": "pet-1", "status": "Searching"}]

        out = run(PetService.list_all_missing_pets(
            repo, limit=20, offset=0, status="Searching",
        ))

        assert out == [{"id": "pet-1", "status": "Searching"}]
        repo.list_all.assert_called_once_with("Searching", 20, 0)

    def test_no_status_filter_when_none(self):
        """UTC-31-TC-02 — status=None reaches the repo as None (= no filter)."""
        repo = _repo()
        repo.list_all.return_value = [{"id": "pet-1"}, {"id": "pet-2"}]

        out = run(PetService.list_all_missing_pets(repo, limit=20, offset=0))

        assert len(out) == 2
        repo.list_all.assert_called_once_with(None, 20, 0)

    def test_error_is_reraised(self):
        """UTC-31-TC-03 — DB failure propagates (API maps it to 500)."""
        repo = _repo()
        repo.list_all.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            run(PetService.list_all_missing_pets(repo))

    def test_empty_page_returns_empty_list(self):
        """UTC-31-TC-04 — an empty page is [], not None."""
        repo = _repo()
        repo.list_all.return_value = []
        assert run(PetService.list_all_missing_pets(repo, limit=20, offset=0)) == []


# --------------------------------------------------------------------------- #
# UTC-32  remove_missing_pet (MD-38, SRS-66) — admin removal.
#
# UD-14's postcondition is "removed from the database and the search map", so
# the deletion is real; "no row deleted" is the not-found signal, which the
# service must turn into a ValueError rather than reporting a phantom success.
# --------------------------------------------------------------------------- #
class TestRemoveMissingPet:
    def test_removes_the_report(self):
        """UTC-32-TC-01 — the row is deleted and the deleted row returned."""
        repo = _repo()
        repo.remove.return_value = {"id": "p1", "pet_name": "Mochi"}

        out = run(PetService.remove_missing_pet(repo, "p1", "a1"))

        assert out == {"id": "p1", "pet_name": "Mochi"}
        repo.remove.assert_called_once_with("p1")

    def test_not_found_raises_valueerror(self):
        """UTC-32-TC-02 — nothing deleted => ValueError (API maps it to 404)."""
        repo = _repo()
        repo.remove.return_value = None
        with pytest.raises(ValueError):
            run(PetService.remove_missing_pet(repo, "ghost", "a1"))

    def test_error_is_reraised(self):
        """UTC-32-TC-03 — DB failure propagates (API maps it to 500)."""
        repo = _repo()
        repo.remove.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            run(PetService.remove_missing_pet(repo, "p1", "a1"))

    def test_removal_is_audit_logged_with_the_admin(self, caplog):
        """The moderation action is recorded — MD-38 says the removal is a
        recorded action, and the log line is where that record lives."""
        repo = _repo()
        repo.remove.return_value = {"id": "p1"}
        with caplog.at_level("WARNING"):
            run(PetService.remove_missing_pet(repo, "p1", "admin-7"))
        assert "admin-7" in caplog.text and "p1" in caplog.text
