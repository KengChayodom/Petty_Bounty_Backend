"""
Unit tests for app/services/pet_service.py:

  * UTC-12  get_nearby_missing_pets (MD-15, SRS-27) — Home Map proximity query:
    WKT centre in POINT(lng lat) form, km->m conversion, limit passthrough,
    empty-result normalisation, transport-error re-raise.
  * UTC-33  register_missing_pet (MD-37, SRS-56–65, trigger of SRS-24) — owner
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
from app.repositories.pagination import Page
from app.schemas.missing_pets import MissingPetCreate
from app.services.ai_service import AIManager, EmbedResult
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
# UTC-33  register_missing_pet
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


def _patch_embed(monkeypatch, *, result):
    """Mock the one shared embed pipeline at the class boundary. The individual
    download → YOLO → isolate → CLIP wiring is asserted in test_ai_service.py."""
    monkeypatch.setattr(
        AIManager, "embed_image", AsyncMock(return_value=result)
    )


class TestRegisterMissingPet:
    def test_success_inserts_isolated_vector(self, monkeypatch):
        vector = [0.1, 0.2, 0.3]
        _patch_embed(monkeypatch, result=EmbedResult(
            feature_vector=vector, species="Dog", confidence=0.9,
            bbox=[1.0, 2.0, 3.0, 4.0], isolated_image="CROP_IMG",
            used_full_frame=False,
        ))
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
        # embed is constrained to the user-confirmed species, and measures the
        # coat colour the same way the sighting side does
        AIManager.embed_image.assert_awaited_once_with(
            "https://img.example/pet.jpg", expected_species="Dog",
            with_color=True,
        )

    def test_a_report_with_no_colour_stores_the_measured_one(self, monkeypatch):
        """The form's default colour comes from analyze. When it never arrived,
        registration stores the colour measured from the same photo."""
        _patch_embed(monkeypatch, result=EmbedResult(
            feature_vector=[0.1], species="Cat", confidence=0.9,
            bbox=[1.0, 2.0, 3.0, 4.0], isolated_image="CROP_IMG",
            primary_color_hex="#726860", used_full_frame=False,
        ))
        repo = _repo()
        repo.insert_missing_pet.return_value = {"id": "pet-xyz"}

        run(PetService.register_missing_pet(
            repo, _make_pet(species="Cat", primary_color_hex=None)))

        payload = repo.insert_missing_pet.call_args.args[0]
        assert payload["primary_color_hex"] == "#726860"

    def test_yolo_miss_falls_back_to_full_frame(self, monkeypatch):
        vector = [0.4, 0.5]
        _patch_embed(monkeypatch, result=EmbedResult(
            feature_vector=vector, used_full_frame=True,
        ))
        repo = _repo()
        repo.insert_missing_pet.return_value = {"id": "pet-xyz"}

        run(PetService.register_missing_pet(repo, _make_pet()))

        # On a YOLO miss embed_image full-frame-encodes; that vector is inserted.
        # No colour is read off a full frame, so with no owner colour the pet
        # matches on CLIP only.
        payload = repo.insert_missing_pet.call_args.args[0]
        assert payload["feature_vector"] == vector
        assert payload["primary_color_hex"] is None

    def test_insert_returning_no_row_raises_valueerror(self, monkeypatch):
        _patch_embed(monkeypatch, result=EmbedResult(
            feature_vector=[0.1], species="Dog", confidence=0.9,
            bbox=[1.0, 2.0, 3.0, 4.0], isolated_image="CROP_IMG",
        ))
        repo = _repo()
        # The adapter raises MissingPetNotSaved (a ValueError) on an empty insert.
        repo.insert_missing_pet.side_effect = MissingPetNotSaved({})

        with pytest.raises(ValueError):
            run(PetService.register_missing_pet(repo, _make_pet()))

    def test_ai_pipeline_error_is_wrapped_with_context(self, monkeypatch):
        # A NON-ValueError from the AI pipeline is caught and re-raised as a
        # generic Exception carrying context (distinct from the ValueError path).
        monkeypatch.setattr(
            AIManager, "embed_image", AsyncMock(side_effect=RuntimeError("net")))
        repo = _repo()

        with pytest.raises(Exception) as ei:
            run(PetService.register_missing_pet(repo, _make_pet()))

        assert "Failed to register missing pet" in str(ei.value)
        repo.insert_missing_pet.assert_not_called()

    def test_insert_failure_is_wrapped_with_the_same_context(self, monkeypatch):
        """UTC-33-TC-05 [error] - the insert category has two error choices and
        they answer differently: storing no row is a ValueError the route reads
        as a failed creation, while a transport failure is wrapped as the
        generic error the route reads as a server fault. Framing only the first
        left the wrapping clause untested from the repository side."""
        _patch_embed(monkeypatch, result=EmbedResult(
            feature_vector=[0.1], species="Dog", confidence=0.9,
            bbox=[1.0, 2.0, 3.0, 4.0], isolated_image="CROP_IMG",
        ))
        repo = _repo()
        repo.insert_missing_pet.side_effect = RuntimeError("db down")

        with pytest.raises(Exception) as ei:
            run(PetService.register_missing_pet(repo, _make_pet()))

        assert "Failed to register missing pet" in str(ei.value)
        assert not isinstance(ei.value, ValueError)


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

    # --- owner scoping, added 2026-09-09 (UTC-38-TC-03 to TC-05) ------------ #
    # Being signed in was the only check on this read until that date, so any
    # authenticated account could pull any owner's timeline — and these rows
    # carry where a pet was seen plus the hunter's name and telephone number.

    def test_the_owners_own_report_is_read(self):
        repo = _repo()
        repo.get_missing_pet_by_id.return_value = {
            "id": "pet-1", "owner_id": "owner-1"
        }
        repo.sightings_for_pet.return_value = [{"id": "s1"}]
        out = run(PetService.get_sightings_for_pet(
            repo, "pet-1", owner_id="owner-1"
        ))
        assert out == [{"id": "s1"}]

    @pytest.mark.parametrize("pet", [
        {"id": "pet-1", "owner_id": "somebody-else"},   # owned by another
        None,                                          # no such report
    ])
    def test_a_report_the_caller_does_not_own_is_never_read(self, pet):
        """Both answer the same missing-report error, deliberately: a caller who
        owns nothing must not be able to tell an existing report from an absent
        one. The timeline query must not run at all — answering after reading it
        would leak the row count through timing and through any log."""
        repo = _repo()
        repo.get_missing_pet_by_id.return_value = pet
        with pytest.raises(LookupError):
            run(PetService.get_sightings_for_pet(
                repo, "pet-1", owner_id="owner-1"
            ))
        repo.sightings_for_pet.assert_not_called()

    def test_omitting_the_owner_keeps_the_unscoped_read(self):
        """The parameter defaults to None so the signature stayed compatible.
        A caller that omits it gets the old behaviour, which is why the route is
        the thing that must pass it and is asserted separately."""
        repo = _repo()
        repo.sightings_for_pet.return_value = [{"id": "s1"}]
        assert run(PetService.get_sightings_for_pet(repo, "pet-1")) == [{"id": "s1"}]
        repo.get_missing_pet_by_id.assert_not_called()


# --------------------------------------------------------------------------- #
# UTC-34  get_my_missing_pets (MD-38, SRS-68) — the owner's "My Reports" list.
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
        """UTC-34-TC-01 — the caller's own id is what reaches the repo."""
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
        """UTC-34-TC-02 — an owner with no reports gets [], not None, and no
        count query is fired for an empty list of ids."""
        repo = _repo()
        repo.get_by_owner.return_value = []

        assert run(PetService.get_my_missing_pets(repo, "u1")) == []
        repo.get_sighting_links_for_pets.assert_not_called()

    def test_error_is_reraised(self):
        """UTC-34-TC-03 — DB failure propagates (API maps it to 500)."""
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

    def test_an_aged_out_report_with_no_sighting_reads_expired(self):
        """UTC-34-TC-09 [single] - the fourth badge, and the boundary of the
        expiry column. A report that aged out with nothing to show for it is
        no longer matched by the read paths, so the owner is waiting on
        something that can no longer happen and the card must say so. The
        badge rule itself is tested in test_pet_logic.py; what is framed here
        is that this list passes `expires_at` through to it at all."""
        repo = _repo()
        repo.get_by_owner.return_value = [{
            "id": "p1",
            "status": "Searching",
            "expires_at": "2020-01-01T00:00:00Z",
        }]
        repo.get_sighting_links_for_pets.return_value = []

        out = run(PetService.get_my_missing_pets(repo, "u1"))

        assert out[0]["post_status"] == "Expired"
        assert out[0]["sighting_count"] == 0


# --------------------------------------------------------------------------- #
# UTC-36  list_all_missing_pets (MD-41, SRS-71) — admin browse.
#
# The distinction that matters is "no filter" vs "filter on None": passing
# status=None must mean every status, never `status IS NULL`. Both TC-01 and
# TC-02 assert the exact argument handed to the port, because that argument IS
# the filtering behaviour at this layer.
#
# TC-05 pins the OTHER half of the contract: the total travels with the page.
# It is the number the console draws numbered pages from, so a service that
# quietly returned len(items) would produce a pager that stops at page one.
# --------------------------------------------------------------------------- #
class TestListAllMissingPets:
    def test_applies_status_filter_when_given(self):
        """UTC-36-TC-01 — a supplied status is forwarded verbatim."""
        repo = _repo()
        repo.list_all.return_value = Page(
            [{"id": "pet-1", "status": "Searching"}], 1,
        )

        out = run(PetService.list_all_missing_pets(
            repo, limit=20, offset=0, status="Searching",
        ))

        assert out.items == [{"id": "pet-1", "status": "Searching", "sighting_count": 0, "post_status": "Pending"}]
        repo.list_all.assert_called_once_with("Searching", None, limit=10000, offset=0)

    def test_no_status_filter_when_none(self):
        """UTC-36-TC-02 — status=None reaches the repo as None (= no filter)."""
        repo = _repo()
        repo.list_all.return_value = Page([{"id": "pet-1"}, {"id": "pet-2"}], 2)

        out = run(PetService.list_all_missing_pets(repo, limit=20, offset=0))

        assert len(out.items) == 2
        repo.list_all.assert_called_once_with(None, None, 20, 0)

    def test_error_is_reraised(self):
        """UTC-36-TC-03 — DB failure propagates (API maps it to 500)."""
        repo = _repo()
        repo.list_all.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            run(PetService.list_all_missing_pets(repo))

    def test_empty_page_returns_empty_list(self):
        """UTC-36-TC-04 — an empty page is [], not None."""
        repo = _repo()
        repo.list_all.return_value = Page([], 0)
        out = run(PetService.list_all_missing_pets(repo, limit=20, offset=0))
        assert out.items == [] and out.total == 0

    def test_total_is_the_filter_count_not_the_page_length(self):
        """UTC-36-TC-05 — a full page of a larger result carries the real total.

        This is the whole point of the count: page 1 of 57 reports must say 57,
        because that is what tells the console pages 2 and 3 exist.
        """
        repo = _repo()
        repo.list_all.return_value = Page([{"id": f"pet-{i}"} for i in range(20)], 57)

        out = run(PetService.list_all_missing_pets(repo, limit=20, offset=0))

        assert len(out.items) == 20
        assert out.total == 57

    def test_a_stored_status_is_forwarded_with_the_window(self):
        """UTC-36-TC-06 - the database path. Found and Resolved are stored
        values, so the predicate and the window both go to the query and the
        total the database reports is already correct. This is the choice
        TC-01 was believed to frame until the audit of 07/09/2026 found that
        Searching takes the other path entirely."""
        repo = _repo()
        repo.list_all.return_value = Page([{"id": "pet-1", "status": "Found"}], 1)

        out = run(PetService.list_all_missing_pets(
            repo, limit=20, offset=0, status="Found",
        ))

        assert out.total == 1
        repo.list_all.assert_called_once_with("Found", None, 20, 0)

    def test_spotted_keeps_only_the_reports_that_have_a_sighting(self):
        """UTC-36-TC-07 [property DERIVED] - Spotted is not a stored value.
        Both Spotted and Searching sit in the database as status='Searching',
        and only the sighting count separates them, so the service reads the
        whole Searching bucket, counts, and filters. Nothing framed this
        branch before 07/09/2026."""
        repo = _repo()
        repo.list_all.return_value = Page([
            {"id": "p1", "status": "Searching"},
            {"id": "p2", "status": "Searching"},
        ], 2)
        repo.get_sighting_links_for_pets.return_value = [
            {"pet_id": "p2", "sighting_id": "s1", "owner_status": None},
        ]

        out = run(PetService.list_all_missing_pets(
            repo, limit=20, offset=0, status="Spotted",
        ))

        assert [p["id"] for p in out.items] == ["p2"]
        assert out.total == 1
        repo.list_all.assert_called_once_with(
            "Searching", None, limit=10_000, offset=0,
        )

    def test_searching_keeps_only_the_reports_that_have_none(self):
        """UTC-36-TC-08 [property DERIVED] - the other half of the same split.
        A console asking for Searching wants the reports nobody has answered
        yet, so a report carrying a sighting belongs to the Spotted bucket and
        must not appear in both."""
        repo = _repo()
        repo.list_all.return_value = Page([
            {"id": "p1", "status": "Searching"},
            {"id": "p2", "status": "Searching"},
        ], 2)
        repo.get_sighting_links_for_pets.return_value = [
            {"pet_id": "p2", "sighting_id": "s1", "owner_status": None},
        ]

        out = run(PetService.list_all_missing_pets(
            repo, limit=20, offset=0, status="Searching",
        ))

        assert [p["id"] for p in out.items] == ["p1"]
        assert out.total == 1

    def test_the_derived_page_is_sliced_after_the_filter(self):
        """UTC-36-TC-09 [if DERIVED] - the window cannot be applied by the
        query on this path, because the filter that decides membership runs
        after the rows come back. The total is therefore the depth of the
        filtered result and the page is cut from it here, which is what keeps
        the console's page arithmetic correct."""
        repo = _repo()
        repo.list_all.return_value = Page(
            [{"id": f"p{i}", "status": "Searching"} for i in range(5)], 5,
        )
        repo.get_sighting_links_for_pets.return_value = []

        out = run(PetService.list_all_missing_pets(
            repo, limit=2, offset=2, status="Searching",
        ))

        assert [p["id"] for p in out.items] == ["p2", "p3"]
        assert out.total == 5

    def test_the_species_filter_is_forwarded(self):
        """UTC-36-TC-10 - the second filter of this browse, unframed until
        07/09/2026. It is independent of the status filter, so a console
        narrowing to one species must not silently widen the status."""
        repo = _repo()
        repo.list_all.return_value = Page([{"id": "p1"}], 1)

        run(PetService.list_all_missing_pets(
            repo, limit=20, offset=0, species="Cat",
        ))

        repo.list_all.assert_called_once_with(None, "Cat", 20, 0)


# --------------------------------------------------------------------------- #
# UTC-37  remove_missing_pet (MD-42, SRS-70) — admin removal.
#
# UD-14's postcondition is "removed from the database and the search map", so
# the deletion is real; "no row deleted" is the not-found signal, which the
# service must turn into a ValueError rather than reporting a phantom success.
# --------------------------------------------------------------------------- #
class TestRemoveMissingPet:
    def test_removes_the_report(self):
        """UTC-37-TC-01 — the row is deleted and the deleted row returned."""
        repo = _repo()
        repo.remove.return_value = {"id": "p1", "pet_name": "Mochi"}

        out = run(PetService.remove_missing_pet(repo, "p1", "a1"))

        assert out == {"id": "p1", "pet_name": "Mochi"}
        repo.remove.assert_called_once_with("p1")

    def test_not_found_raises_valueerror(self):
        """UTC-37-TC-02 — nothing deleted => ValueError (API maps it to 404)."""
        repo = _repo()
        repo.remove.return_value = None
        with pytest.raises(ValueError):
            run(PetService.remove_missing_pet(repo, "ghost", "a1"))

    def test_error_is_reraised(self):
        """UTC-37-TC-03 — DB failure propagates (API maps it to 500)."""
        repo = _repo()
        repo.remove.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            run(PetService.remove_missing_pet(repo, "p1", "a1"))

    def test_removal_is_audit_logged_with_the_admin(self, caplog):
        """The moderation action is recorded — MD-42 says the removal is a
        recorded action, and the log line is where that record lives."""
        repo = _repo()
        repo.remove.return_value = {"id": "p1"}
        with caplog.at_level("WARNING"):
            run(PetService.remove_missing_pet(repo, "p1", "admin-7"))
        assert "admin-7" in caplog.text and "p1" in caplog.text
