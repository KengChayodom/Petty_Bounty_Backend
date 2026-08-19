"""
Route tests for the owner side of the sighting loop (2026-08-17).

Three things live only in the route layer and are invisible to a service test:

  1. `PATCH /missing-pets/{pet_id}/sightings/{sighting_id}` — which domain
     outcome becomes 400 vs 404. `LookupError` and `ValueError` are unrelated
     types here, but the order still matters if either is ever widened.
  2. Ending a search must also close that pet's sightings — and must NOT report
     failure when only that secondary step fails, because the pet is already
     marked Found by then.
  3. Reporting a sighting must schedule the owner push in the BACKGROUND, so
     the hunter's response is never held up by somebody else's notification —
     and must not schedule one when nothing matched.

Seams: the auth dependency and the Supabase client, swapped through FastAPI
dependency_overrides; the adapters the routes build inline are patched at the
module boundary.
"""
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import missing_pets as pets_api
from app.api import sightings as sightings_api
from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client


def _client(router_module, user_id="u1"):
    app = FastAPI()
    app.include_router(router_module.router)
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[get_supabase_client] = lambda: MagicMock()
    return TestClient(app)


def _async_returns(value):
    async def _inner(*a, **k):
        return value
    return _inner


def _async_raises(exc):
    async def _inner(*a, **k):
        raise exc
    return _inner


# --------------------------------------------------------------------------- #
# PATCH /missing-pets/{pet_id}/sightings/{sighting_id}
# --------------------------------------------------------------------------- #
class TestDecideMatchRoute:
    def _patch_service(self, monkeypatch, method):
        service = MagicMock()
        service.decide_match = method
        monkeypatch.setattr(
            pets_api, "SightingService", lambda **kwargs: service
        )
        return service

    def test_success(self, monkeypatch):
        self._patch_service(monkeypatch, _async_returns({
            "match": {"owner_status": "Confirmed"},
            "sighting_status_updated": True,
        }))
        r = _client(pets_api).patch(
            "/missing-pets/p1/sightings/s1", json={"decision": "Confirmed"},
        )
        assert r.status_code == 200
        assert r.json()["data"]["match"]["owner_status"] == "Confirmed"

    def test_not_owned_or_unknown_yields_404(self, monkeypatch):
        self._patch_service(
            monkeypatch, _async_raises(LookupError("not found or not owned")),
        )
        r = _client(pets_api).patch(
            "/missing-pets/p1/sightings/s1", json={"decision": "Confirmed"},
        )
        assert r.status_code == 404

    def test_bad_decision_yields_400(self, monkeypatch):
        self._patch_service(
            monkeypatch, _async_raises(ValueError("decision must be one of")),
        )
        r = _client(pets_api).patch(
            "/missing-pets/p1/sightings/s1", json={"decision": "Maybe"},
        )
        assert r.status_code == 400

    def test_failure_yields_500(self, monkeypatch):
        self._patch_service(monkeypatch, _async_raises(RuntimeError("db down")))
        r = _client(pets_api).patch(
            "/missing-pets/p1/sightings/s1", json={"decision": "Confirmed"},
        )
        assert r.status_code == 500

    def test_identity_comes_from_the_token_not_the_body(self, monkeypatch):
        """The owner id handed to the service must be the JWT's, so a client
        cannot rule on a pet by naming a different owner."""
        seen = {}

        async def _capture(pet_id, sighting_id, owner_id, decision):
            seen.update(
                pet_id=pet_id, sighting_id=sighting_id,
                owner_id=owner_id, decision=decision,
            )
            return {"match": {}, "sighting_status_updated": False}

        self._patch_service(monkeypatch, _capture)
        _client(pets_api, user_id="real-owner").patch(
            "/missing-pets/p1/sightings/s1",
            json={"decision": "Rejected", "owner_id": "somebody-else"},
        )
        assert seen == {
            "pet_id": "p1", "sighting_id": "s1",
            "owner_id": "real-owner", "decision": "Rejected",
        }


# --------------------------------------------------------------------------- #
# PATCH /missing-pets/{pet_id} — ending the search closes its sightings
# --------------------------------------------------------------------------- #
class TestEndSearchClosesSightings:
    def _repo(self, monkeypatch, updated={"id": "p1", "status": "Found"}):
        repo = MagicMock()
        repo.update_missing_pet_owned.return_value = updated
        repo.close_sightings_for_pet.return_value = 2
        monkeypatch.setattr(
            pets_api, "SupabaseMissingPetRepository", lambda db: repo
        )
        return repo

    def test_found_closes_the_pets_sightings(self, monkeypatch):
        repo = self._repo(monkeypatch)
        r = _client(pets_api).patch("/missing-pets/p1", json={"status": "Found"})
        assert r.status_code == 200
        repo.close_sightings_for_pet.assert_called_once_with("p1")

    def test_an_edit_that_is_not_a_closure_leaves_sightings_alone(
        self, monkeypatch
    ):
        """Renaming the pet or raising the bounty is not the end of a search."""
        repo = self._repo(monkeypatch, updated={"id": "p1", "pet_name": "Mochi"})
        r = _client(pets_api).patch(
            "/missing-pets/p1", json={"pet_name": "Mochi"},
        )
        assert r.status_code == 200
        repo.close_sightings_for_pet.assert_not_called()

    def test_still_searching_leaves_sightings_alone(self, monkeypatch):
        repo = self._repo(monkeypatch)
        _client(pets_api).patch("/missing-pets/p1", json={"status": "Searching"})
        repo.close_sightings_for_pet.assert_not_called()

    def test_a_pet_not_owned_closes_nothing(self, monkeypatch):
        """404 must short-circuit before the closure: otherwise a stranger
        could close a pet's sightings by PATCHing it."""
        repo = self._repo(monkeypatch, updated=None)
        r = _client(pets_api).patch("/missing-pets/p1", json={"status": "Found"})
        assert r.status_code == 404
        repo.close_sightings_for_pet.assert_not_called()

    def test_closure_failure_does_not_fail_the_request(self, monkeypatch):
        """The pet IS already marked Found. Returning 500 would tell the owner
        their closure failed when it did not."""
        repo = self._repo(monkeypatch)
        repo.close_sightings_for_pet.side_effect = RuntimeError("db down")

        r = _client(pets_api).patch("/missing-pets/p1", json={"status": "Found"})

        assert r.status_code == 200
        assert r.json()["data"]["status"] == "Found"


# --------------------------------------------------------------------------- #
# POST /sightings/ and /sightings/targeted — the owner push is scheduled, not awaited
# --------------------------------------------------------------------------- #
class TestOwnerPushIsScheduled:
    @pytest.fixture
    def wired(self, monkeypatch):
        """Capture what gets scheduled instead of running it: the point of the
        background task is that the hunter's response does not wait for it."""
        scheduled = []
        monkeypatch.setattr(
            sightings_api, "notify_pet_owners",
            lambda *a, **k: scheduled.append(a),
        )
        # NOTE: the service is swapped through dependency_overrides, not
        # monkeypatch — the route's Depends captured the original function
        # object, so replacing the module attribute would leave the real
        # service wired in and the override keyed to something unused.
        return scheduled, MagicMock()

    def _client_with(self, service):
        app = FastAPI()
        app.include_router(sightings_api.router)
        app.dependency_overrides[get_current_user_id] = lambda: "hunter-1"
        app.dependency_overrides[get_supabase_client] = lambda: MagicMock()
        app.dependency_overrides[
            sightings_api.get_sighting_service
        ] = lambda: service
        return TestClient(app)

    # `hunter_id` is required by the schema but the route overwrites it with
    # the JWT identity, so the value here is deliberately a decoy.
    _BODY = {
        "hunter_id": "spoofed-by-client",
        "image_url": "https://example.com/a.jpg",
        "latitude": 13.75, "longitude": 100.5,
        "detected_species": "Cat", "action_type": "Spotted",
    }

    def test_discovery_schedules_a_push_for_every_matched_pet(self, wired):
        scheduled, service = wired
        service.process_and_save_sighting = _async_returns({
            "sighting": {"id": "s1"},
            "matches": [{"id": "p1"}, {"id": "p2"}],
        })

        r = self._client_with(service).post("/sightings/", json=self._BODY)

        assert r.status_code == 200
        assert len(scheduled) == 1
        _db, sighting_id, pet_ids, hunter_id = scheduled[0]
        assert sighting_id == "s1"
        assert pet_ids == ["p1", "p2"]
        assert hunter_id == "hunter-1"

    def test_no_matches_schedules_nothing(self, wired):
        """Nobody's pet was recognised, so there is nobody to notify."""
        scheduled, service = wired
        service.process_and_save_sighting = _async_returns({
            "sighting": {"id": "s1"}, "matches": [],
        })

        self._client_with(service).post("/sightings/", json=self._BODY)

        assert scheduled == []

    def test_targeted_notifies_the_named_pets_owner(self, wired):
        """This endpoint has always answered "Targeted sighting sent to the
        owner." Until now nothing was sent; this is the push behind the claim."""
        scheduled, service = wired
        service.save_targeted_sighting = _async_returns({
            "sighting": {"id": "s9"}, "matches": [],
        })

        r = self._client_with(service).post(
            "/sightings/targeted", json={**self._BODY, "target_pet_id": "p7"},
        )

        assert r.status_code == 200
        _db, sighting_id, pet_ids, hunter_id = scheduled[0]
        assert (sighting_id, pet_ids, hunter_id) == ("s9", ["p7"], "hunter-1")
