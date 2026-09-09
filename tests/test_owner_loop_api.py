"""
Route tests for the owner side of the sighting loop (2026-08-17).

The report closure that used to sit here as TestEndSearchClosesSightings
moved to tests/test_missing_pet_update_api.py on 2026-09-07. It is one
handler and therefore one UTC-35 category-partition table, and three of
the five tests here repeated frames that file already carried.

Two things live only in the route layer and are invisible to a service test:

  1. `PATCH /missing-pets/{pet_id}/sightings/{sighting_id}` — which domain
     outcome becomes 400 vs 404 vs 409. Handler ORDER is load-bearing here:
     the queue refusals (already decided / out of order / search closed)
     subclass ValueError, so a generic `except ValueError` placed first would
     answer 400 and tell the owner their perfectly ordinary request was
     malformed.
  2. Reporting a sighting must schedule the owner push in the BACKGROUND, so
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
from app.repositories.sighting_repository import (
    SearchAlreadyClosed,
    SightingAlreadyDecided,
    SightingOutOfOrder,
)
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
