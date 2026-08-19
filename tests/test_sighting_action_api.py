"""
Route tests for PATCH /sightings/{sighting_id}/action (2026-08-19).

The final-review screen — the step AFTER "Confirm Match" — collects one
choice, JUST SPOTTED or RESCUE, and this endpoint persists it to
`sightings.action_type`.

Two things live only in the route layer and are invisible to a service test:

  1. Which domain outcome becomes which status. `SightingActionLocked`
     SUBCLASSES ValueError, so the 409 clause must be written BEFORE the
     generic 400 — get that order wrong and "already reviewed" silently
     degrades into a generic bad-request. Pinned here.
  2. The hunter identity handed to the service is the JWT's, never anything
     the client can put in the path or body.

Seams: the auth dependency and the Supabase client, swapped through FastAPI
dependency_overrides; the service is patched at the route's factory boundary.
"""
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import sightings as sightings_api
from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client
from app.repositories.sighting_repository import SightingActionLocked


def _client(service, user_id="hunter-1"):
    app = FastAPI()
    app.include_router(sightings_api.router)
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[get_supabase_client] = lambda: MagicMock()
    app.dependency_overrides[sightings_api.get_sighting_service] = lambda: service
    return TestClient(app)


def _service(method):
    service = MagicMock()
    service.confirm_sighting_action = method
    return service


def _async_returns(value):
    async def _inner(*a, **k):
        return value
    return _inner


def _async_raises(exc):
    async def _inner(*a, **k):
        raise exc
    return _inner


class TestConfirmSightingActionRoute:
    def test_success_reports_the_stored_value(self):
        service = _service(_async_returns({
            "sighting": {"id": "s1", "action_type": "Caught"},
            "action_type": "Caught",
            "changed": True,
        }))

        r = _client(service).patch(
            "/sightings/s1/action", json={"action_type": "Rescue"},
        )

        assert r.status_code == 200
        body = r.json()
        assert body["data"]["action_type"] == "Caught"
        assert body["message"] == "Report status set to Caught."

    def test_not_yours_or_unknown_yields_404(self):
        service = _service(_async_raises(LookupError("not found or not yours")))
        r = _client(service).patch(
            "/sightings/s1/action", json={"action_type": "Caught"},
        )
        assert r.status_code == 404

    def test_already_reviewed_yields_409_not_400(self):
        """SightingActionLocked subclasses ValueError. If the except clauses
        are ever reordered, this test fails with 400 — which is the whole
        reason it exists: the client needs to tell "too late" apart from
        "you sent nonsense"."""
        service = _service(_async_raises(SightingActionLocked("s1", "Verified")))

        r = _client(service).patch(
            "/sightings/s1/action", json={"action_type": "Caught"},
        )

        assert r.status_code == 409
        assert "Verified" in r.json()["detail"]

    def test_unknown_action_yields_400(self):
        service = _service(_async_raises(ValueError("action_type must be one of")))
        r = _client(service).patch(
            "/sightings/s1/action", json={"action_type": "Maybe"},
        )
        assert r.status_code == 400

    def test_failure_yields_500(self):
        service = _service(_async_raises(RuntimeError("db down")))
        r = _client(service).patch(
            "/sightings/s1/action", json={"action_type": "Caught"},
        )
        assert r.status_code == 500

    def test_missing_action_type_is_a_422(self):
        service = _service(_async_returns({}))
        r = _client(service).patch("/sightings/s1/action", json={})
        assert r.status_code == 422

    def test_identity_comes_from_the_token_not_the_body(self):
        """A hunter must not be able to confirm somebody else's report by
        naming a different hunter_id."""
        seen = {}

        async def _capture(sighting_id, hunter_id, action_type):
            seen.update(
                sighting_id=sighting_id, hunter_id=hunter_id,
                action_type=action_type,
            )
            return {"sighting": {}, "action_type": "Caught", "changed": True}

        _client(_service(_capture), user_id="real-hunter").patch(
            "/sightings/s1/action",
            json={"action_type": "Caught", "hunter_id": "somebody-else"},
        )

        assert seen == {
            "sighting_id": "s1", "hunter_id": "real-hunter",
            "action_type": "Caught",
        }
