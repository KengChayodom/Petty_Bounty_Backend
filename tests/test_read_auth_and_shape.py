"""
Route tests for the reads that were hardened on 2026-09-09.

Three defects, all of them invisible to a service test because all three live in
the route signature or its exception mapping:

  1. `GET /missing-pets/{pet_id}/sightings` took the caller's identity from the
     token and never used it, so any signed-in account could read any owner's
     sighting timeline — rows that carry where a pet was seen plus the hunter's
     display name and telephone number.
  2. `GET /sightings/{sighting_id}` and `GET /sightings/{sighting_id}/matches`
     had no authentication dependency at all. They were the only sighting reads
     reachable anonymously, found by walking `app/api/` with ast rather than by
     reading the routes one at a time.
  3. `/matches` returned a bare dict rather than StandardResponse, and mapped
     ValueError to 404, telling a caller their sighting did not exist when the
     real answer was that matches could not be served for it.

Seams: the auth dependency and the Supabase client through dependency_overrides;
the adapter the missing-pets route builds inline is patched at the module
boundary, the same way test_owner_loop_api.py does it.
"""
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import missing_pets as pets_api
from app.api import sightings as sightings_api
from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client

TIMELINE = [{"id": "s1", "sighted_location": "POINT(100 13)"}]


def _app(module, user_id="owner-1"):
    app = FastAPI()
    app.include_router(module.router)
    app.dependency_overrides[get_supabase_client] = lambda: MagicMock()
    if user_id is not None:
        app.dependency_overrides[get_current_user_id] = lambda: user_id
    return app


class TestTheOwnerTimelineIsScopedToItsOwner:
    def _repo(self, monkeypatch, owner):
        repo = MagicMock()
        repo.get_missing_pet_by_id.return_value = (
            None if owner is None else {"id": "p1", "owner_id": owner}
        )
        repo.sightings_for_pet.return_value = TIMELINE
        monkeypatch.setattr(
            pets_api, "SupabaseMissingPetRepository", lambda db: repo
        )
        return repo

    def test_the_owner_reads_their_own_timeline(self, monkeypatch):
        repo = self._repo(monkeypatch, "owner-1")
        r = TestClient(_app(pets_api)).get("/missing-pets/p1/sightings")
        assert r.status_code == 200
        assert r.json()["data"] == TIMELINE
        repo.sightings_for_pet.assert_called_once()

    @pytest.mark.parametrize("owner", ["somebody-else", None])
    def test_anybody_else_gets_a_missing_report_and_reads_nothing(
        self, monkeypatch, owner
    ):
        """404 rather than 403, and the same answer whether the report belongs
        to another account or does not exist, so the endpoint cannot be used to
        discover which identifiers are real. The timeline query must not run at
        all: answering after reading it would leak the row count through timing
        and through the log line the read emits."""
        repo = self._repo(monkeypatch, owner)
        r = TestClient(_app(pets_api)).get("/missing-pets/p1/sightings")
        assert r.status_code == 404
        repo.sightings_for_pet.assert_not_called()

    def test_the_read_is_still_behind_authentication(self, monkeypatch):
        self._repo(monkeypatch, "owner-1")
        client = TestClient(_app(pets_api, user_id=None),
                            raise_server_exceptions=False)
        assert client.get("/missing-pets/p1/sightings").status_code == 401


class TestTheSightingReadsRequireACaller:
    """Both were anonymous until 2026-09-09."""

    @pytest.mark.parametrize("path", [
        "/sightings/s1",
        "/sightings/s1/matches",
    ])
    def test_an_unauthenticated_read_is_refused(self, path):
        client = TestClient(_app(sightings_api, user_id=None),
                            raise_server_exceptions=False)
        assert client.get(path).status_code == 401


class TestTheMatchesResponse:
    def _service(self, app, matches=None, raises=None):
        svc = MagicMock()

        async def _get(*a, **k):
            if raises is not None:
                raise raises
            return matches

        svc.get_matches = _get
        app.dependency_overrides[sightings_api.get_sighting_service] = (
            lambda: svc
        )
        return svc

    def test_the_page_is_wrapped_like_every_other_response(self):
        """It was the one endpoint returning a bare dict, so a client could not
        read it with the same code as the rest of the API."""
        app = _app(sightings_api)
        self._service(app, matches=[{"pet_id": "p1"}])
        r = TestClient(app).get("/sightings/s1/matches")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["data"]["matches"] == [{"pet_id": "p1"}]

    def test_a_request_that_cannot_be_served_is_a_bad_request(self):
        """ValueError here means matches cannot be produced for the sighting,
        not that the sighting is absent. Reporting it as 404 named the wrong
        thing as missing, and a client cannot tell the two apart."""
        app = _app(sightings_api)
        self._service(app, raises=ValueError("no feature vector"))
        assert TestClient(app).get("/sightings/s1/matches").status_code == 400

    def test_a_failure_is_still_a_failure(self):
        app = _app(sightings_api)
        self._service(app, raises=RuntimeError("db down"))
        client = TestClient(app, raise_server_exceptions=False)
        assert client.get("/sightings/s1/matches").status_code == 500
