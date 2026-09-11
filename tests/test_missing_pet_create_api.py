"""
Route tests for POST /missing-pets/ — the create half of the owner's report.

The service test (UTC-33, tests/test_pet_service.py::TestRegisterMissingPet)
verifies what `register_missing_pet` stores. It cannot verify SRS-61, because
the owner identity is bound in the route and the service only ever sees the
payload after the binding has happened. Nothing tested that binding on this
path until 2026-09-10: the one test that looked like it did,
`test_an_owner_id_in_the_body_is_ignored`, is in test_missing_pet_update_api.py
and belongs to the EDIT route (SRS-65/66).

Two things live only in this handler and are invisible to a service test:

  1. SRS-61 — `pet.owner_id` is overwritten from the verified JWT before the
     service is called, so an owner id in the request body is discarded.
  2. Which HTTP status each domain outcome becomes: a ValueError out of the
     service is the client's fault (400), anything else is the server's (500).

Seams: `get_current_user_id` and `get_supabase_client` are replaced through
dependency_overrides, and `PetService.register_missing_pet` is replaced with a
spy so the AI pipeline and the database are never reached.
"""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import missing_pets
from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client


JWT_OWNER = "jwt-owner-2222"

_BODY = {
    "pet_name": "Luna",
    "species": "Dog",
    "characteristics": {"color": "Golden"},
    "bounty_amount": 1000.0,
    "longitude": 100.5018,
    "latitude": 13.7563,
    "last_seen_time": "2025-01-12T10:30:00Z",
    "image_url": "https://example.com/luna.jpg",
}


def _env(monkeypatch, *, register=None):
    """A client whose route reaches a spy instead of the AI + database path."""
    seen = {}

    async def default_register(repo, pet):
        seen["pet"] = pet
        return {"id": "pet-123", "owner_id": pet.owner_id}

    monkeypatch.setattr(
        missing_pets.PetService, "register_missing_pet", register or default_register
    )
    # The fan-out is Feature 2's and is covered by TestMissingPetFanout; here it
    # only has to not run.
    monkeypatch.setattr(missing_pets, "notify_nearby_hunters", lambda *a, **k: None)

    app = FastAPI()
    app.include_router(missing_pets.router)
    app.dependency_overrides[get_current_user_id] = lambda: JWT_OWNER
    app.dependency_overrides[get_supabase_client] = lambda: object()
    return SimpleNamespace(client=TestClient(app), seen=seen)


class TestCreateMissingPetRoute:
    def test_the_owner_is_taken_from_the_token(self, monkeypatch):
        """UTC-74-TC-01 — SRS-61, the ordinary request."""
        env = _env(monkeypatch)

        r = env.client.post("/missing-pets/", json=_BODY)

        assert r.status_code == 200
        assert env.seen["pet"].owner_id == JWT_OWNER

    def test_an_owner_id_in_the_body_is_discarded(self, monkeypatch):
        """UTC-74-TC-02 — SRS-61, the attack this requirement exists for.

        A client that posts somebody else's identifier must not be able to
        file a report under it. The assertion is on the value the service was
        handed, not on the response, because the response would look the same
        either way.
        """
        env = _env(monkeypatch)

        r = env.client.post(
            "/missing-pets/", json={**_BODY, "owner_id": "somebody-else"}
        )

        assert r.status_code == 200
        assert env.seen["pet"].owner_id == JWT_OWNER

    def test_the_stored_report_is_returned_in_the_envelope(self, monkeypatch):
        """UTC-74-TC-03 — the row the service returned reaches the caller."""
        env = _env(monkeypatch)

        r = env.client.post("/missing-pets/", json=_BODY)

        body = r.json()
        assert body["status"] == "success"
        assert body["data"]["id"] == "pet-123"

    def test_a_refused_report_is_the_callers_fault(self, monkeypatch):
        """UTC-74-TC-04 — a ValueError out of the service is 400, not 500."""
        async def refusing(repo, pet):
            raise ValueError("Invalid image URL")

        env = _env(monkeypatch, register=refusing)

        r = env.client.post("/missing-pets/", json=_BODY)

        assert r.status_code == 400
        assert "Invalid image URL" in r.json()["detail"]

    def test_an_unexpected_failure_is_the_servers_fault(self, monkeypatch):
        """UTC-74-TC-05 — anything that is not a ValueError is 500."""
        async def exploding(repo, pet):
            raise RuntimeError("DB connection lost")

        env = _env(monkeypatch, register=exploding)

        r = env.client.post("/missing-pets/", json=_BODY)

        assert r.status_code == 500

    @pytest.mark.parametrize(
        "bad",
        [
            {"species": "dragon"},
            {"characteristics": {}},
            {"primary_color_hex": "red"},
        ],
    )
    def test_a_payload_the_schema_refuses_never_reaches_the_service(
        self, monkeypatch, bad
    ):
        """UTC-74-TC-06 — the schema refuses first, so the route writes nothing.

        This is the seam between the schema and MD-79: the validators of UTC-57
        to UTC-59
        decide the value, and what this case pins is that a refusal there is a
        422 and the service is never called at all.
        """
        env = _env(monkeypatch)

        r = env.client.post("/missing-pets/", json={**_BODY, **bad})

        assert r.status_code == 422
        assert "pet" not in env.seen
