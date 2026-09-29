"""
`GET /missing-pets/{id}` and the owner's contact details.

The detail RPC joins the owner's username, telephone number and profile photo
onto the report. The route took no token, so anyone could list pet ids from the
public `/nearby` (which carries no contact data) and then read a telephone
number for each one without an account.

The route now requires a signed-in caller. The app already sends every signed-out
user to the login screen (see `goRouterProvider`), so this costs it nothing.
`/nearby` stays open: it returns no contact details.
"""
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import missing_pets as pets_api
from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client

PET = {
    "id": "p1", "owner_id": "owner-1", "pet_name": "Mochi", "species": "Cat",
    "status": "Searching", "expires_at": None,
    "owner_username": "somchai", "owner_phone": "0812345678",
    "owner_profile_image_url": "https://x/p.png",
}


def _client(monkeypatch, caller, pet=PET):
    repo = MagicMock()
    repo.get_missing_pet_by_id.return_value = None if pet is None else dict(pet)
    repo.get_sighting_links_for_pets.return_value = []
    monkeypatch.setattr(pets_api, "SupabaseMissingPetRepository", lambda db: repo)
    app = FastAPI()
    app.include_router(pets_api.router)
    app.dependency_overrides[get_supabase_client] = lambda: MagicMock()
    if caller is not None:
        app.dependency_overrides[get_current_user_id] = lambda: caller
    return TestClient(app, raise_server_exceptions=False), repo


class TestThePetPageRequiresASignedInCaller:
    def test_no_token_is_a_401_and_reads_nothing(self, monkeypatch):
        client, repo = _client(monkeypatch, caller=None)
        assert client.get("/missing-pets/p1").status_code == 401
        repo.get_missing_pet_by_id.assert_not_called()

    def test_a_signed_in_caller_gets_the_pet_with_the_owner_contact(
        self, monkeypatch
    ):
        client, _ = _client(monkeypatch, caller="hunter-1")
        r = client.get("/missing-pets/p1")
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["pet_name"] == "Mochi"
        assert data["owner_phone"] == "0812345678"

    def test_a_missing_pet_is_a_404(self, monkeypatch):
        client, _ = _client(monkeypatch, caller="hunter-1", pet=None)
        assert client.get("/missing-pets/p1").status_code == 404
