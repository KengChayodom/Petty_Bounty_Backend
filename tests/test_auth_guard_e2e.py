"""
End-to-end auth-guard wiring test.

Progress-I SRS traceability: SRS-44 + SRS-13 (the JWT is extracted & validated to
authorize the request — a valid bearer token resolves to the user and the route
runs, honouring the authenticated session; a missing/invalid token is rejected
with 401). The login UI itself (SRS-11/12/14) is exercised manually.

test_auth.py proves get_current_user_id's logic in isolation, but nothing proved
the guard is actually WIRED onto a real endpoint. This hits a genuine protected
route (POST /me/location) WITHOUT overriding get_current_user_id, so the real
dependency runs:
  * no Authorization header  -> 401 (the guard rejects).
  * a valid bearer token     -> 200 (the guard resolves and the route runs,
                                scoped to the resolved user id).

Only the boundaries are faked: the Supabase client the guard calls to validate
the token (auth.get_supabase_client) and the DB client the route writes through
(get_supabase_client dependency). The guard itself is the real code.

AUTH_DEV_BYPASS is pinned False so the 401 can't be masked by the dev bypass
inheriting the ambient .env.
"""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import me
from app.core import auth
from app.core.database import get_supabase_client

VALID_USER = "real-user-42"


class _FakeAuthClient:
    """Stands in for the Supabase client the GUARD calls to validate a token.

    Returns VALID_USER for the token 'good', else no user (invalid token).
    """
    @property
    def auth(self):
        def get_user(token):
            if token == "good":
                return SimpleNamespace(user=SimpleNamespace(id=VALID_USER))
            return SimpleNamespace(user=None)
        return SimpleNamespace(get_user=get_user)


@pytest.fixture
def client(monkeypatch, fake_db):
    # The guard must reject, not fall back to a dev user.
    monkeypatch.setattr(auth.settings, "AUTH_DEV_BYPASS", False)
    # Boundary the REAL guard calls internally (not a dependency override).
    monkeypatch.setattr(auth, "get_supabase_client", lambda: _FakeAuthClient())
    # The route body's DB write succeeds for the JWT user.
    fake_db.set_table_result("users", "update", data=[{"id": VALID_USER}])

    app = FastAPI()
    app.include_router(me.router)
    # NOTE: get_current_user_id is deliberately NOT overridden — that's the point.
    app.dependency_overrides[get_supabase_client] = lambda: fake_db
    return TestClient(app)


def test_protected_route_401_without_token(client):
    r = client.post("/me/location", json={"latitude": 1.0, "longitude": 2.0})

    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate") == "Bearer"


def test_protected_route_200_with_valid_token(client, fake_db):
    r = client.post(
        "/me/location",
        json={"latitude": 1.0, "longitude": 2.0},
        headers={"Authorization": "Bearer good"},
    )

    assert r.status_code == 200
    assert r.json()["status"] == "success"
    # Proof the REAL guard resolved the token and handed the id to the route:
    # the update is scoped to the resolved user, not a hardcoded dev id.
    assert ("id", VALID_USER) in fake_db.filters_for("users", "update")
