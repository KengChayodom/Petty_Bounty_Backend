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
the token (auth.get_supabase_client) and the UserRepository the route writes
through (get_user_repository, doubled with MagicMock(spec=...)). The guard
itself is the real code.

AUTH_DEV_BYPASS is pinned False so the 401 can't be masked by the dev bypass
inheriting the ambient .env.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import me
from app.api.me import get_user_repository
from app.core import auth
from app.repositories.user_repository import UserRepository

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
def repo():
    r = MagicMock(spec=UserRepository)
    r.update_last_location.return_value = {"id": VALID_USER}  # write succeeds
    return r


@pytest.fixture
def client(monkeypatch, repo):
    # The guard must reject, not fall back to a dev user.
    monkeypatch.setattr(auth.settings, "AUTH_DEV_BYPASS", False)
    # Boundary the REAL guard calls internally (not a dependency override).
    monkeypatch.setattr(auth, "get_supabase_client", lambda: _FakeAuthClient())

    app = FastAPI()
    app.include_router(me.router)
    # NOTE: get_current_user_id is deliberately NOT overridden — that's the point.
    app.dependency_overrides[get_user_repository] = lambda: repo
    return TestClient(app)


def test_protected_route_401_without_token(client):
    r = client.post("/me/location", json={"latitude": 1.0, "longitude": 2.0})

    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate") == "Bearer"


def test_protected_route_200_with_valid_token(client, repo):
    r = client.post(
        "/me/location",
        json={"latitude": 1.0, "longitude": 2.0},
        headers={"Authorization": "Bearer good"},
    )

    assert r.status_code == 200
    assert r.json()["status"] == "success"
    # Proof the REAL guard resolved the token and handed the id to the route:
    # the update is scoped to the resolved user, not a hardcoded dev id.
    repo.update_last_location.assert_called_once()
    assert repo.update_last_location.call_args.args[0] == VALID_USER
