"""
Every route is behind a login unless it is on a short, written-down list.

Two sighting reads and the pet detail page were reachable by the wrong people
because a route was declared without its auth dependency and nothing noticed.
The list of routes that are meant to be open is short, so it is written here.
Anything else, including a route added next month, must answer 401 without a
token, or this file fails.

Three layers, because each catches something the others cannot:
  * the route table is walked, so a route with no guard is named;
  * every guarded route is actually called with no token, so a guard that is
    declared but does not run is caught;
  * each router declares its guard once for all its routes, so a new route
    added to it is protected without anyone remembering to.
"""
from unittest.mock import MagicMock

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.api import (
    admin, auth, devices, hunters, leaderboard, me, missing_pets, reports,
    sightings, upload,
)
from app.core.auth import get_current_user_id, require_admin
from app.core.config import settings
from app.core.database import get_supabase_client
from app.main import app

# The only routes meant to be reachable without a token.
PUBLIC = {
    ("GET", "/"),                        # "the API is running"
    ("GET", "/health"),                  # the load balancer's probe
    ("GET", "/missing-pets/nearby"),     # the map; returns no contact details
}


def _dependency_calls(dependant, found=None):
    found = set() if found is None else found
    for sub in dependant.dependencies:
        found.add(sub.call)
        _dependency_calls(sub, found)
    return found


def _api_routes():
    for route in app.routes:
        if isinstance(route, APIRoute):
            for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
                yield method, route


def _is_dev_auth(path):
    # Mounted only when ENABLE_DEV_AUTH is set, for minting a token from /docs.
    return path.startswith("/dev/")


class TestTheRouteTable:
    def test_every_route_outside_the_public_list_has_a_guard(self):
        unguarded = []
        for method, route in _api_routes():
            if (method, route.path) in PUBLIC:
                continue
            if _is_dev_auth(route.path) and settings.ENABLE_DEV_AUTH:
                continue
            calls = _dependency_calls(route.dependant)
            if not ({get_current_user_id, require_admin} & calls):
                unguarded.append(f"{method} {route.path}")
        assert unguarded == [], f"routes with no login guard: {unguarded}"

    def test_the_public_list_only_names_routes_that_exist(self):
        """A stale entry would quietly leave a hole for a future route that
        happens to reuse the path."""
        existing = {(m, r.path) for m, r in _api_routes()}
        assert PUBLIC <= existing, PUBLIC - existing

    def test_every_admin_route_requires_an_administrator(self):
        """A plain login is not enough under /admin."""
        weak = [
            f"{m} {r.path}" for m, r in _api_routes()
            if r.path.startswith("/admin")
            and require_admin not in _dependency_calls(r.dependant)
        ]
        assert weak == []

    def test_the_dev_login_routes_are_absent_unless_switched_on(self):
        mounted = any(_is_dev_auth(r.path) for _, r in _api_routes())
        assert mounted == settings.ENABLE_DEV_AUTH


@pytest.fixture
def anonymous(monkeypatch):
    """A client with no token, and with every dev shortcut off, so a route that
    lets the call through would fail here rather than reach a database."""
    monkeypatch.setattr(settings, "AUTH_DEV_BYPASS", False)
    monkeypatch.setattr(settings, "ENABLE_UNAUTHED_ADMIN", False)
    app.dependency_overrides[get_supabase_client] = lambda: MagicMock()
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.pop(get_supabase_client, None)


class TestNoTokenIsRefusedEverywhere:
    @pytest.mark.parametrize(
        "method, path",
        sorted(
            (m, r.path) for m, r in _api_routes()
            if (m, r.path) not in PUBLIC and not _is_dev_auth(r.path)
        ),
    )
    def test_a_call_without_a_token_is_a_401(self, anonymous, method, path):
        url = path.replace("{", "").replace("}", "") or "/"
        r = anonymous.request(method, url)
        assert r.status_code == 401, f"{method} {path} answered {r.status_code}"


PROTECTED = [
    pytest.param(auth, get_current_user_id, id="auth"),
    pytest.param(devices, get_current_user_id, id="devices"),
    pytest.param(hunters, get_current_user_id, id="hunters"),
    pytest.param(leaderboard, get_current_user_id, id="leaderboard"),
    pytest.param(me, get_current_user_id, id="me"),
    pytest.param(reports, get_current_user_id, id="reports"),
    pytest.param(sightings, get_current_user_id, id="sightings"),
    pytest.param(upload, get_current_user_id, id="upload"),
    pytest.param(missing_pets, get_current_user_id, id="missing_pets"),
    pytest.param(admin, require_admin, id="admin"),
]


class TestEachRouterDeclaresItsGuardOnce:
    @pytest.mark.parametrize("module, guard", PROTECTED)
    def test_the_guard_is_on_the_router_so_a_new_route_inherits_it(
        self, module, guard
    ):
        declared = [d.dependency for d in module.router.dependencies]
        assert guard in declared


class TestTheGuardCostsNothingExtra:
    def test_the_token_is_checked_once_per_request_not_once_per_declaration(
        self, monkeypatch
    ):
        """The guard is declared on the router AND asked for by the route that
        wants the id. FastAPI resolves a dependency once per request, so this
        must not double the round trip to the auth service."""
        from app.core import auth as auth_module

        calls = []
        monkeypatch.setattr(
            auth_module, "_validate_token",
            lambda token: calls.append(token) or "u-1",
        )
        app.dependency_overrides[get_supabase_client] = lambda: MagicMock()
        try:
            TestClient(app, raise_server_exceptions=False).get(
                "/leaderboard/users", headers={"Authorization": "Bearer abc"},
            )
        finally:
            app.dependency_overrides.pop(get_supabase_client, None)

        assert calls == ["abc"]
