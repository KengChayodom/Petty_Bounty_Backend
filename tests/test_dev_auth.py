"""
Unit tests for app/api/dev_auth.py — the DEV-ONLY GoTrue proxy (/dev/login,
/dev/register).

Why this is tested: it is an auth-adjacent surface that is *off by default* and
must stay that way. Two classes of regression would be expensive and silent:
  1. A credential leak — passwords or minted tokens written to logs.
  2. A gating failure — the routes existing when ENABLE_DEV_AUTH is false (prod
     exposure) or missing when it's true (broken dev tool).
Both are covered below in addition to the ordinary request/response behaviour.

Boundary rule (per the automated-testing skill): the only thing mocked is the
*boundary* — the outbound httpx call to Supabase GoTrue. respx isn't installed,
so we substitute a small fake AsyncClient that records the exact request and
returns a programmable response. We assert on the route's returned
status/body/sent-payload, never on "a mock was called".

Each test owns its data: the `client` fixture pins SUPABASE_URL/ANON_KEY via
monkeypatch so tests never depend on the ambient .env, and the gating tests
reload main.py under an explicitly monkeypatched flag.
"""
import importlib
import logging

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api import dev_auth


# --------------------------------------------------------------------------- #
# Boundary fake: the httpx surface dev_auth actually touches
#   async with httpx.AsyncClient(...) as c: resp = await c.post(url, headers=, json=)
# --------------------------------------------------------------------------- #
_UNSET = object()


class _FakeResponse:
    def __init__(self, status_code, json_data=_UNSET, json_exc=None):
        self.status_code = status_code
        self._json_data = json_data
        self._json_exc = json_exc

    def json(self):
        if self._json_exc is not None:
            raise self._json_exc
        if self._json_data is _UNSET:
            raise ValueError("no json")
        return self._json_data


@pytest.fixture
def gotrue(monkeypatch):
    """Install a fake httpx.AsyncClient; expose control + the captured request.

    Set `state["resp"]` to a _FakeResponse, or `state["exc"]` to an exception to
    simulate a transport failure. `state["calls"]` records each POST's
    url/headers/json so tests can assert what was actually sent to GoTrue.
    """
    state = {"resp": None, "exc": None, "calls": []}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None):
            state["calls"].append({"url": url, "headers": headers, "json": json})
            if state["exc"] is not None:
                raise state["exc"]
            return state["resp"]

    monkeypatch.setattr(dev_auth.httpx, "AsyncClient", _FakeAsyncClient)
    return state


@pytest.fixture
def client(monkeypatch):
    """A TestClient over a minimal app that mounts ONLY the dev_auth router,
    with the Supabase settings the router reads pinned to known values."""
    monkeypatch.setattr(dev_auth.settings, "SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setattr(dev_auth.settings, "SUPABASE_ANON_KEY", "anon-key-xyz")
    app = FastAPI()
    app.include_router(dev_auth.router)
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
class TestGotrueHeaders:
    def test_includes_anon_key_and_json_content_type(self, monkeypatch):
        monkeypatch.setattr(dev_auth.settings, "SUPABASE_ANON_KEY", "anon-123")
        headers = dev_auth._gotrue_headers()
        # Guards: GoTrue rejects calls without an apikey; the app uses the anon
        # key, and this proxy must match it (not the service key).
        assert headers["apikey"] == "anon-123"
        assert headers["Content-Type"] == "application/json"

    def test_missing_anon_key_raises_500(self, monkeypatch):
        # Guards the request-time guard: an unset anon key must be a clean 500
        # ("not set"), not a None apikey that GoTrue bounces as a vague 401.
        monkeypatch.setattr(dev_auth.settings, "SUPABASE_ANON_KEY", "")
        with pytest.raises(HTTPException) as exc:
            dev_auth._gotrue_headers()
        assert exc.value.status_code == 500
        assert "SUPABASE_ANON_KEY" in exc.value.detail


class TestSafeError:
    @pytest.mark.parametrize(
        "body, expected",
        [
            ({"error_description": "ed", "msg": "m"}, "ed"),   # highest priority
            ({"msg": "m", "message": "x"}, "m"),
            ({"message": "x", "error": "e"}, "x"),
            ({"error": "e"}, "e"),
            ({"unrelated": "z"}, "Supabase auth error"),       # none present
        ],
    )
    def test_extracts_message_in_priority_order(self, body, expected):
        # Guards: the passthrough must surface GoTrue's human message without
        # leaking the whole body, and degrade cleanly when keys are absent.
        assert dev_auth._safe_error(_FakeResponse(400, json_data=body)) == expected

    def test_non_json_body_falls_back_to_status(self):
        # A non-JSON error body (e.g. an HTML 502 from a proxy) must not crash
        # the handler; it degrades to a status-coded message.
        resp = _FakeResponse(503, json_exc=ValueError("not json"))
        assert dev_auth._safe_error(resp) == "Supabase auth error (503)"


# --------------------------------------------------------------------------- #
# /dev/login
# --------------------------------------------------------------------------- #
class TestDevLogin:
    def test_success_passes_token_body_through(self, client, gotrue):
        token_body = {
            "access_token": "jwt.aaa.bbb",
            "refresh_token": "ref.ccc",
            "token_type": "bearer",
            "expires_in": 3600,
            "user": {"id": "u-1"},
        }
        gotrue["resp"] = _FakeResponse(200, json_data=token_body)

        r = client.post("/dev/login", json={"email": "a@b.co", "password": "pw"})

        assert r.status_code == 200
        # The whole point is returning the token for the caller to copy.
        assert r.json() == token_body

    def test_sends_password_grant_with_anon_key(self, client, gotrue):
        gotrue["resp"] = _FakeResponse(200, json_data={"access_token": "t"})

        client.post("/dev/login", json={"email": "a@b.co", "password": "pw"})

        sent = gotrue["calls"][0]
        # Guards: wrong endpoint or grant type → no token is ever minted.
        assert sent["url"] == (
            "https://proj.supabase.co/auth/v1/token?grant_type=password"
        )
        assert sent["json"] == {"email": "a@b.co", "password": "pw"}
        assert sent["headers"]["apikey"] == "anon-key-xyz"

    def test_gotrue_failure_passes_status_and_message_through(self, client, gotrue):
        gotrue["resp"] = _FakeResponse(
            400, json_data={"error_description": "Invalid login credentials"}
        )

        r = client.post("/dev/login", json={"email": "a@b.co", "password": "wrong"})

        # Guards: a real 400 must not be masked as a 500, and the user-facing
        # GoTrue message must survive.
        assert r.status_code == 400
        assert r.json()["detail"] == "Invalid login credentials"

    def test_network_error_becomes_502(self, client, gotrue):
        gotrue["exc"] = httpx.ConnectError("supabase down")

        r = client.post("/dev/login", json={"email": "a@b.co", "password": "pw"})

        # Guards: a Supabase outage must be a clean 502, not an unhandled 500.
        assert r.status_code == 502

    def test_missing_anon_key_returns_500(self, client, gotrue, monkeypatch):
        monkeypatch.setattr(dev_auth.settings, "SUPABASE_ANON_KEY", "")
        gotrue["resp"] = _FakeResponse(200, json_data={"access_token": "t"})

        r = client.post("/dev/login", json={"email": "a@b.co", "password": "pw"})

        assert r.status_code == 500
        # And it must short-circuit before ever reaching GoTrue.
        assert gotrue["calls"] == []


# --------------------------------------------------------------------------- #
# /dev/register
# --------------------------------------------------------------------------- #
class TestDevRegister:
    @pytest.mark.parametrize("status_code", [200, 201])
    def test_signup_success_for_both_2xx_codes(self, client, gotrue, status_code):
        # GoTrue returns 200 (session) or 200/201 (confirm-email) — the handler's
        # `status in (200, 201)` boundary must accept both as success.
        body = {"user": {"id": "u-9"}}
        gotrue["resp"] = _FakeResponse(status_code, json_data=body)

        r = client.post(
            "/dev/register", json={"email": "a@b.co", "password": "longpw"}
        )

        assert r.status_code == 200
        assert r.json() == body

    def test_display_name_is_sent_as_user_metadata(self, client, gotrue):
        gotrue["resp"] = _FakeResponse(200, json_data={"user": {"id": "u"}})

        client.post(
            "/dev/register",
            json={"email": "a@b.co", "password": "longpw", "display_name": "Ken"},
        )

        sent = gotrue["calls"][0]
        # Guards the handle_new_user trigger contract: display_name must land in
        # `data` (→ user_metadata), not be dropped.
        assert sent["url"] == "https://proj.supabase.co/auth/v1/signup"
        assert sent["json"]["data"] == {"display_name": "Ken"}

    def test_no_display_name_omits_data_key(self, client, gotrue):
        gotrue["resp"] = _FakeResponse(200, json_data={"user": {"id": "u"}})

        client.post(
            "/dev/register", json={"email": "a@b.co", "password": "longpw"}
        )

        # Guards: don't send an empty/garbage metadata object when unspecified.
        assert "data" not in gotrue["calls"][0]["json"]

    def test_gotrue_failure_passes_status_through(self, client, gotrue):
        gotrue["resp"] = _FakeResponse(
            422, json_data={"msg": "User already registered"}
        )

        r = client.post(
            "/dev/register", json={"email": "a@b.co", "password": "longpw"}
        )

        assert r.status_code == 422
        assert r.json()["detail"] == "User already registered"

    def test_network_error_becomes_502(self, client, gotrue):
        gotrue["exc"] = httpx.ConnectError("down")

        r = client.post(
            "/dev/register", json={"email": "a@b.co", "password": "longpw"}
        )

        assert r.status_code == 502

    def test_short_password_rejected_before_calling_gotrue(self, client, gotrue):
        # The module's only real validation: password min_length=6. A 5-char pw
        # must 422 at the schema and never reach GoTrue.
        r = client.post(
            "/dev/register", json={"email": "a@b.co", "password": "short"}
        )

        assert r.status_code == 422
        assert gotrue["calls"] == []


# --------------------------------------------------------------------------- #
# Security: credentials must never reach the logs
# --------------------------------------------------------------------------- #
class TestNoCredentialLeak:
    SECRET_PW = "S3cretP@ssw0rd-unique-marker"
    SECRET_TOKEN = "jwt.SUPER-SECRET-TOKEN-marker.zzz"

    def test_login_success_logs_neither_password_nor_token(
        self, client, gotrue, caplog
    ):
        gotrue["resp"] = _FakeResponse(
            200, json_data={"access_token": self.SECRET_TOKEN, "user": {"id": "u"}}
        )
        with caplog.at_level(logging.DEBUG, logger="app.api.dev_auth"):
            client.post(
                "/dev/login", json={"email": "a@b.co", "password": self.SECRET_PW}
            )

        assert self.SECRET_PW not in caplog.text
        assert self.SECRET_TOKEN not in caplog.text

    def test_login_failure_does_not_log_password(self, client, gotrue, caplog):
        gotrue["resp"] = _FakeResponse(400, json_data={"msg": "bad creds"})
        with caplog.at_level(logging.DEBUG, logger="app.api.dev_auth"):
            client.post(
                "/dev/login", json={"email": "a@b.co", "password": self.SECRET_PW}
            )

        # The warning path logs the email + status only — never the password.
        assert self.SECRET_PW not in caplog.text


# --------------------------------------------------------------------------- #
# Gating: the router must be mounted IFF settings.ENABLE_DEV_AUTH (main.py)
# --------------------------------------------------------------------------- #
class TestGating:
    """These reload app.main under an explicit flag and inspect app.routes
    (no TestClient context → lifespan/model warm-load never runs)."""

    @staticmethod
    def _dev_paths_after_reload(monkeypatch, enabled):
        # main.py transitively imports firebase_admin (a declared dependency,
        # requirements.txt). Skip rather than fail where the dep isn't installed
        # — this is an environment gap, not a gating regression.
        pytest.importorskip("firebase_admin")
        from app.core import config

        monkeypatch.setattr(config.settings, "ENABLE_DEV_AUTH", enabled)
        import app.main as main

        importlib.reload(main)
        return {r.path for r in main.app.routes}

    def test_routes_present_when_enabled(self, monkeypatch):
        paths = self._dev_paths_after_reload(monkeypatch, True)
        assert "/dev/login" in paths
        assert "/dev/register" in paths

    def test_routes_absent_when_disabled(self, monkeypatch):
        # The security-critical case: flag off → routes do not exist (404),
        # not merely disabled.
        paths = self._dev_paths_after_reload(monkeypatch, False)
        assert "/dev/login" not in paths
        assert "/dev/register" not in paths
