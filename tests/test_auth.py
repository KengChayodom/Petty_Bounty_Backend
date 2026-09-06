"""
Unit tests for app/core/auth.py — the Feature #1 (Authentication) server-side
trust boundary.

Covered (per test_plan.md / test_traceability.md):
  * UTC-01 _strip_bearer                  (MD-05, SRS-21)
  * UTC-02 _resolve_user_id               (MD-07, SRS-21)
  * UTC-03 get_current_user_id            (MD-02, SRS-13) + auth-guard e2e
  * UTC-04 get_current_user_id_optional   (MD-03, SRS-21)
  * UTC-05 require_admin                  (MD-04, SRS-22)

SRS-21 = extract & validate the JWT to authorize requests; SRS-22 = role-based
access control restricting admin functions; SRS-13 = maintain the authenticated
session.

Every protected route depends on this module, and a silent regression here is a
security hole (auth bypass) or a lockout (false 401). The logic is pure and
branch-heavy, so it's unit-tested exhaustively across the token-parsing,
dev-bypass, and admin-role branches.

Boundary rule (per the automated-testing skill): the only thing mocked is the
*boundary* — the Supabase client returned by `get_supabase_client` (its
`auth.get_user` and the `users.role` lookup). We assert on the returned
identity / raised HTTP status, never on "a mock was called" (except where the
contract IS "must not hit the DB" — the admin short-circuit).

Each test owns its data: an autouse fixture pins BOTH dev-bypass flags to False
so tests never depend on the ambient .env (where ENABLE_UNAUTHED_ADMIN may be
true); tests that exercise a bypass opt in explicitly.
"""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, status

from app.core import auth


# --------------------------------------------------------------------------- #
# Boundary fake: the Supabase client surface auth.py actually touches.
# --------------------------------------------------------------------------- #
class _FakeAuthApi:
    def __init__(self, user_id=None, raises=False):
        self._user_id = user_id
        self._raises = raises
        self.tokens_seen = []

    def get_user(self, token):
        self.tokens_seen.append(token)
        if self._raises:
            raise RuntimeError("invalid/expired token")
        if self._user_id is None:
            return SimpleNamespace(user=None)
        return SimpleNamespace(user=SimpleNamespace(id=self._user_id))


class _RoleChain:
    """Supports .select(...).eq(...).maybe_single().execute() and returns .data.

    Mirrors real supabase-py: maybe_single() yields data=None for a missing row
    (it does NOT raise — that's why require_admin uses it), while a transport
    error still raises from execute().
    """

    def __init__(self, role_result):
        self._role_result = role_result

    def select(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    def maybe_single(self):
        return self

    def execute(self):
        if isinstance(self._role_result, Exception):
            raise self._role_result
        return SimpleNamespace(data=self._role_result)


class FakeClient:
    def __init__(self, user_id=None, raises=False, role_result=None):
        self.auth = _FakeAuthApi(user_id, raises)
        self._role_result = role_result
        self.tables_seen = []

    def table(self, name):
        self.tables_seen.append(name)
        return _RoleChain(self._role_result)


@pytest.fixture(autouse=True)
def _secure_defaults(monkeypatch):
    """Pin both dev-escape-hatches OFF so tests don't inherit the .env."""
    monkeypatch.setattr(auth.settings, "AUTH_DEV_BYPASS", False)
    monkeypatch.setattr(auth.settings, "ENABLE_UNAUTHED_ADMIN", False)


def _install_client(monkeypatch, **kwargs):
    """Point auth.get_supabase_client at a FakeClient; return it + a call flag."""
    client = FakeClient(**kwargs)
    calls = {"n": 0}

    def _factory():
        calls["n"] += 1
        return client

    monkeypatch.setattr(auth, "get_supabase_client", _factory)
    return client, calls


# --------------------------------------------------------------------------- #
# Token parsing  (_strip_bearer, exercised through the public surface)
# --------------------------------------------------------------------------- #
class TestStripBearer:
    @pytest.mark.parametrize("raw", [None, "", "   ", "Bearer ", "bearer   "])
    def test_missing_or_empty_token_yields_no_token(self, raw):
        # Guards: a blank/headerless request must not be parsed into a token.
        assert auth._strip_bearer(raw) is None

    def test_strips_bearer_prefix(self):
        assert auth._strip_bearer("Bearer abc.def") == "abc.def"

    def test_scheme_is_case_insensitive(self):
        # Guards a regression to a case-sensitive "Bearer " check.
        assert auth._strip_bearer("bearer abc.def") == "abc.def"

    def test_bare_token_without_scheme_is_kept(self):
        assert auth._strip_bearer("abc.def") == "abc.def"


# --------------------------------------------------------------------------- #
# _resolve_user_id — validates against Supabase, never raises
# --------------------------------------------------------------------------- #
class TestResolveUserId:
    def test_valid_token_returns_supabase_user_id(self, monkeypatch):
        client, _ = _install_client(monkeypatch, user_id="user-123")
        assert auth._resolve_user_id("Bearer good") == "user-123"
        assert client.auth.tokens_seen == ["good"]  # prefix stripped before call

    def test_none_header_short_circuits_without_calling_supabase(self, monkeypatch):
        client, calls = _install_client(monkeypatch, user_id="user-123")
        assert auth._resolve_user_id(None) is None
        # Guards an extra network round-trip on every anonymous request.
        assert calls["n"] == 0
        assert client.auth.tokens_seen == []

    def test_supabase_returns_no_user_yields_none(self, monkeypatch):
        _install_client(monkeypatch, user_id=None)
        assert auth._resolve_user_id("Bearer whatever") is None

    def test_supabase_raising_is_swallowed_to_none(self, monkeypatch):
        # Guards: an invalid/expired token must become a clean None (→401 later),
        # never an unhandled 500.
        _install_client(monkeypatch, raises=True)
        assert auth._resolve_user_id("Bearer expired") is None


# --------------------------------------------------------------------------- #
# get_current_user_id — the protected-route gate
# --------------------------------------------------------------------------- #
class TestGetCurrentUserId:
    def test_valid_token_returns_id(self, monkeypatch):
        _install_client(monkeypatch, user_id="u-1")
        assert auth.get_current_user_id("Bearer good") == "u-1"

    def test_invalid_token_without_bypass_raises_401(self, monkeypatch):
        _install_client(monkeypatch, raises=True)
        with pytest.raises(HTTPException) as exc:
            auth.get_current_user_id("Bearer bad")
        assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED
        # The WWW-Authenticate header is part of the 401 contract.
        assert exc.value.headers.get("WWW-Authenticate") == "Bearer"

    def test_missing_header_without_bypass_raises_401(self, monkeypatch):
        _install_client(monkeypatch, user_id="u-1")
        with pytest.raises(HTTPException) as exc:
            auth.get_current_user_id(None)
        assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_invalid_token_with_dev_bypass_falls_back_to_test_user(self, monkeypatch):
        monkeypatch.setattr(auth.settings, "AUTH_DEV_BYPASS", True)
        _install_client(monkeypatch, raises=True)
        assert auth.get_current_user_id(None) == auth.TEST_USER_ID

    def test_valid_token_wins_even_when_bypass_enabled(self, monkeypatch):
        # Guards: the dev bypass must not shadow a real authenticated user.
        monkeypatch.setattr(auth.settings, "AUTH_DEV_BYPASS", True)
        _install_client(monkeypatch, user_id="real-user")
        assert auth.get_current_user_id("Bearer good") == "real-user"


# --------------------------------------------------------------------------- #
# get_current_user_id_optional — same resolution, never 401s
# --------------------------------------------------------------------------- #
class TestGetCurrentUserIdOptional:
    def test_valid_token_returns_id(self, monkeypatch):
        _install_client(monkeypatch, user_id="u-9")
        assert auth.get_current_user_id_optional("Bearer good") == "u-9"

    def test_invalid_without_bypass_returns_none_not_raise(self, monkeypatch):
        _install_client(monkeypatch, raises=True)
        assert auth.get_current_user_id_optional("Bearer bad") is None

    def test_invalid_with_bypass_returns_test_user(self, monkeypatch):
        monkeypatch.setattr(auth.settings, "AUTH_DEV_BYPASS", True)
        _install_client(monkeypatch, raises=True)
        assert auth.get_current_user_id_optional(None) == auth.TEST_USER_ID


# --------------------------------------------------------------------------- #
# require_admin — server-authoritative role gate
# --------------------------------------------------------------------------- #
class TestRequireAdmin:
    def test_unauthed_admin_flag_short_circuits_with_no_db_call(self, monkeypatch):
        monkeypatch.setattr(auth.settings, "ENABLE_UNAUTHED_ADMIN", True)
        client, calls = _install_client(monkeypatch, user_id="ignored")
        # Even a garbage/None token returns the dev admin id...
        assert auth.require_admin(None) == auth.TEST_USER_ID
        # ...and crucially never touches Supabase (no token validation, no role
        # lookup) — that's the whole point of the local-dev escape hatch.
        assert calls["n"] == 0
        assert client.tables_seen == []

    def test_admin_role_returns_user_id(self, monkeypatch):
        _install_client(monkeypatch, user_id="admin-1", role_result={"role": "admin"})
        assert auth.require_admin("Bearer good") == "admin-1"

    def test_non_admin_role_raises_403(self, monkeypatch):
        _install_client(monkeypatch, user_id="user-1", role_result={"role": "user"})
        with pytest.raises(HTTPException) as exc:
            auth.require_admin("Bearer good")
        assert exc.value.status_code == status.HTTP_403_FORBIDDEN

    def test_missing_profile_row_raises_403(self, monkeypatch):
        # maybe_single() with no row → data=None → role None → not admin (403,
        # NOT a 500). Guards against regressing back to .single(), which raises
        # on no-row and would surface as a 500.
        _install_client(monkeypatch, user_id="ghost", role_result=None)
        with pytest.raises(HTTPException) as exc:
            auth.require_admin("Bearer good")
        assert exc.value.status_code == status.HTTP_403_FORBIDDEN

    def test_no_token_without_bypass_raises_401(self, monkeypatch):
        _install_client(monkeypatch, raises=True)
        with pytest.raises(HTTPException) as exc:
            auth.require_admin(None)
        assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_role_lookup_error_raises_500(self, monkeypatch):
        # A DB failure during role lookup must surface as 500, not be mistaken
        # for "not an admin" (403) or — worse — fall through as authorized.
        _install_client(
            monkeypatch, user_id="admin-1", role_result=RuntimeError("db down")
        )
        with pytest.raises(HTTPException) as exc:
            auth.require_admin("Bearer good")
        assert exc.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
