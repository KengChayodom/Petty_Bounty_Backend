"""
Edge branches in core config + auth that the route/service suites don't hit.

  * Settings.__init__ refuses to boot without the required Supabase env vars.
  * _extract_token falls through to the raw Authorization header when the
    HTTPBearer credentials are present but blank.
"""
import pytest
from fastapi.security import HTTPAuthorizationCredentials

from app.core.auth import _extract_token
from app.core.config import Settings


class TestConfigRequiredSettings:
    def test_missing_supabase_url_raises(self, monkeypatch):
        monkeypatch.setattr(Settings, "SUPABASE_URL", "")
        with pytest.raises(ValueError):
            Settings()

    def test_missing_service_key_raises(self, monkeypatch):
        # URL stays set (real class attr) so the first guard passes and the
        # second one is the branch under test.
        monkeypatch.setattr(Settings, "SUPABASE_SERVICE_KEY", "")
        with pytest.raises(ValueError):
            Settings()


class TestExtractTokenFallthrough:
    def test_blank_bearer_credentials_falls_through_to_raw_header(self):
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="   ")
        # credentials present but blank -> the `if token:` guard is False, so it
        # falls through to the raw Authorization header (bare-token fallback).
        assert _extract_token("Bearer raw-tok", creds) == "raw-tok"
