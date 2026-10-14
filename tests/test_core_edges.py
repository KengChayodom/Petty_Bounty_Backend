"""
Edge branches in core config that the route/service suites don't hit.

  * Settings.__init__ refuses to boot without the required Supabase env vars.

The `_extract_token` case that lived here moved to tests/test_auth.py on
2026-10-13, where it is one of the cases of that method's own unit block.
"""
import pytest

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
