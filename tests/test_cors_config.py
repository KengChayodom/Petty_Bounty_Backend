"""CORS: allowed origins come from CORS_ALLOW_ORIGINS, and credentials stay off.

Credentials are off because no route authenticates by cookie (the token is a
bearer header), and "*" together with credentials would let any site make
credentialed calls.
"""
from fastapi.testclient import TestClient

from app.core.config import settings


class TestCorsAllowOrigins:
    def test_unset_means_any_origin(self, monkeypatch):
        monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
        assert settings.cors_allow_origins() == ["*"]

    def test_a_comma_separated_list_is_trimmed(self, monkeypatch):
        monkeypatch.setenv(
            "CORS_ALLOW_ORIGINS", " https://admin.example.com , http://localhost:5173 ,",
        )
        assert settings.cors_allow_origins() == [
            "https://admin.example.com", "http://localhost:5173",
        ]


def test_the_app_never_allows_credentials():
    from app.main import app

    r = TestClient(app).options(
        "/health",
        headers={
            "Origin": "https://somewhere.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.headers.get("access-control-allow-origin") is not None
    assert r.headers.get("access-control-allow-credentials") is None
