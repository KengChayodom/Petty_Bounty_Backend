"""
Shared fixtures + boundary stubs.

Design rule (per db-testing-seams): service/route tests double the repository
*ports* this codebase owns (MagicMock(spec=<Repo>)), never the supabase-py
client. There is deliberately NO hand-rolled Supabase fake here — the vendor
client is exercised only by the adapter integration suite (tests/integration).
"""
import sys
import types

import pytest


# --------------------------------------------------------------------------- #
# firebase-admin boundary stub.
# firebase-admin is a declared dependency (requirements.txt) but is a heavy SDK
# (grpcio, google-cloud) that the FAST unit suite must not require. Every push
# path is guarded behind is_firebase_ready() / mocks the messaging boundary, so
# the real SDK is never exercised by a unit test. Where it isn't installed, we
# inject a minimal stub exposing only the surface app code imports, so
# app.core.firebase / app.services.notification_service import cleanly. On a
# machine where the real SDK IS installed we leave it untouched (the tests
# monkeypatch the boundary either way).
# --------------------------------------------------------------------------- #
def _install_firebase_stub() -> None:
    try:
        import firebase_admin  # noqa: F401
        return  # real SDK present — don't shadow it
    except ModuleNotFoundError:
        pass

    fb = types.ModuleType("firebase_admin")
    fb._apps = {}  # is_firebase_ready() reads this; empty => not ready
    fb.initialize_app = lambda *a, **k: None

    credentials = types.ModuleType("firebase_admin.credentials")
    credentials.Certificate = lambda *a, **k: object()

    messaging = types.ModuleType("firebase_admin.messaging")

    class _Notification:
        def __init__(self, title=None, body=None):
            self.title, self.body = title, body

    class _AndroidNotification:
        def __init__(self, channel_id=None):
            self.channel_id = channel_id

    class _AndroidConfig:
        def __init__(self, priority=None, notification=None):
            self.priority, self.notification = priority, notification

    class _MulticastMessage:
        def __init__(self, tokens=None, notification=None, data=None, android=None):
            self.tokens, self.notification = tokens, notification
            self.data, self.android = data, android

    class UnregisteredError(Exception):
        pass

    def _send_each_for_multicast(message):  # tests monkeypatch this
        raise NotImplementedError("FCM boundary must be mocked in tests")

    messaging.Notification = _Notification
    messaging.AndroidNotification = _AndroidNotification
    messaging.AndroidConfig = _AndroidConfig
    messaging.MulticastMessage = _MulticastMessage
    messaging.UnregisteredError = UnregisteredError
    messaging.send_each_for_multicast = _send_each_for_multicast

    fb.credentials = credentials
    fb.messaging = messaging
    sys.modules["firebase_admin"] = fb
    sys.modules["firebase_admin.credentials"] = credentials
    sys.modules["firebase_admin.messaging"] = messaging


_install_firebase_stub()

from app.services.ai_cache import AnalyzeCache


# --------------------------------------------------------------------------- #
# Integration-test gate. The tests/integration suite needs a real Postgres
# (pgvector+PostGIS) via Docker/Testcontainers, which is far too slow to spin
# up on every run of the fast unit suite. So integration tests are SKIPPED
# unless `--integration` is passed; the Postgres container fixture is
# session-scoped and only materialises when a non-skipped integration test
# actually runs.
# --------------------------------------------------------------------------- #
def pytest_addoption(parser):
    parser.addoption(
        "--integration", action="store_true", default=False,
        help="run integration tests (requires Docker; spins up an ephemeral Postgres)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration"):
        return
    skip_integration = pytest.mark.skip(reason="needs --integration (real Postgres)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


# --------------------------------------------------------------------------- #
# AnalyzeCache is a class-level singleton that survives across requests (and,
# without this, across tests). Every test must own its data, so wipe it before
# and after each test to kill order-dependence.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clean_analyze_cache():
    AnalyzeCache.clear()
    yield
    AnalyzeCache.clear()
