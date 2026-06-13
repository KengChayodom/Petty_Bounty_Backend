"""
Shared fixtures + a Supabase test double.

Design rule (per the automated-testing skill): mock only at the *boundary*
— here that boundary is the supabase-py client. The fake records the exact
payloads the service hands to the DB and returns programmable responses, so
tests assert on real results/state rather than "a mock was called".
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


# --------------------------------------------------------------------------- #
# Fake Supabase client
# --------------------------------------------------------------------------- #
class FakeResult:
    """Mimics the supabase-py APIResponse shape the service reads: .data/.count."""

    def __init__(self, data=None, count=None):
        self.data = data
        self.count = count


class _Chain:
    """A fluent query builder. Filters/modifiers are no-ops returning self;
    .execute() records the terminal op and returns the queued result."""

    def __init__(self, db, table):
        self._db = db
        self._table = table
        self._op = "select"
        self._payload = None
        self._on_conflict = None
        self._filters = []   # [(col, value), ...] captured from .eq(...)

    # mutating ops capture their payload
    def insert(self, payload):
        self._op, self._payload = "insert", payload
        return self

    def upsert(self, rows, on_conflict=None):
        self._op, self._payload, self._on_conflict = "upsert", rows, on_conflict
        return self

    def update(self, payload):
        self._op, self._payload = "update", payload
        return self

    def select(self, *_a, **_k):
        self._op = "select"
        return self

    def delete(self):
        self._op = "delete"
        return self

    # eq is captured so tests can assert a write was scoped to the right row
    # (e.g. /me/location updating ONLY the JWT user). Other filters are swallowed.
    def eq(self, *args, **_k):
        if args:
            self._filters.append(tuple(args))
        return self

    def in_(self, *_a, **_k):
        return self

    def order(self, *_a, **_k):
        return self

    def range(self, *_a, **_k):
        return self

    def execute(self):
        self._db.recorded.append((self._table, self._op, self._payload, self._on_conflict))
        self._db.recorded_filters.append((self._table, self._op, list(self._filters)))
        return self._db._table_results.get((self._table, self._op), FakeResult(data=[]))


class _RpcChain:
    def __init__(self, db, name):
        self._db = db
        self._name = name

    def execute(self):
        return self._db._rpc_results.get(self._name, FakeResult(data=[]))


class FakeSupabase:
    """Programmable stand-in for the supabase-py client."""

    def __init__(self):
        self.recorded = []          # [(table, op, payload, on_conflict), ...]
        self.recorded_filters = []  # [(table, op, [(col, value), ...]), ...]
        self.rpc_calls = []         # [(name, params), ...]
        self._table_results = {}    # (table, op) -> FakeResult
        self._rpc_results = {}      # name -> FakeResult

    # --- programming the responses ---------------------------------------- #
    def set_table_result(self, table, op, data=None, count=None):
        self._table_results[(table, op)] = FakeResult(data=data, count=count)

    def set_rpc_result(self, name, data):
        self._rpc_results[name] = FakeResult(data=data)

    # --- the client surface the service actually calls -------------------- #
    def table(self, name):
        return _Chain(self, name)

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        return _RpcChain(self, name)

    # --- helpers for assertions ------------------------------------------- #
    def payload_for(self, table, op):
        """Return the payload from the (last) recorded op on table, or None."""
        for t, o, payload, _oc in reversed(self.recorded):
            if t == table and o == op:
                return payload
        return None

    def on_conflict_for(self, table, op):
        """Return the on_conflict key from the (last) recorded op, or None."""
        for t, o, _payload, oc in reversed(self.recorded):
            if t == table and o == op:
                return oc
        return None

    def filters_for(self, table, op):
        """Return the [(col, value), ...] eq-filters from the last recorded op."""
        for t, o, filters in reversed(self.recorded_filters):
            if t == table and o == op:
                return filters
        return []


@pytest.fixture
def fake_db():
    return FakeSupabase()
