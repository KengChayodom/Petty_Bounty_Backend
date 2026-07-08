"""
Unit tests for app/services/notification_service.send_to_users (UTC-10, MD-13, SRS-21).

Progress-I SRS traceability: SRS-21 (a push is sent to nearby hunters — the
multicast token/title/body/data assertions) and SRS-22 (owner exclusion is
enforced upstream by get_nearby_hunters; see integration/test_get_nearby_hunters).

Boundary rule (per the automated-testing skill): the only things mocked are the
*boundaries* — the FCM SDK (messaging.send_each_for_multicast) and the Supabase
client (token select + stale-token delete). We assert on WHAT was sent (the
multicast tokens / title / body / data) and WHAT was pruned (the deleted
tokens), never on "a mock was called".

The firebase SDK boundary is provided by the conftest stub when the real SDK
isn't installed, so this suite is hermetic and fast.

Each test owns its data: is_firebase_ready is pinned per-test (default OFF is
the production-safe state) and a fresh fake DB is built each time.
"""
import logging
from types import SimpleNamespace

import pytest

from app.services import notification_service as ns

messaging = ns.messaging


# --------------------------------------------------------------------------- #
# Boundary fakes
# --------------------------------------------------------------------------- #
class _NotifQuery:
    """The supabase chain send_to_users uses: select/delete + .in_ + .execute."""

    def __init__(self, db):
        self._db = db
        self._op = "select"
        self._filter_vals = None

    def select(self, *_a, **_k):
        self._op = "select"
        return self

    def delete(self):
        self._op = "delete"
        return self

    def in_(self, _column, values):
        self._filter_vals = list(values)
        return self

    def execute(self):
        if self._op == "select":
            self._db.selected_user_ids = self._filter_vals
            if self._db.select_raises:
                raise RuntimeError("token load failed")
            return SimpleNamespace(data=self._db.token_rows)
        # delete (prune)
        self._db.deleted_tokens = self._filter_vals
        if self._db.delete_raises:
            raise RuntimeError("prune failed")
        return SimpleNamespace(data=[])


class FakeDB:
    def __init__(self, token_rows=None, select_raises=False, delete_raises=False):
        self.token_rows = token_rows if token_rows is not None else []
        self.select_raises = select_raises
        self.delete_raises = delete_raises
        self.selected_user_ids = None
        self.deleted_tokens = None
        self.table_calls = []

    def table(self, name):
        self.table_calls.append(name)
        return _NotifQuery(self)


def _rows(*tokens):
    return [{"fcm_token": t} for t in tokens]


def _batch(*results):
    """Build a send_each_for_multicast response: results are (success, exc)."""
    responses = [SimpleNamespace(success=s, exception=e) for s, e in results]
    return SimpleNamespace(
        responses=responses,
        success_count=sum(1 for s, _ in results if s),
        failure_count=sum(1 for s, _ in results if not s),
    )


@pytest.fixture
def fcm(monkeypatch):
    """Capture the multicast message(s) sent; program the response or an error."""
    state = {"messages": [], "response": None, "raises": None}

    def fake_send(message):
        state["messages"].append(message)
        if state["raises"] is not None:
            raise state["raises"]
        return state["response"]

    monkeypatch.setattr(messaging, "send_each_for_multicast", fake_send)
    return state


@pytest.fixture
def ready(monkeypatch):
    """Force is_firebase_ready True (the default for send tests)."""
    monkeypatch.setattr(ns, "is_firebase_ready", lambda: True)


# --------------------------------------------------------------------------- #
# No-op guards
# --------------------------------------------------------------------------- #
def test_noop_when_no_user_ids(monkeypatch, fcm):
    # Empty recipient list short-circuits before readiness/DB/FCM are touched.
    monkeypatch.setattr(ns, "is_firebase_ready", lambda: True)
    db = FakeDB(token_rows=_rows("t0"))

    ns.send_to_users(db, [], "T", "B")

    assert db.table_calls == []
    assert fcm["messages"] == []


def test_noop_when_firebase_not_ready(monkeypatch, fcm):
    # The whole feature is OFF when Firebase isn't configured: no DB, no send.
    monkeypatch.setattr(ns, "is_firebase_ready", lambda: False)
    db = FakeDB(token_rows=_rows("t0"))

    ns.send_to_users(db, ["u1"], "T", "B")

    assert db.table_calls == []
    assert fcm["messages"] == []


def test_noop_when_no_tokens_for_users(ready, fcm):
    db = FakeDB(token_rows=[])  # users have registered no devices

    ns.send_to_users(db, ["u1"], "T", "B")

    assert db.selected_user_ids == ["u1"]  # it DID query...
    assert fcm["messages"] == []           # ...but sent nothing


def test_db_load_failure_is_swallowed(ready, fcm):
    db = FakeDB(token_rows=_rows("t0"), select_raises=True)

    ns.send_to_users(db, ["u1"], "T", "B")  # must not raise

    assert fcm["messages"] == []


def test_send_failure_is_swallowed(ready, fcm):
    db = FakeDB(token_rows=_rows("t0"))
    fcm["raises"] = RuntimeError("FCM unreachable")

    ns.send_to_users(db, ["u1"], "T", "B")  # must not raise

    assert db.deleted_tokens is None  # never reaches the prune step


# --------------------------------------------------------------------------- #
# Happy path: sends exactly the right tokens / payload
# --------------------------------------------------------------------------- #
def test_sends_all_tokens_with_stringified_data(ready, fcm):
    db = FakeDB(token_rows=_rows("tok-a", "tok-b", "tok-c"))
    fcm["response"] = _batch((True, None), (True, None), (True, None))

    ns.send_to_users(
        db, ["u1", "u2"], "Title!", "Body!", data={"petId": 42, "kind": "lost"}
    )

    assert db.selected_user_ids == ["u1", "u2"]
    assert len(fcm["messages"]) == 1
    msg = fcm["messages"][0]
    assert msg.tokens == ["tok-a", "tok-b", "tok-c"]
    assert msg.notification.title == "Title!"
    assert msg.notification.body == "Body!"
    # FCM requires string data values — ints must be coerced.
    assert msg.data == {"petId": "42", "kind": "lost"}
    assert db.deleted_tokens is None  # all delivered -> nothing pruned


# --------------------------------------------------------------------------- #
# Pruning of dead tokens
# --------------------------------------------------------------------------- #
def test_prunes_only_unregistered_tokens(ready, fcm):
    db = FakeDB(token_rows=_rows("good", "dead", "transient"))
    # good -> ok; dead -> UnregisteredError (prune); transient -> other error (keep)
    fcm["response"] = _batch(
        (True, None),
        (False, messaging.UnregisteredError("uninstalled")),
        (False, ValueError("temporary")),
    )

    ns.send_to_users(db, ["u1"], "T", "B")

    # Exactly the UnregisteredError token is deleted; the transient one is kept.
    assert db.deleted_tokens == ["dead"]


def test_no_prune_when_all_delivered(ready, fcm):
    db = FakeDB(token_rows=_rows("t0", "t1"))
    fcm["response"] = _batch((True, None), (True, None))

    ns.send_to_users(db, ["u1"], "T", "B")

    assert db.deleted_tokens is None
    assert db.table_calls == ["device_tokens"]  # select only, no delete call


def test_prune_failure_is_swallowed(ready, fcm):
    db = FakeDB(token_rows=_rows("dead"), delete_raises=True)
    fcm["response"] = _batch((False, messaging.UnregisteredError("x")))

    ns.send_to_users(db, ["u1"], "T", "B")  # delete blows up but is caught

    assert db.deleted_tokens == ["dead"]  # it attempted the prune


# --------------------------------------------------------------------------- #
# Security: device tokens must never reach the logs
# --------------------------------------------------------------------------- #
def test_never_logs_tokens(ready, fcm, caplog):
    secret = "FCM-SECRET-TOKEN-do-not-log"
    db = FakeDB(token_rows=_rows(secret, "another-SECRET-tok"))
    fcm["response"] = _batch(
        (True, None),
        (False, messaging.UnregisteredError("gone")),
    )

    with caplog.at_level(logging.DEBUG, logger="app.services.notification_service"):
        ns.send_to_users(db, ["u1"], "T", "B")

    assert secret not in caplog.text
    assert "another-SECRET-tok" not in caplog.text
