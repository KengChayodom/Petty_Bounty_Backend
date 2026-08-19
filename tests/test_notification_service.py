"""
Unit tests for app/services/notification_service.send_to_users (UTC-10, MD-13, SRS-21).

Progress-I SRS traceability: SRS-21 (a push is sent to nearby hunters — the
multicast token/title/body/data assertions) and SRS-22 (owner exclusion is
enforced upstream by get_nearby_hunters; see integration/test_get_nearby_hunters).

Boundary rule (per db-testing-seams): the DB is reached only through the
NotificationRepository port owned by this codebase, so we double THAT with
MagicMock(spec=...) — never a hand-rolled Supabase client. The FCM SDK
(messaging.send_each_for_multicast) is the other boundary. We assert on WHAT was
sent (the multicast tokens / title / body / data) and WHAT was pruned (the token
list handed to delete_device_tokens), never on incidental call plumbing.

The firebase SDK boundary is provided by the conftest stub when the real SDK
isn't installed, so this suite is hermetic and fast.

Each test owns its data: is_firebase_ready is pinned per-test (default OFF is
the production-safe state) and a fresh repo double is built each time.
"""
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.repositories.notification_repository import NotificationRepository
from app.services import notification_service as ns

messaging = ns.messaging


# --------------------------------------------------------------------------- #
# Boundary doubles
# --------------------------------------------------------------------------- #
def _repo(tokens=None, tokens_raise=False, delete_raises=False):
    """A NotificationRepository double: get_fcm_tokens_for_users returns the
    given tokens (or raises); delete_device_tokens records / optionally raises."""
    repo = MagicMock(spec=NotificationRepository)
    if tokens_raise:
        repo.get_fcm_tokens_for_users.side_effect = RuntimeError("token load failed")
    else:
        repo.get_fcm_tokens_for_users.return_value = list(tokens or [])
    if delete_raises:
        repo.delete_device_tokens.side_effect = RuntimeError("prune failed")
    return repo


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
    repo = _repo(tokens=["t0"])

    ns.send_to_users(repo, [], "T", "B")

    repo.get_fcm_tokens_for_users.assert_not_called()
    assert fcm["messages"] == []


def test_noop_when_firebase_not_ready(monkeypatch, fcm):
    # The whole feature is OFF when Firebase isn't configured: no DB, no send.
    monkeypatch.setattr(ns, "is_firebase_ready", lambda: False)
    repo = _repo(tokens=["t0"])

    ns.send_to_users(repo, ["u1"], "T", "B")

    repo.get_fcm_tokens_for_users.assert_not_called()
    assert fcm["messages"] == []


def test_noop_when_no_tokens_for_users(ready, fcm):
    repo = _repo(tokens=[])  # users have registered no devices

    ns.send_to_users(repo, ["u1"], "T", "B")

    repo.get_fcm_tokens_for_users.assert_called_once_with(["u1"])  # it DID query...
    assert fcm["messages"] == []                                   # ...but sent nothing


def test_db_load_failure_is_swallowed(ready, fcm):
    repo = _repo(tokens_raise=True)

    ns.send_to_users(repo, ["u1"], "T", "B")  # must not raise

    assert fcm["messages"] == []


def test_send_failure_is_swallowed(ready, fcm):
    repo = _repo(tokens=["t0"])
    fcm["raises"] = RuntimeError("FCM unreachable")

    ns.send_to_users(repo, ["u1"], "T", "B")  # must not raise

    repo.delete_device_tokens.assert_not_called()  # never reaches the prune step


# --------------------------------------------------------------------------- #
# Happy path: sends exactly the right tokens / payload
# --------------------------------------------------------------------------- #
def test_sends_all_tokens_with_stringified_data(ready, fcm):
    repo = _repo(tokens=["tok-a", "tok-b", "tok-c"])
    fcm["response"] = _batch((True, None), (True, None), (True, None))

    ns.send_to_users(
        repo, ["u1", "u2"], "Title!", "Body!", data={"petId": 42, "kind": "lost"}
    )

    repo.get_fcm_tokens_for_users.assert_called_once_with(["u1", "u2"])
    assert len(fcm["messages"]) == 1
    msg = fcm["messages"][0]
    assert msg.tokens == ["tok-a", "tok-b", "tok-c"]
    assert msg.notification.title == "Title!"
    assert msg.notification.body == "Body!"
    # FCM requires string data values — ints must be coerced.
    assert msg.data == {"petId": "42", "kind": "lost"}
    repo.delete_device_tokens.assert_not_called()  # all delivered -> nothing pruned


# --------------------------------------------------------------------------- #
# Pruning of dead tokens
# --------------------------------------------------------------------------- #
def test_prunes_only_unregistered_tokens(ready, fcm):
    repo = _repo(tokens=["good", "dead", "transient"])
    # good -> ok; dead -> UnregisteredError (prune); transient -> other error (keep)
    fcm["response"] = _batch(
        (True, None),
        (False, messaging.UnregisteredError("uninstalled")),
        (False, ValueError("temporary")),
    )

    ns.send_to_users(repo, ["u1"], "T", "B")

    # Exactly the UnregisteredError token is deleted; the transient one is kept.
    repo.delete_device_tokens.assert_called_once_with(["dead"])


def test_no_prune_when_all_delivered(ready, fcm):
    repo = _repo(tokens=["t0", "t1"])
    fcm["response"] = _batch((True, None), (True, None))

    ns.send_to_users(repo, ["u1"], "T", "B")

    repo.delete_device_tokens.assert_not_called()


def test_prune_failure_is_swallowed(ready, fcm):
    repo = _repo(tokens=["dead"], delete_raises=True)
    fcm["response"] = _batch((False, messaging.UnregisteredError("x")))

    ns.send_to_users(repo, ["u1"], "T", "B")  # delete blows up but is caught

    repo.delete_device_tokens.assert_called_once_with(["dead"])  # it attempted the prune


# --------------------------------------------------------------------------- #
# Security: device tokens must never reach the logs
# --------------------------------------------------------------------------- #
def test_never_logs_tokens(ready, fcm, caplog):
    secret = "FCM-SECRET-TOKEN-do-not-log"
    repo = _repo(tokens=[secret, "another-SECRET-tok"])
    fcm["response"] = _batch(
        (True, None),
        (False, messaging.UnregisteredError("gone")),
    )

    with caplog.at_level(logging.DEBUG, logger="app.services.notification_service"):
        ns.send_to_users(repo, ["u1"], "T", "B")

    assert secret not in caplog.text
    assert "another-SECRET-tok" not in caplog.text


# --------------------------------------------------------------------------- #
# notify_nearby_hunters — the fan-out entry point (runs as a BackgroundTask).
# It builds its own NotificationRepository from the raw client, so we patch that
# construction + send_to_users. Category-Partition:
#   * firebase not ready            -> no-op                       [single]
#   * get_nearby_hunters raises      -> swallowed, no send          [error]
#   * get_nearby_hunters returns []  -> no-op                       [single]
#   * returns ids                    -> send_to_users with geo+payload
# --------------------------------------------------------------------------- #
class TestNotifyNearbyHunters:
    @pytest.fixture
    def wired(self, monkeypatch):
        """Patch the boundaries the fn builds/calls internally: the repo it
        constructs from the raw client, and send_to_users (spied)."""
        repo = MagicMock(spec=NotificationRepository)
        monkeypatch.setattr(ns, "SupabaseNotificationRepository", lambda db: repo)
        sends = []
        monkeypatch.setattr(ns, "send_to_users", lambda *a, **k: sends.append((a, k)))
        return SimpleNamespace(repo=repo, sends=sends)

    @staticmethod
    def _notify():
        ns.notify_nearby_hunters(
            object(), "pet-1", 13.7563, 100.5018, "owner-1", 10.0, "Luna", "Dog",
            max_age_hours=12,
        )

    def test_noop_when_firebase_not_ready(self, monkeypatch, wired):
        monkeypatch.setattr(ns, "is_firebase_ready", lambda: False)
        self._notify()
        wired.repo.get_nearby_hunters.assert_not_called()
        assert wired.sends == []

    def test_swallows_nearby_query_failure(self, monkeypatch, wired):
        monkeypatch.setattr(ns, "is_firebase_ready", lambda: True)
        wired.repo.get_nearby_hunters.side_effect = RuntimeError("rpc down")
        self._notify()  # must not raise
        assert wired.sends == []

    def test_noop_when_no_nearby_hunters(self, monkeypatch, wired):
        monkeypatch.setattr(ns, "is_firebase_ready", lambda: True)
        wired.repo.get_nearby_hunters.return_value = []
        self._notify()
        assert wired.sends == []

    def test_sends_to_nearby_hunters_with_geo_and_payload(self, monkeypatch, wired):
        monkeypatch.setattr(ns, "is_firebase_ready", lambda: True)
        wired.repo.get_nearby_hunters.return_value = ["u1", "u2"]
        self._notify()
        # geo query: WKT (lng lat), km->m, freshness window, owner excluded
        wired.repo.get_nearby_hunters.assert_called_once_with(
            "POINT(100.5018 13.7563)", 10000.0, 12, "owner-1"
        )
        assert len(wired.sends) == 1
        args, kwargs = wired.sends[0]
        assert args[0] is wired.repo          # the same repo instance is reused
        assert args[1] == ["u1", "u2"]        # the resolved hunter ids
        assert kwargs["data"] == {"petId": "pet-1"}
        assert "Luna" in kwargs["body"] and "Dog" in kwargs["body"]


# --------------------------------------------------------------------------- #
# notify_pet_owners — the owner half of the loop (2026-08-17).
#
# Until this existed a sighting landed in the database and nobody told the
# owner, so `sighting_status` never left Pending_Analysis. The rule these tests
# protect is narrow and important: the status advances to Notified_Owner ONLY
# when FCM actually delivered something. A status that claims a push nobody
# received is worse than one that stays put — the owner would read "we told
# you" having never been told.
# --------------------------------------------------------------------------- #
class TestNotifyPetOwners:
    @pytest.fixture
    def wired(self, monkeypatch):
        """Double the two ports the function builds from the raw client, plus
        send_to_users (whose RETURN VALUE is the delivery signal under test)."""
        notif_repo = MagicMock(spec=NotificationRepository)
        sighting_repo = MagicMock()
        sighting_repo.get_pet_owners.return_value = {"p1": "owner-1"}
        monkeypatch.setattr(
            ns, "SupabaseNotificationRepository", lambda db: notif_repo
        )
        monkeypatch.setattr(
            ns, "SupabaseSightingRepository", lambda db: sighting_repo
        )
        monkeypatch.setattr(ns, "is_firebase_ready", lambda: True)

        sends = []

        def fake_send(repo, user_ids, **kwargs):
            sends.append((user_ids, kwargs))
            return state["delivered"]

        state = {"delivered": 1}
        monkeypatch.setattr(ns, "send_to_users", fake_send)
        return SimpleNamespace(
            notif=notif_repo, sightings=sighting_repo, sends=sends, state=state,
        )

    def test_no_pets_short_circuits(self, wired):
        """A sighting that matched nothing has nobody to tell."""
        assert ns.notify_pet_owners(object(), "s1", [], "h1") == []
        assert wired.sends == []
        wired.sightings.set_sighting_status.assert_not_called()

    def test_notifies_owner_and_advances_the_status(self, wired):
        out = ns.notify_pet_owners(object(), "s1", ["p1"], "h1")

        assert out == ["owner-1"]
        assert wired.sends[0][0] == ["owner-1"]
        assert wired.sends[0][1]["data"] == {"sightingId": "s1"}
        wired.sightings.set_sighting_status.assert_called_once_with(
            "s1", "Notified_Owner"
        )

    def test_owner_reporting_their_own_pet_is_not_pushed(self, wired):
        """An owner who spots their own pet and reports it does not need a
        notification telling them about it."""
        wired.sightings.get_pet_owners.return_value = {"p1": "h1"}

        out = ns.notify_pet_owners(object(), "s1", ["p1"], hunter_id="h1")

        assert out == []
        assert wired.sends == []
        wired.sightings.set_sighting_status.assert_not_called()

    def test_one_push_per_owner_when_several_of_their_pets_match(self, wired):
        """Two of the same person's pets matching one photo is one push, not
        two notifications about the same sighting."""
        wired.sightings.get_pet_owners.return_value = {
            "p1": "owner-1", "p2": "owner-1", "p3": "owner-2",
        }

        ns.notify_pet_owners(object(), "s1", ["p1", "p2", "p3"], "h1")

        assert wired.sends[0][0] == ["owner-1", "owner-2"]

    def test_status_not_written_when_nothing_was_delivered(self, wired):
        """No device token / FCM rejected everything => the owner was NOT
        notified, so the sighting stays Pending_Analysis."""
        wired.state["delivered"] = 0

        out = ns.notify_pet_owners(object(), "s1", ["p1"], "h1")

        assert out == []
        wired.sightings.set_sighting_status.assert_not_called()

    def test_status_not_written_when_firebase_is_unconfigured(self, monkeypatch):
        """Local dev without credentials must neither push nor pretend to."""
        sighting_repo = MagicMock()
        monkeypatch.setattr(ns, "is_firebase_ready", lambda: False)
        monkeypatch.setattr(
            ns, "SupabaseSightingRepository", lambda db: sighting_repo
        )

        assert ns.notify_pet_owners(object(), "s1", ["p1"], "h1") == []
        sighting_repo.set_sighting_status.assert_not_called()

    def test_owner_lookup_failure_is_swallowed(self, wired):
        """Runs as a BackgroundTask — it must never raise into the request."""
        wired.sightings.get_pet_owners.side_effect = RuntimeError("db down")

        assert ns.notify_pet_owners(object(), "s1", ["p1"], "h1") == []
        assert wired.sends == []

    def test_status_write_failure_does_not_undo_a_delivered_push(self, wired):
        """The owners DID get the notification; failing to record it is
        bookkeeping, not a failed notification."""
        wired.sightings.set_sighting_status.side_effect = RuntimeError("db down")

        out = ns.notify_pet_owners(object(), "s1", ["p1"], "h1")

        assert out == ["owner-1"]      # still reported as notified


# --------------------------------------------------------------------------- #
# send_to_users' return value — the signal notify_pet_owners gates on.
# --------------------------------------------------------------------------- #
class TestSendToUsersReturnsDeliveredCount:
    def test_returns_success_count(self, ready, fcm):
        repo = _repo(tokens=["tok-a", "tok-b"])
        fcm["response"] = _batch((True, None), (True, None))
        assert ns.send_to_users(repo, ["u1"], "T", "B") == 2

    def test_zero_when_firebase_unconfigured(self, monkeypatch, fcm):
        monkeypatch.setattr(ns, "is_firebase_ready", lambda: False)
        assert ns.send_to_users(_repo(tokens=["t"]), ["u1"], "T", "B") == 0

    def test_zero_when_no_tokens(self, ready, fcm):
        assert ns.send_to_users(_repo(tokens=[]), ["u1"], "T", "B") == 0

    def test_zero_when_no_recipients(self, ready, fcm):
        assert ns.send_to_users(_repo(tokens=["t"]), [], "T", "B") == 0

    def test_zero_when_the_send_raises(self, ready, fcm):
        fcm["raises"] = RuntimeError("fcm down")
        assert ns.send_to_users(_repo(tokens=["t"]), ["u1"], "T", "B") == 0

    def test_zero_when_token_load_fails(self, ready, fcm):
        assert ns.send_to_users(
            _repo(tokens_raise=True), ["u1"], "T", "B"
        ) == 0
