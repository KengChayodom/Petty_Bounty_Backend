"""
Route unit tests for the FCM geo-push surface (SRS-FR-12):
  * POST /devices/register   — upsert keyed on fcm_token (re-registration reassigns)  [SRS-19]
  * POST /devices/unregister — drop the caller's token on logout                       [SRS-20]
  * POST /me/location        — write last_location + last_location_at for the JWT user [SRS-23]
  * POST /missing-pets/      — fan out notify_nearby_hunters via BackgroundTasks  [SRS-21, SRS-22]

Progress-I SRS traceability: SRS-19 (TestRegisterDevice), SRS-20 (TestUnregisterDevice),
SRS-23 (TestUpdateLocation), SRS-21 + SRS-22 (TestMissingPetFanout).

Boundary rule: the auth dependency and the Supabase client are the boundaries;
both are replaced via FastAPI dependency_overrides. We assert on the payload the
route hands the DB (and, for the fan-out, on the scheduled background task and
its ordering), never on "a mock was called".

The shared FakeSupabase (conftest) is the DB double; its default execute returns
data=[], so success tests must program a row explicitly.
"""
import starlette.background as background
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import devices, me, missing_pets
from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client
from app.utils.postgis import create_postgis_point

JWT_USER = "jwt-user-1111"


def _app(router, fake_db):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user_id] = lambda: JWT_USER
    app.dependency_overrides[get_supabase_client] = lambda: fake_db
    return TestClient(app)


# --------------------------------------------------------------------------- #
# POST /devices/register  (SRS-19: capture/store the FCM token on login)
# --------------------------------------------------------------------------- #
class TestRegisterDevice:
    def test_upserts_keyed_on_fcm_token_with_jwt_user(self, fake_db):
        fake_db.set_table_result(
            "device_tokens", "upsert",
            data=[{"id": "row-1", "user_id": JWT_USER,
                   "fcm_token": "tok-abc", "platform": "android"}],
        )
        client = _app(devices.router, fake_db)

        r = client.post(
            "/devices/register",
            json={"fcm_token": "tok-abc", "platform": "android"},
        )

        assert r.status_code == 200
        assert r.json()["status"] == "success"
        payload = fake_db.payload_for("device_tokens", "upsert")
        assert payload["fcm_token"] == "tok-abc"
        assert payload["platform"] == "android"
        assert payload["user_id"] == JWT_USER
        assert "updated_at" in payload
        # The dedup contract: upsert MUST conflict-resolve on fcm_token so a
        # token moving accounts reassigns instead of duplicating.
        assert fake_db.on_conflict_for("device_tokens", "upsert") == "fcm_token"

    def test_user_id_comes_from_jwt_not_request_body(self, fake_db):
        fake_db.set_table_result(
            "device_tokens", "upsert", data=[{"id": "row-1"}],
        )
        client = _app(devices.router, fake_db)

        # A client trying to register a token for someone else must be ignored.
        client.post(
            "/devices/register",
            json={"fcm_token": "t", "platform": "ios", "user_id": "attacker"},
        )

        assert fake_db.payload_for("device_tokens", "upsert")["user_id"] == JWT_USER

    def test_invalid_platform_is_422_and_never_writes(self, fake_db):
        client = _app(devices.router, fake_db)

        r = client.post(
            "/devices/register",
            json={"fcm_token": "t", "platform": "windows"},
        )

        assert r.status_code == 422
        assert fake_db.recorded == []

    def test_empty_fcm_token_is_422(self, fake_db):
        client = _app(devices.router, fake_db)
        r = client.post(
            "/devices/register",
            json={"fcm_token": "", "platform": "android"},
        )
        assert r.status_code == 422

    def test_upsert_returning_no_row_is_500(self, fake_db):
        # Default fake execute returns data=[] -> the route's "no row" guard.
        client = _app(devices.router, fake_db)

        r = client.post(
            "/devices/register",
            json={"fcm_token": "t", "platform": "android"},
        )

        assert r.status_code == 500


# --------------------------------------------------------------------------- #
# POST /devices/unregister  (SRS-20: drop the token on logout)
# --------------------------------------------------------------------------- #
class TestUnregisterDevice:
    def test_deletes_scoped_to_jwt_user_and_token(self, fake_db):
        client = _app(devices.router, fake_db)

        r = client.post("/devices/unregister", json={"fcm_token": "tok-abc"})

        assert r.status_code == 200
        assert r.json()["status"] == "success"
        # The delete MUST be scoped to BOTH the JWT user and the given token,
        # so a client can never drop another account's token.
        filters = fake_db.filters_for("device_tokens", "delete")
        assert ("user_id", JWT_USER) in filters
        assert ("fcm_token", "tok-abc") in filters

    def test_deleting_absent_token_is_still_success(self, fake_db):
        # Default execute -> data=[]. A re-logout / already-rotated token must
        # NOT 404 — unregister is idempotent.
        client = _app(devices.router, fake_db)
        r = client.post("/devices/unregister", json={"fcm_token": "gone"})
        assert r.status_code == 200

    def test_empty_fcm_token_is_422_and_never_writes(self, fake_db):
        client = _app(devices.router, fake_db)
        r = client.post("/devices/unregister", json={"fcm_token": ""})
        assert r.status_code == 422
        assert fake_db.recorded == []


# --------------------------------------------------------------------------- #
# POST /me/location  (SRS-23: keep the hunter's location fresh for geo-alerts)
# --------------------------------------------------------------------------- #
class TestUpdateLocation:
    def test_writes_location_and_timestamp_for_jwt_user(self, fake_db):
        fake_db.set_table_result("users", "update", data=[{"id": JWT_USER}])
        client = _app(me.router, fake_db)

        r = client.post(
            "/me/location", json={"latitude": 13.7563, "longitude": 100.5018}
        )

        assert r.status_code == 200
        payload = fake_db.payload_for("users", "update")
        # PostGIS POINT is (lng lat) — guards a lat/lon swap.
        assert payload["last_location"] == create_postgis_point(13.7563, 100.5018)
        assert payload["last_location"] == "POINT(100.5018 13.7563)"
        assert "last_location_at" in payload and payload["last_location_at"]
        # The update must be scoped to the authenticated user's row only.
        assert ("id", JWT_USER) in fake_db.filters_for("users", "update")

    def test_missing_profile_row_is_404(self, fake_db):
        # Default execute -> data=[] -> "User profile not found".
        client = _app(me.router, fake_db)
        r = client.post(
            "/me/location", json={"latitude": 1.0, "longitude": 2.0}
        )
        assert r.status_code == 404

    def test_out_of_range_latitude_is_422(self, fake_db):
        client = _app(me.router, fake_db)
        r = client.post(
            "/me/location", json={"latitude": 200.0, "longitude": 2.0}
        )
        assert r.status_code == 422


# --------------------------------------------------------------------------- #
# POST /missing-pets/  — the geo-push fan-out
# (SRS-21: push to hunters within 10 km; SRS-22: exclude the owner)
# --------------------------------------------------------------------------- #
class TestMissingPetFanout:
    _BODY = {
        "pet_name": "Luna",
        "species": "Dog",
        "characteristics": {"color": "Golden"},
        "bounty_amount": 1000.0,
        "longitude": 100.5018,
        "latitude": 13.7563,
        "last_seen_time": "2025-01-12T10:30:00Z",
        "image_url": "https://example.com/luna.jpg",
    }

    @pytest.fixture
    def env(self, monkeypatch, fake_db):
        """Patch the AI/DB insert and the notify boundary; spy BackgroundTasks."""
        seq = []

        async def fake_register(supabase, pet):
            seq.append("insert")
            return {"id": "pet-123", "pet_name": pet.pet_name}

        monkeypatch.setattr(
            missing_pets.PetService, "register_missing_pet", fake_register
        )

        notify_calls = []

        def fake_notify(*args, **kwargs):
            seq.append("notify_ran")
            notify_calls.append((args, kwargs))

        monkeypatch.setattr(missing_pets, "notify_nearby_hunters", fake_notify)

        scheduled = []
        orig_add = background.BackgroundTasks.add_task

        def spy_add(self, func, *args, **kwargs):
            seq.append("schedule")
            scheduled.append((func, args, kwargs))
            return orig_add(self, func, *args, **kwargs)

        monkeypatch.setattr(background.BackgroundTasks, "add_task", spy_add)

        client = _app(missing_pets.router, fake_db)
        return SimpleNamespace(
            client=client, db=fake_db, seq=seq,
            scheduled=scheduled, notify_calls=notify_calls,
            notify=fake_notify,
        )

    def test_schedules_notify_after_insert_via_background_task(self, env):
        r = env.client.post("/missing-pets/", json=self._BODY)

        assert r.status_code == 200
        assert r.json()["data"]["id"] == "pet-123"

        # Exactly one background task was scheduled, and it is the notify fn.
        assert len(env.scheduled) == 1
        func, args, _kwargs = env.scheduled[0]
        assert func is env.notify

        # Args: (supabase, pet_id, latitude, longitude, owner_id, radius,
        #        pet_name, species) — owner is the JWT user, id is the inserted
        # row, species/name are the user-confirmed values.
        assert args[0] is env.db
        assert args[1] == "pet-123"
        assert args[2] == 13.7563          # latitude
        assert args[3] == 100.5018         # longitude
        assert args[4] == JWT_USER         # owner from JWT, not body
        assert args[6] == "Luna"
        assert args[7] == "Dog"

    def test_fanout_runs_after_response_and_after_insert(self, env):
        env.client.post("/missing-pets/", json=self._BODY)

        # Ordering proves: insert happens first, the notify is SCHEDULED (not
        # awaited inline), and it actually runs as a background task afterward.
        assert env.seq == ["insert", "schedule", "notify_ran"]
        assert len(env.notify_calls) == 1
