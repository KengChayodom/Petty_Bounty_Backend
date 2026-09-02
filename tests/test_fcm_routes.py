"""
Route unit tests for the FCM geo-push surface (UTC-07, UTC-08, UTC-09, UTC-11):
  * POST /devices/register   — upsert keyed on fcm_token (re-registration reassigns)  [SRS-19]
  * POST /devices/unregister — drop the caller's token on logout                       [SRS-20]
  * POST /me/location        — write last_location + last_location_at for the JWT user [SRS-26]
  * POST /missing-pets/      — fan out notify_nearby_hunters via BackgroundTasks  [SRS-24, SRS-25]

Progress-I SRS traceability: SRS-19 (TestRegisterDevice), SRS-20 (TestUnregisterDevice),
SRS-26 (TestUpdateLocation), SRS-24 + SRS-25 (TestMissingPetFanout).

Boundary rule (per db-testing-seams): the auth dependency and the repository
ports are the boundaries; both are replaced via FastAPI dependency_overrides
with MagicMock(spec=<Repo>) — never a hand-rolled Supabase client. We assert on
the payload/arguments the route hands the repo (scoped to the JWT user) and on
the response, never on incidental plumbing. DB-engine details the adapter owns
(the upsert on-conflict key, the location timestamp) are verified in the adapter
integration suite, not here.
"""
import starlette.background as background
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import devices, me, missing_pets
from app.api.devices import get_device_token_repository
from app.api.me import get_user_repository
from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client
from app.repositories.device_token_repository import DeviceTokenRepository
from app.repositories.user_repository import UserRepository
from app.utils.postgis import create_postgis_point

JWT_USER = "jwt-user-1111"


def _client(router, repo_dep, repo):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user_id] = lambda: JWT_USER
    app.dependency_overrides[repo_dep] = lambda: repo
    return TestClient(app)


# --------------------------------------------------------------------------- #
# POST /devices/register  (SRS-19: capture/store the FCM token on login)
# --------------------------------------------------------------------------- #
class TestRegisterDevice:
    def test_upserts_with_jwt_user(self):
        repo = MagicMock(spec=DeviceTokenRepository)
        repo.upsert_device_token.return_value = {
            "id": "row-1", "user_id": JWT_USER,
            "fcm_token": "tok-abc", "platform": "android",
        }
        client = _client(devices.router, get_device_token_repository, repo)

        r = client.post(
            "/devices/register",
            json={"fcm_token": "tok-abc", "platform": "android"},
        )

        assert r.status_code == 200
        assert r.json()["status"] == "success"
        repo.upsert_device_token.assert_called_once()
        row = repo.upsert_device_token.call_args.args[0]
        assert row["fcm_token"] == "tok-abc"
        assert row["platform"] == "android"
        assert row["user_id"] == JWT_USER
        assert "updated_at" in row

    def test_user_id_comes_from_jwt_not_request_body(self):
        repo = MagicMock(spec=DeviceTokenRepository)
        repo.upsert_device_token.return_value = {"id": "row-1"}
        client = _client(devices.router, get_device_token_repository, repo)

        # A client trying to register a token for someone else must be ignored.
        client.post(
            "/devices/register",
            json={"fcm_token": "t", "platform": "ios", "user_id": "attacker"},
        )

        assert repo.upsert_device_token.call_args.args[0]["user_id"] == JWT_USER

    def test_invalid_platform_is_422_and_never_writes(self):
        repo = MagicMock(spec=DeviceTokenRepository)
        client = _client(devices.router, get_device_token_repository, repo)

        r = client.post(
            "/devices/register",
            json={"fcm_token": "t", "platform": "windows"},
        )

        assert r.status_code == 422
        repo.upsert_device_token.assert_not_called()

    def test_empty_fcm_token_is_422(self):
        repo = MagicMock(spec=DeviceTokenRepository)
        client = _client(devices.router, get_device_token_repository, repo)
        r = client.post(
            "/devices/register",
            json={"fcm_token": "", "platform": "android"},
        )
        assert r.status_code == 422

    def test_upsert_returning_no_row_is_500(self):
        repo = MagicMock(spec=DeviceTokenRepository)
        repo.upsert_device_token.return_value = None  # the route's "no row" guard
        client = _client(devices.router, get_device_token_repository, repo)

        r = client.post(
            "/devices/register",
            json={"fcm_token": "t", "platform": "android"},
        )

        assert r.status_code == 500


# --------------------------------------------------------------------------- #
# POST /devices/unregister  (SRS-20: drop the token on logout)
# --------------------------------------------------------------------------- #
class TestUnregisterDevice:
    def test_deletes_scoped_to_jwt_user_and_token(self):
        repo = MagicMock(spec=DeviceTokenRepository)
        client = _client(devices.router, get_device_token_repository, repo)

        r = client.post("/devices/unregister", json={"fcm_token": "tok-abc"})

        assert r.status_code == 200
        assert r.json()["status"] == "success"
        # The delete MUST be scoped to BOTH the JWT user and the given token,
        # so a client can never drop another account's token.
        repo.delete_device_token.assert_called_once_with(JWT_USER, "tok-abc")

    def test_deleting_absent_token_is_still_success(self):
        # A re-logout / already-rotated token must NOT 404 — unregister is
        # idempotent, so the repo returning nothing is still a success.
        repo = MagicMock(spec=DeviceTokenRepository)
        client = _client(devices.router, get_device_token_repository, repo)
        r = client.post("/devices/unregister", json={"fcm_token": "gone"})
        assert r.status_code == 200

    def test_empty_fcm_token_is_422_and_never_writes(self):
        repo = MagicMock(spec=DeviceTokenRepository)
        client = _client(devices.router, get_device_token_repository, repo)
        r = client.post("/devices/unregister", json={"fcm_token": ""})
        assert r.status_code == 422
        repo.delete_device_token.assert_not_called()


# --------------------------------------------------------------------------- #
# POST /me/location  (SRS-26: keep the hunter's location fresh for geo-alerts)
# --------------------------------------------------------------------------- #
class TestUpdateLocation:
    def test_writes_location_for_jwt_user(self):
        repo = MagicMock(spec=UserRepository)
        repo.update_last_location.return_value = {"id": JWT_USER}
        client = _client(me.router, get_user_repository, repo)

        r = client.post(
            "/me/location", json={"latitude": 13.7563, "longitude": 100.5018}
        )

        assert r.status_code == 200
        repo.update_last_location.assert_called_once()
        user_id, location_point = repo.update_last_location.call_args.args
        # The update must be scoped to the authenticated user's row only.
        assert user_id == JWT_USER
        # PostGIS POINT is (lng lat) — guards a lat/lon swap.
        assert location_point == create_postgis_point(13.7563, 100.5018)
        assert location_point == "POINT(100.5018 13.7563)"

    def test_missing_profile_row_is_404(self):
        repo = MagicMock(spec=UserRepository)
        repo.update_last_location.return_value = None  # no row matched the id
        client = _client(me.router, get_user_repository, repo)
        r = client.post(
            "/me/location", json={"latitude": 1.0, "longitude": 2.0}
        )
        assert r.status_code == 404

    def test_out_of_range_latitude_is_422(self):
        repo = MagicMock(spec=UserRepository)
        client = _client(me.router, get_user_repository, repo)
        r = client.post(
            "/me/location", json={"latitude": 200.0, "longitude": 2.0}
        )
        assert r.status_code == 422


# --------------------------------------------------------------------------- #
# POST /missing-pets/  — the geo-push fan-out
# (SRS-24: push to hunters within 10 km; SRS-25: exclude the owner)
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
    def env(self, monkeypatch):
        """Patch the AI/DB insert and the notify boundary; spy BackgroundTasks.

        The register + notify boundaries are monkeypatched wholesale, so the
        client this route is handed is an opaque sentinel — the fan-out just
        needs to be scheduled with it, not to be a real DB.
        """
        seq = []
        db = object()  # opaque client sentinel handed to the background task

        async def fake_register(repo, pet):
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

        app = FastAPI()
        app.include_router(missing_pets.router)
        app.dependency_overrides[get_current_user_id] = lambda: JWT_USER
        app.dependency_overrides[get_supabase_client] = lambda: db
        client = TestClient(app)
        return SimpleNamespace(
            client=client, db=db, seq=seq,
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

        # Args: (client, pet_id, latitude, longitude, owner_id, radius,
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
