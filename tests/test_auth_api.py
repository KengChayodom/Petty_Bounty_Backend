"""
Route unit test for GET /auth/me  (UTC-06, MD-08, SRS-23).

SRS-23 = allow authenticated users to retrieve their own profile securely: the
read is scoped to the JWT user id on the `users` table.

Boundary rule: the auth dependency and the Supabase client are the boundaries,
replaced via FastAPI dependency_overrides. The route reads the caller's profile
from `users` via a `.single()` terminal, so the DB double here exposes exactly
that terminal. We assert on the returned body and on the fact that the read was
scoped to the JWT user id.
"""
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import auth as auth_api
from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client


# --------------------------------------------------------------------------- #
# Minimal Supabase double exposing the select → eq → single → execute chain.
# --------------------------------------------------------------------------- #
class _SingleQuery:
    def __init__(self, result=None, raises=False):
        self._result = result
        self._raises = raises
        self.filters = []          # [(col, value), ...] captured from .eq(...)

    def select(self, *_a, **_k):
        return self

    def eq(self, *args, **_k):
        self.filters.append(tuple(args))
        return self

    def single(self):
        return self

    def execute(self):
        # supabase-py's `.single()` raises when there is no exactly-one row.
        if self._raises:
            raise Exception("PGRST116: no rows returned")
        return SimpleNamespace(data=self._result)


class _DB:
    def __init__(self, result=None, raises=False):
        self.table_name = None
        self.q = _SingleQuery(result=result, raises=raises)

    def table(self, name):
        self.table_name = name
        return self.q


def _client(db, user_id="user-1"):
    app = FastAPI()
    app.include_router(auth_api.router)
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[get_supabase_client] = lambda: db
    return TestClient(app)


class TestGetMe:
    def test_profile_returned_for_caller(self):
        profile = {
            "id": "user-1",
            "display_name": "Jamal Johnson",
            "phone": "0812345678",
            "role": "user",
            "total_score": 0,
            "profile_image_url": None,
            "created_at": "2026-01-01T00:00:00Z",
        }
        db = _DB(result=profile)
        r = _client(db, user_id="user-1").get("/auth/me")

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["data"] == profile
        # The read must be scoped to the JWT user, on the users table.
        assert db.table_name == "users"
        assert ("id", "user-1") in db.q.filters

    def test_missing_profile_row_is_404(self):
        db = _DB(raises=True)
        r = _client(db, user_id="ghost").get("/auth/me")

        assert r.status_code == 404
