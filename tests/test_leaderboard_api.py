"""
Route unit tests for the Rank List boards.

  UTC-50  GET /leaderboard/users     (MD-55, SRS-91/92)
  UTC-51  GET /leaderboard/bounties  (MD-56, SRS-93)

These two are the one documented exception to the repository seam — they are
straight ordered reads with no schema of their own, so they live in the route
handler and talk to the Supabase client directly. The seam is therefore
`get_supabase_client`, replaced through FastAPI's dependency_overrides with a
chainable double that records the query it was asked to build.

What is worth asserting is not that a list comes back. It is:

  * the caller's standing is GLOBAL, counted as "how many strictly outscore me,
    plus one" — not their index in the returned page. SRS-92 says the figure must
    read the same on every page, and a rank computed from the page would not.
  * the page carries a second sort key. Ordering by score alone lets a row shift
    between two requests when scores tie, so pagination would repeat or skip
    hunters (SRS-91, SRS-93).
  * the bounty board excludes closed searches, or it advertises rewards for pets
    that have already been found (SRS-93).
"""
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import leaderboard as lb
from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client


class _Result:
    def __init__(self, data=None, count=None):
        self.data = data
        self.count = count


class _Query:
    """Records every chained call, then hands back the queued result."""

    def __init__(self, log, result):
        self._log = log
        self._result = result

    def _record(self, name, *args, **kwargs):
        self._log.append((name, args, kwargs))
        return self

    def select(self, *a, **k):
        return self._record("select", *a, **k)

    def order(self, *a, **k):
        return self._record("order", *a, **k)

    def range(self, *a, **k):
        return self._record("range", *a, **k)

    def eq(self, *a, **k):
        return self._record("eq", *a, **k)

    def gt(self, *a, **k):
        return self._record("gt", *a, **k)

    def limit(self, *a, **k):
        return self._record("limit", *a, **k)

    def in_(self, *a, **k):
        return self._record("in_", *a, **k)

    @property
    def not_(self):
        self._log.append(("not_", (), {}))
        return self

    def execute(self):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _Supabase:
    """Serves one queued result per `.table(...)` call, in order."""

    def __init__(self, *results):
        self._results = list(results)
        self.log = []

    def table(self, name):
        self.log.append(("table", (name,), {}))
        return _Query(self.log, self._results.pop(0))

    def calls(self, op):
        return [(a, k) for n, a, k in self.log if n == op]


def _client(fake, user_id="me-1"):
    app = FastAPI()
    app.include_router(lb.router)
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[get_supabase_client] = lambda: fake
    return TestClient(app)


def _user(uid, score, name=None):
    return {
        "id": uid,
        "display_name": name or uid,
        "profile_image_url": None,
        "total_score": score,
    }


# ----------------------------------------------------------------- UTC-50 ---
class TestLeaderboardUsers:
    """UTC-50 · MD-55 `leaderboard_users` · SRS-91, SRS-92."""

    def test_tc01_ranks_run_from_one_and_follow_the_offset(self):
        page = [_user("a", 90), _user("b", 80), _user("c", 70)]
        fake = _Supabase(_Result(page), _Result([_user("me-1", 80)]), _Result(count=1))
        r = _client(fake).get("/leaderboard/users?limit=3&offset=10")
        assert r.status_code == 200
        entries = r.json()["data"]["entries"]
        # The rank shown is the row's position in the whole board, so the second
        # page must not restart at 1.
        assert [e["rank"] for e in entries] == [11, 12, 13]
        assert [e["user_id"] for e in entries] == ["a", "b", "c"]

    def test_tc02_my_standing_counts_who_outscores_me_not_my_page_position(self):
        # I am last on the page, but only two people in the whole table beat me.
        page = [_user("a", 90), _user("b", 85), _user("me-1", 80)]
        fake = _Supabase(_Result(page), _Result([_user("me-1", 80)]), _Result(count=2))
        r = _client(fake).get("/leaderboard/users")
        me = r.json()["data"]["me"]
        assert me["rank"] == 3  # 2 strictly greater, + 1
        assert me["total_score"] == 80
        # The count query is the one that decides it, and it must be strict.
        assert fake.calls("gt") == [(("total_score", 80), {})]

    def test_tc03_my_standing_is_the_same_on_a_later_page(self):
        # SRS-92: the pinned figure must not move when the page does.
        deep = [_user(f"u{i}", 10) for i in range(3)]
        fake = _Supabase(_Result(deep), _Result([_user("me-1", 80)]), _Result(count=2))
        r = _client(fake).get("/leaderboard/users?offset=40")
        assert r.json()["data"]["me"]["rank"] == 3

    def test_tc04_a_hunter_who_has_never_scored_reads_as_zero_not_null(self):
        fake = _Supabase(
            _Result([_user("a", None)]),
            _Result([_user("me-1", None)]),
            _Result(count=0),
        )
        r = _client(fake).get("/leaderboard/users")
        assert r.json()["data"]["entries"][0]["total_score"] == 0
        assert r.json()["data"]["me"]["total_score"] == 0
        assert r.json()["data"]["me"]["rank"] == 1

    def test_tc05_the_page_carries_a_tiebreak_after_the_score(self):
        # SRS-91: without the second key, equal scores can shift between requests
        # and a hunter is shown twice or not at all across page boundaries.
        fake = _Supabase(_Result([]), _Result([_user("me-1", 0)]), _Result(count=0))
        _client(fake).get("/leaderboard/users")
        orders = fake.calls("order")
        assert orders[0] == (("total_score",), {"desc": True})
        assert orders[1] == (("id",), {})

    def test_tc06_a_database_failure_is_reported_as_500(self):
        fake = _Supabase(RuntimeError("connection reset"))
        r = _client(fake).get("/leaderboard/users")
        assert r.status_code == 500
        assert "leaderboard" in r.json()["detail"].lower()

    @pytest.mark.parametrize("qs", ["limit=0", "limit=101", "offset=-1"])
    def test_tc07_out_of_range_paging_is_refused_before_any_query(self, qs):
        fake = _Supabase()  # nothing queued: a query would raise IndexError
        r = _client(fake).get(f"/leaderboard/users?{qs}")
        assert r.status_code == 422
        assert fake.log == []

    def test_tc08_the_board_requires_a_signed_in_caller(self):
        app = FastAPI()
        app.include_router(lb.router)
        app.dependency_overrides[get_supabase_client] = lambda: _Supabase()
        # Auth is left un-overridden, so the real dependency runs and rejects.
        assert TestClient(app).get("/leaderboard/users").status_code == 401


# ----------------------------------------------------------------- UTC-51 ---
class TestLeaderboardBounties:
    """UTC-51 · MD-56 `leaderboard_bounties` · SRS-93."""

    @staticmethod
    def _pet(pid, amount):
        return {
            "id": pid,
            "pet_name": pid.upper(),
            "image_url": None,
            "bounty_amount": amount,
        }

    def test_tc01_ranks_run_from_one_and_follow_the_offset(self):
        rows = [self._pet("p1", 5000), self._pet("p2", 3000)]
        fake = _Supabase(_Result(rows))
        r = _client(fake).get("/leaderboard/bounties?limit=2&offset=4")
        assert r.status_code == 200
        entries = r.json()["data"]["entries"]
        assert [e["rank"] for e in entries] == [5, 6]
        assert [e["pet_id"] for e in entries] == ["p1", "p2"]

    def test_tc02_closed_searches_are_excluded(self):
        # A board that advertised a bounty on a pet already found would send
        # hunters after an animal that is home.
        fake = _Supabase(_Result([]))
        _client(fake).get("/leaderboard/bounties")
        assert ("not_", (), {}) in fake.log
        assert fake.calls("in_") == [(("status", ["Found", "Resolved"]), {})]

    def test_tc03_the_page_carries_a_tiebreak_after_the_bounty(self):
        fake = _Supabase(_Result([]))
        _client(fake).get("/leaderboard/bounties")
        orders = fake.calls("order")
        assert orders[0] == (("bounty_amount",), {"desc": True})
        assert orders[1] == (("id",), {})

    def test_tc04_a_missing_bounty_reads_as_zero_and_always_as_a_number(self):
        fake = _Supabase(_Result([self._pet("p1", None), self._pet("p2", "250.50")]))
        entries = _client(fake).get("/leaderboard/bounties").json()["data"]["entries"]
        assert entries[0]["bounty_amount"] == 0.0
        assert entries[1]["bounty_amount"] == 250.50
        assert all(isinstance(e["bounty_amount"], float) for e in entries)

    def test_tc05_a_database_failure_is_reported_as_500(self):
        fake = _Supabase(RuntimeError("connection reset"))
        r = _client(fake).get("/leaderboard/bounties")
        assert r.status_code == 500

    def test_tc06_the_board_requires_a_signed_in_caller(self):
        app = FastAPI()
        app.include_router(lb.router)
        app.dependency_overrides[get_supabase_client] = lambda: _Supabase()
        assert TestClient(app).get("/leaderboard/bounties").status_code == 401
