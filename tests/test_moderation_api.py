"""
Route tests for the Progress-II moderation endpoints.

The service tests above verify the decisions; these verify the one thing that
lives only in the route layer and that every MD entry states explicitly — the
HTTP status each domain outcome becomes. Getting 404 vs 409 vs 400 wrong is
invisible to a service test and very visible to the admin panel (UD-16 [E1]
needs 409 to show "already moderated" instead of a generic failure).

Seams: the auth dependencies (`get_current_user_id` / `require_admin`) and the
repository ports, swapped through FastAPI dependency_overrides — the same
pattern as tests/test_me_profile_api.py.

`POST /reports`   — MD-39: 400 bad reason, 404 unknown sighting, 500 failure
`PATCH /admin/reports/{id}` — MD-40: 400 / 404 / 409 / 500
`DELETE /admin/missing-pets/{id}` — MD-38: 404 / 500
"""
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import admin as admin_api
from app.api import missing_pets as pets_api
from app.api import reports as reports_api
from app.core.auth import get_current_user_id, require_admin
from app.core.database import get_supabase_client
from app.repositories.missing_pet_repository import MissingPetRepository
from app.repositories.pagination import Page
from app.repositories.report_repository import (
    ReportAlreadyModerated,
    ReportNotFound,
)
from app.services.admin_service import AdminService
from app.services.report_service import ReportService


def _admin_client(service=None, pet_repo=None):
    app = FastAPI()
    app.include_router(admin_api.router)
    app.dependency_overrides[require_admin] = lambda: "admin-1"
    if service is not None:
        app.dependency_overrides[admin_api.get_admin_service] = lambda: service
    if pet_repo is not None:
        app.dependency_overrides[
            admin_api.get_missing_pet_repository
        ] = lambda: pet_repo
    return TestClient(app)


def _reports_client(service):
    app = FastAPI()
    app.include_router(reports_api.router)
    app.dependency_overrides[get_current_user_id] = lambda: "r1"
    app.dependency_overrides[reports_api.get_report_service] = lambda: service
    return TestClient(app)


def _service():
    """An AdminService double. spec= keeps the route honest: a renamed service
    method fails here instead of silently returning a Mock."""
    return MagicMock(spec=AdminService)


async def _raise(exc):
    raise exc


def _async_raises(exc):
    return lambda *a, **k: _raise(exc)


def _async_returns(value):
    async def _inner(*a, **k):
        return value
    return _inner


# --------------------------------------------------------------------------- #
# POST /reports — MD-39
# --------------------------------------------------------------------------- #
class TestFlagSightingRoute:
    def test_bad_reason_yields_400(self):
        service = MagicMock(spec=ReportService)
        service.flag_sighting = _async_raises(ValueError("reason must be one of"))
        r = _reports_client(service).post(
            "/reports", json={"sighting_id": "s1", "reason": "Ugly"},
        )
        assert r.status_code == 400

    def test_unknown_sighting_yields_404(self):
        from app.repositories.report_repository import FlagTargetNotFound

        service = MagicMock(spec=ReportService)
        service.flag_sighting = _async_raises(FlagTargetNotFound("ghost"))
        r = _reports_client(service).post(
            "/reports", json={"sighting_id": "ghost", "reason": "Spam"},
        )
        # FlagTargetNotFound subclasses ValueError, so this also pins the
        # except-clause ORDER in the route: 404 must win over the 400 handler.
        assert r.status_code == 404

    def test_success_returns_the_flag(self):
        service = MagicMock(spec=ReportService)
        service.flag_sighting = _async_returns(
            {"id": "r1", "status": "Pending"}
        )
        r = _reports_client(service).post(
            "/reports", json={"sighting_id": "s1", "reason": "Spam"},
        )
        assert r.status_code == 200
        assert r.json()["data"]["status"] == "Pending"

    def test_repository_failure_yields_500(self):
        service = MagicMock(spec=ReportService)
        service.flag_sighting = _async_raises(RuntimeError("db down"))
        r = _reports_client(service).post(
            "/reports", json={"sighting_id": "s1", "reason": "Spam"},
        )
        assert r.status_code == 500


# --------------------------------------------------------------------------- #
# GET /admin/reports — MD-51
# --------------------------------------------------------------------------- #
class TestListReportsRoute:
    def test_forwards_filter_and_pagination(self):
        service = _service()
        service.list_reports = _async_returns(Page([{"id": "r1"}], 1))
        r = _admin_client(service).get(
            "/admin/reports?status=Pending&limit=5&offset=10"
        )
        assert r.status_code == 200
        assert r.json()["data"] == {
            "items": [{"id": "r1"}], "total": 1, "limit": 5, "offset": 10,
        }

    def test_page_carries_the_queue_depth_and_the_window_asked_for(self):
        """The console draws numbered pages from `total`, and echoing back the
        limit/offset it was served keeps a stale response from being read as a
        page it is not."""
        service = _service()
        service.list_reports = _async_returns(
            Page([{"id": f"r{i}"} for i in range(5)], 143),
        )
        body = _admin_client(service).get(
            "/admin/reports?limit=5&offset=20"
        ).json()["data"]
        assert body["total"] == 143
        assert body["limit"] == 5 and body["offset"] == 20
        assert len(body["items"]) == 5

    def test_static_path_is_not_captured_by_the_report_id_route(self):
        """`/admin/reports` must reach the listing, not be read as a
        report_id by `PATCH /admin/reports/{report_id}`. They differ by method
        so ordering is safe, but a later GET /reports/{id} would break this."""
        service = _service()
        service.list_reports = _async_returns(Page([], 0))
        r = _admin_client(service).get("/admin/reports")
        assert r.status_code == 200
        assert r.json()["data"]["items"] == []

    @pytest.mark.parametrize("exc,expected", [
        (ValueError("status must be one of"), 400),
        (RuntimeError("db down"), 500),
    ])
    def test_status_mapping(self, exc, expected):
        """An unknown filter is the caller's error (400), not the server's."""
        service = _service()
        service.list_reports = _async_raises(exc)
        r = _admin_client(service).get("/admin/reports?status=Banned")
        assert r.status_code == expected

    @pytest.mark.parametrize("query", [
        "?limit=0", "?limit=201", "?offset=-1",
    ])
    def test_pagination_bounds_are_enforced_by_the_route(self, query):
        service = _service()
        service.list_reports = _async_returns(Page([], 0))
        assert _admin_client(service).get(
            f"/admin/reports{query}"
        ).status_code == 422


# --------------------------------------------------------------------------- #
# PATCH /admin/reports/{report_id} — MD-40
# --------------------------------------------------------------------------- #
class TestReviewReportRoute:
    @pytest.mark.parametrize("exc,expected", [
        (ReportNotFound("r1"), 404),
        (ReportAlreadyModerated("r1", "Dismissed"), 409),
        (ValueError("decision must be one of"), 400),
        (RuntimeError("db down"), 500),
    ])
    def test_status_mapping(self, exc, expected):
        """Both domain errors subclass ValueError, so this also pins the
        except-clause order: 404 and 409 must be caught before the 400."""
        service = _service()
        service.review_report = _async_raises(exc)
        r = _admin_client(service).patch(
            "/admin/reports/r1", json={"decision": "Dismissed"},
        )
        assert r.status_code == expected

    def test_success_returns_the_outcome(self):
        service = _service()
        service.review_report = _async_returns({
            "report": {"id": "r1", "status": "Reviewed_Penalty"},
            "sighting_dismissed": True,
            "penalty": {"points": 10, "points_applied": 10,
                        "total_score_after": 5},
        })
        r = _admin_client(service).patch(
            "/admin/reports/r1", json={"decision": "Reviewed and banned"},
        )
        assert r.status_code == 200
        body = r.json()["data"]
        assert body["sighting_dismissed"] is True
        assert body["penalty"]["points_applied"] == 10

    def test_penalty_points_reach_the_service(self):
        """The admin's explicit ruling must survive the route, including 0 —
        which means "uphold, withdraw the sighting, charge nothing" and is the
        value a truthiness check would silently turn back into the default."""
        service = _service()
        # Set return_value rather than replacing the attribute: spec= already
        # made this an AsyncMock, and replacing it would throw away call_args.
        service.review_report.return_value = {
            "report": {"id": "r1", "status": "Reviewed_Penalty"},
            "sighting_dismissed": True, "penalty": None,
        }
        r = _admin_client(service).patch(
            "/admin/reports/r1",
            json={"decision": "Uphold and Penalise User", "penalty_points": 0},
        )
        assert r.status_code == 200
        assert service.review_report.call_args.kwargs["penalty_points"] == 0

    def test_penalty_points_default_to_none_when_omitted(self):
        """Omitting the field must not become 0 — that is the difference
        between "charge the per-reason default" and "charge nothing"."""
        service = _service()
        service.review_report.return_value = {
            "report": {"id": "r1", "status": "Reviewed_Penalty"},
            "sighting_dismissed": True, "penalty": None,
        }
        r = _admin_client(service).patch(
            "/admin/reports/r1", json={"decision": "Reviewed_Penalty"},
        )
        assert r.status_code == 200
        assert service.review_report.call_args.kwargs["penalty_points"] is None

    def test_out_of_range_penalty_is_a_400_not_a_422(self):
        """Consistent with `decision`: this endpoint reports bad input as 400,
        so the bound lives in moderation_logic and not in the Pydantic field."""
        service = _service()
        service.review_report = _async_raises(
            ValueError("penalty_points must not exceed 100")
        )
        r = _admin_client(service).patch(
            "/admin/reports/r1",
            json={"decision": "Reviewed_Penalty", "penalty_points": 5000},
        )
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
# GET/DELETE /admin/missing-pets — MD-37, MD-38
# --------------------------------------------------------------------------- #
class TestAdminMissingPetRoutes:
    def test_list_forwards_the_status_filter(self):
        repo = MagicMock(spec=MissingPetRepository)
        repo.list_all.return_value = Page([{"id": "p1", "sighting_count": 0}], 1)
        r = _admin_client(pet_repo=repo).get(
            "/admin/missing-pets?status=Searching&limit=5&offset=0"
        )
        assert r.status_code == 200
        repo.list_all.assert_called_once_with("Searching", None, limit=10000, offset=0)
        assert r.json()["data"]["items"][0]["id"] == "p1"
        assert r.json()["data"]["limit"] == 5
        assert r.json()["data"]["offset"] == 0

    def test_list_reports_the_total_for_the_filter_not_the_page(self):
        """Page one of a larger result must say how large it is, or the console
        has no way to know pages two and three exist."""
        repo = MagicMock(spec=MissingPetRepository)
        repo.list_all.return_value = Page(
            [{"id": f"p{i}"} for i in range(20)], 57,
        )
        body = _admin_client(pet_repo=repo).get(
            "/admin/missing-pets?limit=20&offset=0"
        ).json()["data"]
        assert body["total"] == 57 and len(body["items"]) == 20

    def test_list_without_filter_passes_none(self):
        repo = MagicMock(spec=MissingPetRepository)
        repo.list_all.return_value = Page([], 0)
        r = _admin_client(pet_repo=repo).get("/admin/missing-pets")
        assert r.status_code == 200
        assert repo.list_all.call_args.args[0] is None

    def test_delete_unknown_pet_yields_404(self):
        repo = MagicMock(spec=MissingPetRepository)
        repo.remove.return_value = None
        r = _admin_client(pet_repo=repo).delete("/admin/missing-pets/ghost")
        assert r.status_code == 404

    def test_delete_removes_and_returns_the_row(self):
        repo = MagicMock(spec=MissingPetRepository)
        repo.remove.return_value = {"id": "p1"}
        r = _admin_client(pet_repo=repo).delete("/admin/missing-pets/p1")
        assert r.status_code == 200
        assert r.json()["data"] == {"id": "p1"}

    def test_delete_failure_yields_500(self):
        repo = MagicMock(spec=MissingPetRepository)
        repo.remove.side_effect = RuntimeError("db down")
        r = _admin_client(pet_repo=repo).delete("/admin/missing-pets/p1")
        assert r.status_code == 500



# --------------------------------------------------------------------------- #
# GET /missing-pets/me — MD-34
# --------------------------------------------------------------------------- #
class TestMyReportsRoute:
    def _client(self, repo, user_id="u1"):
        app = FastAPI()
        app.include_router(pets_api.router)
        app.dependency_overrides[get_current_user_id] = lambda: user_id
        app.dependency_overrides[get_supabase_client] = lambda: MagicMock()
        return app, TestClient(app)

    def test_me_is_not_captured_by_the_pet_id_route(self, monkeypatch):
        """The reason /me is declared before /{pet_id}: with the order
        reversed, "me" is read as a pet id and this endpoint never runs."""
        seen = {}

        async def _fake_get_my(repo, owner_id):
            seen["owner_id"] = owner_id
            return [{"id": "p1", "owner_id": owner_id}]

        monkeypatch.setattr(
            pets_api.PetService, "get_my_missing_pets", _fake_get_my,
        )
        _, client = self._client(MagicMock())

        r = client.get("/missing-pets/me")

        assert r.status_code == 200
        assert seen["owner_id"] == "u1"      # identity came from the JWT
        assert r.json()["data"][0]["id"] == "p1"

    def test_failure_yields_500(self, monkeypatch):
        async def _boom(repo, owner_id):
            raise RuntimeError("db down")

        monkeypatch.setattr(pets_api.PetService, "get_my_missing_pets", _boom)
        _, client = self._client(MagicMock())
        assert client.get("/missing-pets/me").status_code == 500
