"""Unit tests for the Feature 6 bounty routes and service.

The SQL functions are tested against a real database in
tests/integration/test_bounty_settlement.py. These cover what sits above them:
the mapping of their RAISE messages to HTTP statuses, the owner's and hunter's
view, and that each route passes the caller's id from the JWT, never from the
body. Feature 6 has no UTC numbers yet; they are issued in the Phase 6 test plan.
"""
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import admin as admin_api
from app.api import bounty as bounty_api
from app.core.auth import get_current_user_id, require_admin
from app.core.database import get_supabase_client
from app.repositories.bounty_repository import BountyConflict, BountyNotFound, BountyRepository
from app.services.bounty_service import BountyService, translate_db_error
from app.services.promptpay import is_thai_mobile


def run(coro):
    import asyncio
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# translate_db_error
# --------------------------------------------------------------------------- #
class TestTranslateDbError:
    @pytest.mark.parametrize("msg, kind", [
        ("Missing pet x not found or not owned by you", BountyNotFound),
        ("Payment x not found", BountyNotFound),
        ("Missing pet x already has a payment awaiting confirmation or confirmed", BountyConflict),
        ("Missing pet x is not awaiting a bounty payment", BountyConflict),
        ("Missing pet x is not unpaid", BountyConflict),
        ("The dispute window for payment x has closed", BountyConflict),
        ("Payment x is already Verified", BountyConflict),
        ("A payment is not accepted without a slip image", ValueError),
    ])
    def test_known_messages_map_to_their_kind(self, msg, kind):
        assert type(translate_db_error(Exception(msg))) is kind

    def test_an_unknown_error_is_left_alone(self):
        e = RuntimeError("connection reset")
        assert translate_db_error(e) is e


class TestIsThaiMobile:
    def test_mobile_numbers(self):
        assert is_thai_mobile("0812345678")
        assert is_thai_mobile("061-234-5678")

    def test_not_mobile_numbers(self):
        for n in (None, "", "021234567", "1234567890123", "0812345"):
            assert not is_thai_mobile(n)


# --------------------------------------------------------------------------- #
# BountyService.get_view
# --------------------------------------------------------------------------- #
def _repo(*, status="Found", bounty=2000, payment=None, phone="0812345678"):
    r = MagicMock(spec=BountyRepository)
    r.get_pet.return_value = {"id": "p1", "owner_id": "owner-1", "pet_name": "Mali",
                              "image_url": "http://img", "status": status,
                              "bounty_amount": bounty, "found_at": "t",
                              "unpaid_referred_at": None, "bounty_unpaid": False}
    r.get_final_sighting.return_value = {"sighting_id": "s1", "hunter_id": "hunter-1"}
    r.get_latest_payment.return_value = payment
    r.get_user_phone.return_value = phone
    r.get_username.return_value = "Khun B"
    return r


class TestGetView:
    def test_an_owner_who_owes_sees_the_hunters_promptpay(self):
        v = run(BountyService(_repo()).get_view("p1", "owner-1"))
        assert (v["role"], v["unpaid"], v["hunter_promptpay"]) == ("owner", True, "0812345678")
        assert v["pet_name"] == "Mali"
        assert v["hunter_username"] == "Khun B"

    def test_a_phone_that_is_not_a_mobile_number_is_withheld(self):
        v = run(BountyService(_repo(phone="021234567")).get_view("p1", "owner-1"))
        assert v["hunter_promptpay"] is None

    def test_once_a_payment_awaits_the_number_is_no_longer_shown(self):
        repo = _repo(payment={"id": "t1", "status": "Pending_Verification"})
        v = run(BountyService(repo).get_view("p1", "owner-1"))
        assert (v["unpaid"], v["hunter_promptpay"]) == (False, None)
        repo.get_user_phone.assert_not_called()

    def test_a_rejected_payment_leaves_the_report_unpaid(self):
        repo = _repo(payment={"id": "t1", "status": "Rejected"})
        v = run(BountyService(repo).get_view("p1", "owner-1"))
        assert v["unpaid"] is True

    def test_the_hunter_sees_the_payment_but_not_their_own_number(self):
        repo = _repo(payment={"id": "t1", "status": "Pending_Verification"})
        v = run(BountyService(repo).get_view("p1", "hunter-1"))
        assert (v["role"], v["hunter_promptpay"]) == ("hunter", None)
        assert v["payment"]["id"] == "t1"

    def test_anyone_else_reads_not_found(self):
        with pytest.raises(BountyNotFound):
            run(BountyService(_repo()).get_view("p1", "stranger"))

    def test_an_unknown_report_reads_not_found(self):
        repo = _repo()
        repo.get_pet.return_value = None
        with pytest.raises(BountyNotFound):
            run(BountyService(repo).get_view("p1", "owner-1"))


class TestServiceValidation:
    def test_an_unknown_hunter_answer_is_a_400_before_the_database(self):
        repo = _repo()
        with pytest.raises(ValueError):
            run(BountyService(repo).hunter_answer("t1", "hunter-1", "Maybe"))
        repo.hunter_answer.assert_not_called()

    def test_answers_are_case_insensitive(self):
        repo = _repo()
        repo.hunter_answer.return_value = {"status": "Verified"}
        run(BountyService(repo).hunter_answer("t1", "hunter-1", "confirm"))
        repo.hunter_answer.assert_called_once_with("t1", "hunter-1", "Confirm")

    def test_an_unknown_admin_decision_is_a_400(self):
        with pytest.raises(ValueError):
            run(BountyService(_repo()).admin_rule("t1", "admin-1", "Approve"))


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@pytest.fixture
def service():
    return MagicMock(spec=BountyService)


@pytest.fixture
def client(service):
    app = FastAPI()
    app.include_router(bounty_api.router)
    app.include_router(admin_api.router)
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    app.dependency_overrides[require_admin] = lambda: "admin-1"
    app.dependency_overrides[get_supabase_client] = lambda: MagicMock()
    app.dependency_overrides[bounty_api.get_bounty_service] = lambda: service
    return TestClient(app)


def _async(value=None, exc=None):
    async def f(*a, **k):
        if exc:
            raise exc
        return value
    return f


class TestRoutes:
    def test_submit_passes_the_caller_as_owner(self, client, service):
        service.submit_payment.side_effect = _async({"payment_id": "t1"})
        r = client.post("/bounty/p1/payment", json={"slip_image_url": "u", "reference_no": "R"})
        assert r.status_code == 200
        service.submit_payment.assert_called_once_with("p1", "user-1", "u", "R")

    def test_submit_without_a_slip_is_rejected_by_the_schema(self, client, service):
        r = client.post("/bounty/p1/payment", json={"slip_image_url": ""})
        assert r.status_code == 422
        service.submit_payment.assert_not_called()

    @pytest.mark.parametrize("exc, code", [
        (BountyNotFound("x"), 404), (BountyConflict("x"), 409),
        (ValueError("x"), 400), (RuntimeError("x"), 500),
    ])
    def test_errors_map_to_statuses(self, client, service, exc, code):
        service.submit_payment.side_effect = _async(exc=exc)
        r = client.post("/bounty/p1/payment", json={"slip_image_url": "u"})
        assert r.status_code == code

    def test_answer_passes_the_caller_as_hunter(self, client, service):
        service.hunter_answer.side_effect = _async({"status": "Verified"})
        r = client.post("/bounty/payments/t1/answer", json={"answer": "Confirm"})
        assert r.status_code == 200
        service.hunter_answer.assert_called_once_with("t1", "user-1", "Confirm")

    def test_refer_unpaid_passes_the_caller(self, client, service):
        service.refer_unpaid.side_effect = _async({"referred": True})
        assert client.post("/bounty/p1/unpaid").status_code == 200
        service.refer_unpaid.assert_called_once_with("p1", "user-1")

    def test_view(self, client, service):
        service.get_view.side_effect = _async({"role": "owner"})
        r = client.get("/bounty/p1")
        assert r.status_code == 200 and r.json()["data"]["role"] == "owner"

    def test_admin_queue(self, client, service):
        service.ruling_queue.side_effect = _async([{"kind": "disputed"}])
        r = client.get("/admin/bounty-queue")
        assert r.status_code == 200 and r.json()["data"] == [{"kind": "disputed"}]

    def test_admin_ruling_records_the_admin(self, client, service):
        service.admin_rule.side_effect = _async({"status": "Verified"})
        r = client.post("/admin/bounty-payments/t1/ruling", json={"decision": "Confirm"})
        assert r.status_code == 200
        service.admin_rule.assert_called_once_with("t1", "admin-1", "Confirm")

    def test_a_rejection_notifies_the_owner(self, client, service, monkeypatch):
        sent = MagicMock(return_value=1)
        monkeypatch.setattr(admin_api, "send_to_users", sent)
        service.admin_rule.side_effect = _async({"status": "Rejected", "owner_id": "owner-1"})
        r = client.post("/admin/bounty-payments/t1/ruling", json={"decision": "Reject"})
        assert r.status_code == 200
        assert sent.call_args.args[1] == ["owner-1"]

    def test_admin_close_unpaid(self, client, service):
        service.admin_close_unpaid.side_effect = _async({"pet_status": "Resolved"})
        r = client.post("/admin/missing-pets/p1/close-unpaid")
        assert r.status_code == 200
        service.admin_close_unpaid.assert_called_once_with("p1", "admin-1")


class TestMine:
    def test_mine_is_routed_before_the_pet_id(self, client, service):
        service.bounties_for_hunter.side_effect = _async([{"pet_id": "p1"}])
        r = client.get("/bounty/mine")
        assert r.status_code == 200
        service.bounties_for_hunter.assert_called_once_with("user-1")
        service.get_view.assert_not_called()


class TestNoHunterOwesNothing:
    def test_a_found_report_with_no_returning_hunter_is_not_unpaid(self):
        """Regression (code review 2026-10-08): reports hand-closed before
        Feature 6 are "Found" with no hunter. They must not show a payment
        card for someone who does not exist."""
        repo = _repo()
        repo.get_final_sighting.return_value = None
        v = run(BountyService(repo).get_view("p1", "owner-1"))
        assert (v["unpaid"], v["hunter_promptpay"]) == (False, None)
