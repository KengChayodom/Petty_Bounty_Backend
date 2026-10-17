"""Unit tests for the Feature 6 bounty routes and service
.

The SQL functions are tested against a real database in
tests/integration/test_bounty_settlement.py. These cover what sits above them:
the mapping of their RAISE messages to HTTP statuses, the owner's and hunter's
view, and that each route passes the caller's id from the JWT.

Case selection. The cases below are the test frames of the category-partition
tables in the unit test plan, one test class per method and one test method
per frame. A frame is written only for a decision the method under test makes
itself. The error mapper the routes share is framed once, in TestSubmitPayment,
and request validation by the framework is not framed. Route-level cases stub
BountyService at the boundary and assert the HTTP status. The slip of only
spaces runs the real BountyService over a stubbed BountyRepository, so the
chain from the repository's error to the HTTP status is exercised once.
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


def _app_with(service):
    app = FastAPI()
    app.include_router(bounty_api.router)
    app.include_router(admin_api.router)
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    app.dependency_overrides[require_admin] = lambda: "admin-1"
    app.dependency_overrides[get_supabase_client] = lambda: MagicMock()
    app.dependency_overrides[bounty_api.get_bounty_service] = lambda: service
    return TestClient(app)


@pytest.fixture
def service():
    return MagicMock(spec=BountyService)


@pytest.fixture
def client(service):
    return _app_with(service)


@pytest.fixture
def sent(monkeypatch):
    """Records the push notifications the admin routes schedule."""
    s = MagicMock(return_value=1)
    monkeypatch.setattr(admin_api, "send_to_users", s)
    return s


def _async(value=None, exc=None):
    async def f(*a, **k):
        if exc:
            raise exc
        return value
    return f


# --------------------------------------------------------------------------- #
# get_bounty and the view it returns
# --------------------------------------------------------------------------- #
class TestGetBounty:
    # Route level
    def test_view(self, client, service):
        service.get_view.side_effect = _async({"role": "owner"})
        r = client.get("/bounty/p1")
        assert r.status_code == 200 and r.json()["data"]["role"] == "owner"
        service.get_view.assert_called_once_with("p1", "user-1")

    def test_not_found_from_the_service_is_a_404(self, client, service):
        service.get_view.side_effect = _async(exc=BountyNotFound("Missing pet p1 not found"))
        assert client.get("/bounty/p1").status_code == 404

    # Service level (BountyService.get_view)
    def test_an_owner_who_owes_sees_the_hunters_promptpay(self):
        v = run(BountyService(_repo()).get_view("p1", "owner-1"))
        assert (v["role"], v["unpaid"], v["hunter_promptpay"]) == ("owner", True, "0812345678")
        assert v["pet_name"] == "Mali"
        assert v["hunter_username"] == "Khun B"

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

# --------------------------------------------------------------------------- #
# submit_payment
# --------------------------------------------------------------------------- #
class TestSubmitPayment:
    def test_submit_passes_the_caller_as_owner(self, client, service):
        service.submit_payment.side_effect = _async({"payment_id": "t1"})
        r = client.post("/bounty/p1/payment", json={"slip_image_url": "u", "reference_no": "R"})
        assert r.status_code == 200
        service.submit_payment.assert_called_once_with("p1", "user-1", "u", "R")

    def test_a_slip_of_only_spaces_is_a_400(self):
        repo = _repo()
        repo.submit_payment.side_effect = Exception(
            "A payment is not accepted without a slip image")
        r = _app_with(BountyService(repo)).post(
            "/bounty/p1/payment", json={"slip_image_url": "   "})
        assert r.status_code == 400
        repo.submit_payment.assert_called_once_with("p1", "user-1", "   ", None)

    def test_not_found_is_a_404(self, client, service):
        service.submit_payment.side_effect = _async(exc=BountyNotFound("x"))
        assert client.post("/bounty/p1/payment", json={"slip_image_url": "u"}).status_code == 404

    def test_a_conflict_is_a_409(self, client, service):
        service.submit_payment.side_effect = _async(exc=BountyConflict("x"))
        assert client.post("/bounty/p1/payment", json={"slip_image_url": "u"}).status_code == 409

    def test_any_other_failure_is_a_500(self, client, service):
        service.submit_payment.side_effect = _async(exc=RuntimeError("x"))
        assert client.post("/bounty/p1/payment", json={"slip_image_url": "u"}).status_code == 500


# --------------------------------------------------------------------------- #
# answer_payment
# --------------------------------------------------------------------------- #
class TestAnswerPayment:
    # Route level
    def test_answer_passes_the_caller_as_hunter(self, client, service):
        service.hunter_answer.side_effect = _async({"status": "Verified"})
        r = client.post("/bounty/payments/t1/answer", json={"answer": "Confirm"})
        assert r.status_code == 200
        service.hunter_answer.assert_called_once_with("t1", "user-1", "Confirm")

    def test_a_conflict_is_a_409(self, client, service):
        service.hunter_answer.side_effect = _async(exc=BountyConflict("x"))
        r = client.post("/bounty/payments/t1/answer", json={"answer": "Dispute"})
        assert r.status_code == 409

    # Service level (BountyService.hunter_answer)
    def test_answers_are_case_insensitive(self):
        repo = _repo()
        repo.hunter_answer.return_value = {"status": "Verified"}
        run(BountyService(repo).hunter_answer("t1", "hunter-1", "confirm"))
        repo.hunter_answer.assert_called_once_with("t1", "hunter-1", "Confirm")

    def test_an_unknown_hunter_answer_is_refused_before_the_database(self):
        repo = _repo()
        with pytest.raises(ValueError):
            run(BountyService(repo).hunter_answer("t1", "hunter-1", "Maybe"))
        repo.hunter_answer.assert_not_called()


# --------------------------------------------------------------------------- #
# refer_unpaid
# --------------------------------------------------------------------------- #
class TestReferUnpaid:
    def test_refer_unpaid_passes_the_caller(self, client, service):
        service.refer_unpaid.side_effect = _async({"referred": True})
        assert client.post("/bounty/p1/unpaid").status_code == 200
        service.refer_unpaid.assert_called_once_with("p1", "user-1")

    def test_a_conflict_is_a_409(self, client, service):
        service.refer_unpaid.side_effect = _async(exc=BountyConflict("x"))
        assert client.post("/bounty/p1/unpaid").status_code == 409

# --------------------------------------------------------------------------- #
# my_bounties
# --------------------------------------------------------------------------- #
class TestMyBounties:
    def test_mine_is_routed_before_the_pet_id(self, client, service):
        service.bounties_for_hunter.side_effect = _async([{"pet_id": "p1"}])
        r = client.get("/bounty/mine")
        assert r.status_code == 200
        assert r.json()["data"] == [{"pet_id": "p1"}]
        service.bounties_for_hunter.assert_called_once_with("user-1")
        service.get_view.assert_not_called()

    def test_a_failed_read_is_a_500(self, client, service):
        service.bounties_for_hunter.side_effect = _async(exc=RuntimeError("x"))
        assert client.get("/bounty/mine").status_code == 500


# --------------------------------------------------------------------------- #
# bounty_queue
# --------------------------------------------------------------------------- #
class TestBountyQueue:
    def test_admin_queue(self, client, service):
        service.ruling_queue.side_effect = _async([{"kind": "disputed"}])
        r = client.get("/admin/bounty-queue")
        assert r.status_code == 200 and r.json()["data"] == [{"kind": "disputed"}]

    def test_a_failed_read_is_a_500(self, client, service):
        service.ruling_queue.side_effect = _async(exc=RuntimeError("x"))
        assert client.get("/admin/bounty-queue").status_code == 500


# --------------------------------------------------------------------------- #
# rule_bounty_payment
# --------------------------------------------------------------------------- #
class TestRuleBountyPayment:
    # Route level
    def test_admin_ruling_records_the_admin(self, client, service, sent):
        # The owner is named in the result, so only the status check keeps a
        # confirmation from notifying them.
        service.admin_rule.side_effect = _async({"status": "Verified", "owner_id": "owner-1"})
        r = client.post("/admin/bounty-payments/t1/ruling", json={"decision": "Confirm"})
        assert r.status_code == 200
        service.admin_rule.assert_called_once_with("t1", "admin-1", "Confirm")
        sent.assert_not_called()

    def test_a_rejection_notifies_the_owner(self, client, service, sent):
        service.admin_rule.side_effect = _async({"status": "Rejected", "owner_id": "owner-1"})
        r = client.post("/admin/bounty-payments/t1/ruling", json={"decision": "Reject"})
        assert r.status_code == 200
        assert sent.call_args.args[1] == ["owner-1"]
        assert sent.call_args.args[2] == "Bounty payment rejected"

    def test_a_conflict_is_a_409(self, client, service, sent):
        service.admin_rule.side_effect = _async(exc=BountyConflict("x"))
        r = client.post("/admin/bounty-payments/t1/ruling", json={"decision": "Reject"})
        assert r.status_code == 409
        sent.assert_not_called()

    # Service level (BountyService.admin_rule)
    def test_a_decision_in_another_casing_is_normalised(self):
        repo = _repo()
        repo.admin_rule.return_value = {"status": "Rejected"}
        run(BountyService(repo).admin_rule("t1", "admin-1", " reject "))
        repo.admin_rule.assert_called_once_with("t1", "admin-1", "Reject")

    def test_an_unknown_admin_decision_is_refused_before_the_database(self):
        repo = _repo()
        with pytest.raises(ValueError):
            run(BountyService(repo).admin_rule("t1", "admin-1", "Approve"))
        repo.admin_rule.assert_not_called()


# --------------------------------------------------------------------------- #
# close_bounty_unpaid
# --------------------------------------------------------------------------- #
class TestCloseBountyUnpaid:
    def test_admin_close_unpaid(self, client, service):
        service.admin_close_unpaid.side_effect = _async({"pet_status": "Resolved"})
        r = client.post("/admin/missing-pets/p1/close-unpaid")
        assert r.status_code == 200
        service.admin_close_unpaid.assert_called_once_with("p1", "admin-1")

    def test_a_report_no_longer_unpaid_is_a_409(self, client, service):
        service.admin_close_unpaid.side_effect = _async(exc=BountyConflict("x"))
        assert client.post("/admin/missing-pets/p1/close-unpaid").status_code == 409

    def test_the_old_admin_settlement_route_is_gone(self, client):
        """The pre-Feature 6 route let an administrator record a payment as
        Verified and resolve the report in one call, skipping the returning
        hunter's confirmation. A report with a bounty now closes only through
        the routes above."""
        r = client.post("/admin/missing-pets/p1/resolve", json={
            "final_sighting_id": "s1", "slip_image_url": "u"})
        assert r.status_code == 404

# --------------------------------------------------------------------------- #
# is_thai_mobile
# --------------------------------------------------------------------------- #
class TestIsThaiMobile:
    def test_none_is_not_a_mobile_number(self):
        assert is_thai_mobile(None) is False

    def test_ten_digits_starting_08(self):
        assert is_thai_mobile("0812345678") is True

    def test_dashes_are_removed_before_the_check(self):
        assert is_thai_mobile("061-234-5678") is True

    def test_a_landline_number_is_not_a_mobile_number(self):
        assert is_thai_mobile("0212345678") is False

    def test_nine_digits_are_too_few(self):
        assert is_thai_mobile("081234567") is False

# --------------------------------------------------------------------------- #
# translate_db_error
# --------------------------------------------------------------------------- #
class TestTranslateDbError:
    @pytest.mark.parametrize("msg", [
        "Missing pet x not found or not owned by you",
        "Payment x not found",
    ])
    def test_not_found_maps_to_bounty_not_found(self, msg):
        out = translate_db_error(Exception(msg))
        assert type(out) is BountyNotFound and str(out) == msg

    @pytest.mark.parametrize("msg", [
        "Payment x is already Verified",
        "Missing pet x already has a payment awaiting confirmation or confirmed",
    ])
    def test_already_maps_to_conflict(self, msg):
        assert type(translate_db_error(Exception(msg))) is BountyConflict

    def test_without_a_slip_maps_to_value_error(self):
        out = translate_db_error(Exception("A payment is not accepted without a slip image"))
        assert type(out) is ValueError

    def test_an_unknown_error_is_left_alone(self):
        e = RuntimeError("connection reset")
        assert translate_db_error(e) is e
