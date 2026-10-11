"""Integration tests for the Feature 6 owner-paid bounty settlement.

Runs migrations/2026_10_08_bounty_settlement.sql against the real container:
the payment functions, the one-live-payment constraint, and the
owner_decide_sighting changes (found_at, and a zero bounty closing straight to
'Resolved'). Feature 6 has no ITC number yet; it is issued in the Phase 6 test
plan.
"""
import psycopg
import pytest

from _query import pet_status

pytestmark = pytest.mark.integration


def _call(conn, sql, args):
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchone()[0]


def _one(conn, sql, args=()):
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchone()


def _closed_case(conn, seed, *, bounty=2000):
    """An owner, a hunter, and a report closed on the hunter's confirmed catch."""
    owner = seed.user(username="Owner")
    hunter = seed.user(username="Hunter")
    pet = seed.missing_pet(owner_id=owner, bounty=bounty)
    catch = seed.sighting(hunter_id=hunter, action="Caught")
    seed.sighting_match(sighting_id=catch, missing_pet_id=pet)
    _call(conn, "SELECT owner_decide_sighting(%s, %s, %s, 'Confirmed')",
          (pet, catch, owner))
    return owner, hunter, pet, catch


def _pay(conn, pet, owner, slip="http://slip/1.jpg", ref="REF1"):
    return _call(conn, "SELECT submit_bounty_payment(%s, %s, %s, %s)",
                 (pet, owner, slip, ref))


def _age_payment(conn, payment_id, hours):
    with conn.cursor() as cur:
        cur.execute("UPDATE bounty_transactions SET created_at = NOW() - make_interval(hours => %s) "
                    "WHERE id = %s", (hours, payment_id))


def _age_found(conn, pet, hours):
    with conn.cursor() as cur:
        cur.execute("UPDATE missing_pets SET found_at = NOW() - make_interval(hours => %s) "
                    "WHERE id = %s", (hours, pet))


class TestClosingTheCase:
    def test_a_caught_confirmation_closes_at_found_and_stamps_the_time(self, conn, seed):
        _, _, pet, _ = _closed_case(conn, seed)
        assert pet_status(conn, pet) == "Found"
        assert _one(conn, "SELECT found_at IS NOT NULL FROM missing_pets WHERE id = %s", (pet,))[0]

    def test_a_zero_bounty_closes_straight_to_resolved(self, conn, seed):
        _, _, pet, _ = _closed_case(conn, seed, bounty=0)
        assert pet_status(conn, pet) == "Resolved"


class TestOwnerPays:
    def test_payment_is_recorded_awaiting_confirmation_against_the_catch(self, conn, seed):
        owner, _, pet, catch = _closed_case(conn, seed)
        out = _pay(conn, pet, owner)
        row = _one(conn, "SELECT status, sighting_id, amount, slip_image_url, reference_no "
                         "FROM bounty_transactions WHERE id = %s", (out["payment_id"],))
        assert row == ("Pending_Verification", catch, 2000, "http://slip/1.jpg", "REF1")
        assert pet_status(conn, pet) == "Found"

    def test_only_the_owner_may_pay(self, conn, seed):
        _, hunter, pet, _ = _closed_case(conn, seed)
        with pytest.raises(psycopg.errors.RaiseException, match="not found or not owned"):
            _pay(conn, pet, hunter)

    def test_a_slip_is_required(self, conn, seed):
        owner, _, pet, _ = _closed_case(conn, seed)
        with pytest.raises(psycopg.errors.RaiseException, match="without a slip"):
            _pay(conn, pet, owner, slip="  ")

    def test_a_second_live_payment_is_refused(self, conn, seed):
        owner, _, pet, _ = _closed_case(conn, seed)
        _pay(conn, pet, owner)
        with pytest.raises(psycopg.errors.RaiseException, match="already has a payment"):
            _pay(conn, pet, owner)

    def test_an_open_search_cannot_be_paid(self, conn, seed):
        owner = seed.user(username="Owner")
        pet = seed.missing_pet(owner_id=owner)
        with pytest.raises(psycopg.errors.RaiseException, match="not awaiting a bounty payment"):
            _pay(conn, pet, owner)

    def test_the_constraint_holds_even_without_the_function(self, conn, seed):
        owner, _, pet, catch = _closed_case(conn, seed)
        _pay(conn, pet, owner)
        with pytest.raises(psycopg.errors.UniqueViolation):
            with conn.cursor() as cur:
                cur.execute("INSERT INTO bounty_transactions (sighting_id, missing_pet_id, owner_id, "
                            "amount, slip_image_url, status) VALUES (%s, %s, %s, 1, 'x', 'Verified')",
                            (catch, pet, owner))


class TestHunterAnswers:
    def test_confirming_resolves_the_report(self, conn, seed):
        owner, hunter, pet, _ = _closed_case(conn, seed)
        pid = _pay(conn, pet, owner)["payment_id"]
        _call(conn, "SELECT hunter_answer_bounty_payment(%s, %s, 'Confirm')", (pid, hunter))
        assert _one(conn, "SELECT status, verified_by FROM bounty_transactions WHERE id = %s",
                    (pid,)) == ("Verified", hunter)
        assert pet_status(conn, pet) == "Resolved"

    def test_confirming_is_allowed_after_the_window(self, conn, seed):
        owner, hunter, pet, _ = _closed_case(conn, seed)
        pid = _pay(conn, pet, owner)["payment_id"]
        _age_payment(conn, pid, 100)
        _call(conn, "SELECT hunter_answer_bounty_payment(%s, %s, 'Confirm')", (pid, hunter))
        assert pet_status(conn, pet) == "Resolved"

    def test_disputing_within_the_window_marks_it(self, conn, seed):
        owner, hunter, pet, _ = _closed_case(conn, seed)
        pid = _pay(conn, pet, owner)["payment_id"]
        _call(conn, "SELECT hunter_answer_bounty_payment(%s, %s, 'Dispute')", (pid, hunter))
        assert _one(conn, "SELECT status, disputed_at IS NOT NULL FROM bounty_transactions "
                          "WHERE id = %s", (pid,)) == ("Pending_Verification", True)

    def test_disputing_after_the_window_is_refused(self, conn, seed):
        owner, hunter, pet, _ = _closed_case(conn, seed)
        pid = _pay(conn, pet, owner)["payment_id"]
        _age_payment(conn, pid, 73)
        with pytest.raises(psycopg.errors.RaiseException, match="window"):
            _call(conn, "SELECT hunter_answer_bounty_payment(%s, %s, 'Dispute')", (pid, hunter))

    def test_a_stranger_reads_as_not_found(self, conn, seed):
        owner, _, pet, _ = _closed_case(conn, seed)
        pid = _pay(conn, pet, owner)["payment_id"]
        stranger = seed.user(username="Stranger")
        with pytest.raises(psycopg.errors.RaiseException, match="not found"):
            _call(conn, "SELECT hunter_answer_bounty_payment(%s, %s, 'Confirm')", (pid, stranger))

    def test_the_owner_cannot_confirm_their_own_payment(self, conn, seed):
        owner, _, pet, _ = _closed_case(conn, seed)
        pid = _pay(conn, pet, owner)["payment_id"]
        with pytest.raises(psycopg.errors.RaiseException, match="not found"):
            _call(conn, "SELECT hunter_answer_bounty_payment(%s, %s, 'Confirm')", (pid, owner))


class TestUnpaidReferral:
    def test_the_hunter_refers_within_three_days(self, conn, seed):
        _, hunter, pet, _ = _closed_case(conn, seed)
        _call(conn, "SELECT refer_bounty_unpaid(%s, %s)", (pet, hunter))
        assert _one(conn, "SELECT unpaid_referred_at IS NOT NULL FROM missing_pets WHERE id = %s",
                    (pet,))[0]

    def test_a_second_referral_is_refused(self, conn, seed):
        _, hunter, pet, _ = _closed_case(conn, seed)
        _call(conn, "SELECT refer_bounty_unpaid(%s, %s)", (pet, hunter))
        with pytest.raises(psycopg.errors.RaiseException, match="already referred"):
            _call(conn, "SELECT refer_bounty_unpaid(%s, %s)", (pet, hunter))

    def test_after_three_days_the_window_has_closed(self, conn, seed):
        _, hunter, pet, _ = _closed_case(conn, seed)
        _age_found(conn, pet, 73)
        with pytest.raises(psycopg.errors.RaiseException, match="window"):
            _call(conn, "SELECT refer_bounty_unpaid(%s, %s)", (pet, hunter))

    def test_a_paid_report_cannot_be_referred(self, conn, seed):
        owner, hunter, pet, _ = _closed_case(conn, seed)
        _pay(conn, pet, owner)
        with pytest.raises(psycopg.errors.RaiseException, match="is not unpaid"):
            _call(conn, "SELECT refer_bounty_unpaid(%s, %s)", (pet, hunter))

    def test_paying_closes_the_referral(self, conn, seed):
        owner, hunter, pet, _ = _closed_case(conn, seed)
        _call(conn, "SELECT refer_bounty_unpaid(%s, %s)", (pet, hunter))
        _pay(conn, pet, owner)
        assert _one(conn, "SELECT unpaid_referred_at FROM missing_pets WHERE id = %s", (pet,))[0] is None


class TestAdministrator:
    def test_confirm_resolves_and_records_the_admin(self, conn, seed):
        owner, _, pet, _ = _closed_case(conn, seed)
        admin = seed.user(username="Admin", role="admin")
        pid = _pay(conn, pet, owner)["payment_id"]
        _call(conn, "SELECT admin_rule_bounty_payment(%s, %s, 'Confirm')", (pid, admin))
        assert _one(conn, "SELECT status, verified_by FROM bounty_transactions WHERE id = %s",
                    (pid,)) == ("Verified", admin)
        assert pet_status(conn, pet) == "Resolved"

    def test_reject_keeps_the_report_found_and_lets_the_owner_pay_again(self, conn, seed):
        owner, _, pet, _ = _closed_case(conn, seed)
        admin = seed.user(username="Admin", role="admin")
        pid = _pay(conn, pet, owner)["payment_id"]
        _call(conn, "SELECT admin_rule_bounty_payment(%s, %s, 'Reject')", (pid, admin))
        assert _one(conn, "SELECT status FROM bounty_transactions WHERE id = %s", (pid,))[0] == "Rejected"
        assert pet_status(conn, pet) == "Found"
        again = _pay(conn, pet, owner, slip="http://slip/2.jpg")
        assert again["status"] == "Pending_Verification"

    def test_a_settled_payment_cannot_be_ruled_again(self, conn, seed):
        owner, hunter, pet, _ = _closed_case(conn, seed)
        admin = seed.user(username="Admin", role="admin")
        pid = _pay(conn, pet, owner)["payment_id"]
        _call(conn, "SELECT hunter_answer_bounty_payment(%s, %s, 'Confirm')", (pid, hunter))
        with pytest.raises(psycopg.errors.RaiseException, match="already Verified"):
            _call(conn, "SELECT admin_rule_bounty_payment(%s, %s, 'Reject')", (pid, admin))

    def test_close_as_unpaid(self, conn, seed):
        _, _, pet, _ = _closed_case(conn, seed)
        admin = seed.user(username="Admin", role="admin")
        _call(conn, "SELECT admin_close_bounty_unpaid(%s, %s)", (pet, admin))
        assert _one(conn, "SELECT status, bounty_unpaid, unpaid_closed_by FROM missing_pets "
                          "WHERE id = %s", (pet,)) == ("Resolved", True, admin)

    def test_close_as_unpaid_is_refused_while_a_payment_awaits(self, conn, seed):
        owner, _, pet, _ = _closed_case(conn, seed)
        admin = seed.user(username="Admin", role="admin")
        _pay(conn, pet, owner)
        with pytest.raises(psycopg.errors.RaiseException, match="is not unpaid"):
            _call(conn, "SELECT admin_close_bounty_unpaid(%s, %s)", (pet, admin))


class TestQueue:
    def _kinds(self, conn, pet):
        with conn.cursor() as cur:
            cur.execute("SELECT kind FROM bounty_ruling_queue() WHERE pet_id = %s", (pet,))
            return sorted(r[0] for r in cur.fetchall())

    def test_a_fresh_payment_is_not_in_the_queue(self, conn, seed):
        owner, _, pet, _ = _closed_case(conn, seed)
        _pay(conn, pet, owner)
        assert self._kinds(conn, pet) == []

    def test_a_disputed_payment_is(self, conn, seed):
        owner, hunter, pet, _ = _closed_case(conn, seed)
        pid = _pay(conn, pet, owner)["payment_id"]
        _call(conn, "SELECT hunter_answer_bounty_payment(%s, %s, 'Dispute')", (pid, hunter))
        assert self._kinds(conn, pet) == ["disputed"]

    def test_a_payment_unconfirmed_past_the_window_is(self, conn, seed):
        owner, _, pet, _ = _closed_case(conn, seed)
        pid = _pay(conn, pet, owner)["payment_id"]
        _age_payment(conn, pid, 73)
        assert self._kinds(conn, pet) == ["unconfirmed"]

    def test_a_referred_report_is(self, conn, seed):
        _, hunter, pet, _ = _closed_case(conn, seed)
        _call(conn, "SELECT refer_bounty_unpaid(%s, %s)", (pet, hunter))
        assert self._kinds(conn, pet) == ["unpaid"]


class TestBountiesForHunter:
    def _rows(self, conn, hunter):
        with conn.cursor() as cur:
            cur.execute("SELECT pet_id, pet_status, payment_status FROM bounties_for_hunter(%s)",
                        (hunter,))
            return cur.fetchall()

    def test_lists_the_report_the_hunter_returned_with_its_payment(self, conn, seed):
        owner, hunter, pet, _ = _closed_case(conn, seed)
        assert self._rows(conn, hunter) == [(pet, "Found", None)]
        _pay(conn, pet, owner)
        assert self._rows(conn, hunter) == [(pet, "Found", "Pending_Verification")]

    def test_other_users_see_nothing(self, conn, seed):
        owner, _, _, _ = _closed_case(conn, seed)
        assert self._rows(conn, owner) == []

    def test_a_zero_bounty_report_is_not_listed(self, conn, seed):
        _, hunter, _, _ = _closed_case(conn, seed, bounty=0)
        assert self._rows(conn, hunter) == []


class TestReviewFixes:
    """Regression cases for the code review of 2026-10-08."""

    def _kinds(self, conn, pet):
        with conn.cursor() as cur:
            cur.execute("SELECT kind FROM bounty_ruling_queue() WHERE pet_id = %s", (pet,))
            return sorted(r[0] for r in cur.fetchall())

    def _reject(self, conn, seed, pet, owner, hours_ago):
        admin = seed.user(username="Admin", role="admin")
        pid = _pay(conn, pet, owner)["payment_id"]
        _call(conn, "SELECT admin_rule_bounty_payment(%s, %s, 'Reject')", (pid, admin))
        with conn.cursor() as cur:
            cur.execute("UPDATE bounty_transactions SET verified_at = NOW() - make_interval(hours => %s) "
                        "WHERE id = %s", (hours_ago, pid))

    def test_a_rejection_not_paid_again_for_three_days_reaches_the_queue(self, conn, seed):
        owner, _, pet, _ = _closed_case(conn, seed)
        self._reject(conn, seed, pet, owner, 73)
        assert self._kinds(conn, pet) == ["rejected_unpaid"]

    def test_a_fresh_rejection_is_not_yet_in_the_queue(self, conn, seed):
        owner, _, pet, _ = _closed_case(conn, seed)
        self._reject(conn, seed, pet, owner, 1)
        assert self._kinds(conn, pet) == []

    def test_paying_again_takes_it_out_of_the_queue(self, conn, seed):
        owner, _, pet, _ = _closed_case(conn, seed)
        self._reject(conn, seed, pet, owner, 73)
        _pay(conn, pet, owner, slip="http://slip/2.jpg")
        assert self._kinds(conn, pet) == []

    def test_backfill_closes_found_reports_with_nobody_to_pay(self, conn, seed):
        owner = seed.user(username="Owner")
        hand_closed = seed.missing_pet(owner_id=owner, status="Found", bounty=1000)
        zero = seed.missing_pet(owner_id=owner, status="Found", bounty=0)
        _call(conn, "SELECT bounty_backfill_existing()", ())
        assert pet_status(conn, hand_closed) == "Resolved"
        assert pet_status(conn, zero) == "Resolved"

    def test_backfill_stamps_found_at_on_reports_with_a_hunter(self, conn, seed):
        owner = seed.user(username="Owner")
        hunter = seed.user(username="Hunter")
        pet = seed.missing_pet(owner_id=owner, status="Found", bounty=1000)
        catch = seed.sighting(hunter_id=hunter, action="Caught")
        seed.sighting_match(sighting_id=catch, missing_pet_id=pet, owner_status="Confirmed")
        _call(conn, "SELECT bounty_backfill_existing()", ())
        assert pet_status(conn, pet) == "Found"
        assert _one(conn, "SELECT found_at IS NOT NULL FROM missing_pets WHERE id = %s", (pet,))[0]
        _call(conn, "SELECT refer_bounty_unpaid(%s, %s)", (pet, hunter))
