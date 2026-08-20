"""
Integration tests for apply_score_penalty() — the RPC behind an upheld flag
(SRS-68, migrations/2026_08_20_flag_penalty_not_ban.sql).

Everything worth testing here lives in the database and is invisible to the
unit suite, which stops at the repository port:

  * the score floor — a hunter with 3 points penalised 10 lands on 0, not -7;
  * idempotency on report_id — `AdminService.review_report` writes the flag's
    status LAST so a part-failed review is retryable, and that retry must not
    deduct a second time;
  * atomicity — the audit row and the balance change are one unit;
  * the enum rename — 'Reviewed_Ban' is gone and 'Reviewed_Penalty' exists.

Progress-II SRS traceability: SRS-68 (review a flag; uphold now deducts score
rather than banning the reported user).
"""
import json
import uuid

import psycopg
import pytest

pytestmark = pytest.mark.integration


def _penalise(conn, *, user_id, sighting_id, report_id, points,
              reason="Not_a_pet", admin_id=None):
    """Mirror SupabaseAdminRepository.apply_score_penalty."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT apply_score_penalty(%s, %s, %s, %s, %s::report_reason, %s)",
            (user_id, sighting_id, report_id, points, reason, admin_id),
        )
        out = cur.fetchone()[0]
    return out if isinstance(out, dict) else json.loads(out)


def _penalty_count(conn, user_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM score_penalties WHERE user_id = %s",
            (user_id,),
        )
        return cur.fetchone()[0]


def _score(conn, user_id):
    with conn.cursor() as cur:
        cur.execute("SELECT total_score FROM users WHERE id = %s", (user_id,))
        return cur.fetchone()[0]


@pytest.fixture
def flagged(conn, seed):
    """A hunter with 30 points, their sighting, and a Pending flag against it."""
    hunter = seed.user(display_name="Hunter", total_score=30)
    reporter = seed.user(display_name="Reporter")
    sighting = seed.sighting(hunter_id=hunter)
    report = seed.report(sighting_id=sighting, reporter_id=reporter)
    return {"hunter": hunter, "reporter": reporter,
            "sighting": sighting, "report": report}


class TestDeduction:
    def test_deducts_and_writes_the_audit_row(self, conn, flagged):
        out = _penalise(
            conn, user_id=flagged["hunter"], sighting_id=flagged["sighting"],
            report_id=flagged["report"], points=10,
        )

        assert out["already_applied"] is False
        assert out["points_applied"] == 10
        assert out["total_score_after"] == 20
        assert _score(conn, flagged["hunter"]) == 20
        assert _penalty_count(conn, flagged["hunter"]) == 1

    def test_zero_points_records_the_ruling_without_charging(
        self, conn, flagged
    ):
        """Upholding with a 0 penalty still belongs in the audit trail — the
        administrator ruled on the flag, they just charged nothing for it."""
        out = _penalise(
            conn, user_id=flagged["hunter"], sighting_id=flagged["sighting"],
            report_id=flagged["report"], points=0,
        )

        assert out["points_applied"] == 0
        assert _score(conn, flagged["hunter"]) == 30
        assert _penalty_count(conn, flagged["hunter"]) == 1

    def test_score_floors_at_zero(self, conn, seed):
        """A negative total_score would sort a penalised hunter below someone
        who has never contributed, which reads as a broken leaderboard."""
        hunter = seed.user(total_score=3)
        sighting = seed.sighting(hunter_id=hunter)
        report = seed.report(sighting_id=sighting)

        out = _penalise(
            conn, user_id=hunter, sighting_id=sighting,
            report_id=report, points=10,
        )

        assert out["total_score_after"] == 0
        assert _score(conn, hunter) == 0

    def test_the_ruling_is_recorded_even_when_it_could_not_be_charged(
        self, conn, seed
    ):
        """`points` is what the admin RULED; `points_applied` is what the
        balance could absorb. Collapsing the two would erase the fact that a
        20-point offence was judged."""
        hunter = seed.user(total_score=3)
        sighting = seed.sighting(hunter_id=hunter)
        report = seed.report(sighting_id=sighting)

        out = _penalise(
            conn, user_id=hunter, sighting_id=sighting,
            report_id=report, points=20,
        )

        assert out["points"] == 20
        assert out["points_applied"] == 3
        with conn.cursor() as cur:
            cur.execute("SELECT points FROM score_penalties WHERE report_id = %s",
                        (report,))
            assert cur.fetchone()[0] == 20

    def test_penalties_accumulate_across_separate_flags(self, conn, seed):
        """UNIQUE is on report_id, not on the hunter: two upheld flags are two
        deductions. (Contrast score_awards, which is UNIQUE per pet+user.)"""
        hunter = seed.user(total_score=30)
        s1 = seed.sighting(hunter_id=hunter)
        s2 = seed.sighting(hunter_id=hunter)
        r1 = seed.report(sighting_id=s1)
        r2 = seed.report(sighting_id=s2)

        _penalise(conn, user_id=hunter, sighting_id=s1, report_id=r1, points=10)
        out = _penalise(
            conn, user_id=hunter, sighting_id=s2, report_id=r2, points=5,
        )

        assert out["total_score_after"] == 15
        assert _penalty_count(conn, hunter) == 2


class TestIdempotency:
    def test_replaying_the_same_flag_does_not_double_charge(
        self, conn, flagged
    ):
        """review_report closes the flag LAST, so a crash after the deduction
        leaves a Pending flag an admin will retry. The retry must be free."""
        first = _penalise(
            conn, user_id=flagged["hunter"], sighting_id=flagged["sighting"],
            report_id=flagged["report"], points=10,
        )
        second = _penalise(
            conn, user_id=flagged["hunter"], sighting_id=flagged["sighting"],
            report_id=flagged["report"], points=10,
        )

        assert first["already_applied"] is False
        assert second["already_applied"] is True
        assert second["points_applied"] == 0
        assert second["total_score_after"] == 20
        assert _score(conn, flagged["hunter"]) == 20
        assert _penalty_count(conn, flagged["hunter"]) == 1

    def test_retry_reports_the_standing_balance_not_a_stale_one(
        self, conn, flagged
    ):
        """The replay path re-reads total_score rather than echoing the caller,
        so an admin panel refreshing on retry shows the true figure."""
        _penalise(
            conn, user_id=flagged["hunter"], sighting_id=flagged["sighting"],
            report_id=flagged["report"], points=10,
        )
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET total_score = 99 WHERE id = %s",
                        (flagged["hunter"],))

        out = _penalise(
            conn, user_id=flagged["hunter"], sighting_id=flagged["sighting"],
            report_id=flagged["report"], points=10,
        )

        assert out["total_score_after"] == 99


class TestRejections:
    def test_negative_points_are_refused(self, conn, flagged):
        """A negative penalty is a score AWARD wearing the wrong hat — the
        award path has its own table and its own spam rule."""
        with pytest.raises(psycopg.errors.RaiseException):
            _penalise(
                conn, user_id=flagged["hunter"],
                sighting_id=flagged["sighting"],
                report_id=flagged["report"], points=-5,
            )

    def test_unknown_user_is_refused(self, conn, flagged):
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            _penalise(
                conn, user_id=uuid.uuid4(), sighting_id=flagged["sighting"],
                report_id=flagged["report"], points=10,
            )

    def test_the_audit_row_carries_the_full_provenance(self, conn, seed,
                                                       flagged):
        """Who was charged, for which sighting, off which flag, by which admin.
        Without the admin id a deduction cannot be questioned after the fact."""
        admin = seed.user(display_name="Admin", role="admin")
        _penalise(
            conn, user_id=flagged["hunter"], sighting_id=flagged["sighting"],
            report_id=flagged["report"], points=10, admin_id=admin,
        )

        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, sighting_id, report_id, points, reason, "
                "       penalised_by "
                "FROM score_penalties WHERE report_id = %s",
                (flagged["report"],),
            )
            row = cur.fetchone()

        assert row == (flagged["hunter"], flagged["sighting"],
                       flagged["report"], 10, "Not_a_pet", admin)


class TestEnumRename:
    def test_reviewed_penalty_replaced_reviewed_ban(self, conn):
        """The rename is the visible half of "we deduct, we do not ban". If the
        old label survived, an admin build could still write it."""
        with conn.cursor() as cur:
            cur.execute(
                "SELECT enumlabel FROM pg_enum e JOIN pg_type t "
                "ON t.oid = e.enumtypid WHERE t.typname = 'report_status'"
            )
            labels = {r[0] for r in cur.fetchall()}

        assert labels == {"Pending", "Reviewed_Penalty", "Dismissed"}
