"""
Unit tests for app/services/admin_service.py — verification, timeline, the
resolve wrapper, and the Progress-II moderation methods — all through the
repository ports (MagicMock(spec=...)).

Category-Partition highlights:
  * verify_sighting: invalid status [error] / row not found [error] / success /
    unexpected repo error [error]
  * resolve_missing_pet: the DB function's known RAISE prefixes are translated to
    ValueError (400); any other error propagates as-is.
  * review_report (UTC-35): unknown [404] / already moderated [409] / dismiss /
    uphold / write-ordering / bad decision [400] / null target / db error [500]

Withdrawn on 2026-08-17 along with the feature: the account dashboard
(list/search) and the suspend/deactivate blocks. Admin moderation no longer
touches accounts at all — see the admin_service module docstring.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from app.repositories.admin_repository import AdminRepository
from app.repositories.report_repository import (
    ReportAlreadyModerated,
    ReportNotFound,
    ReportRepository,
)
from app.services.admin_service import AdminService


def run(coro):
    return asyncio.run(coro)


def _repo():
    return MagicMock(spec=AdminRepository)


# --------------------------------------------------------------------------- #
# verify_sighting
# --------------------------------------------------------------------------- #
class TestVerifySighting:
    def test_rejects_invalid_status(self):
        repo = _repo()
        with pytest.raises(ValueError):
            run(AdminService(repo).verify_sighting("s1", "Bogus"))
        repo.update_sighting_verification.assert_not_called()

    def test_not_found_raises(self):
        repo = _repo()
        repo.update_sighting_verification.return_value = None
        with pytest.raises(ValueError):
            run(AdminService(repo).verify_sighting("s1", "Verified"))

    def test_success_strips_vector_and_returns_row(self):
        repo = _repo()
        repo.update_sighting_verification.return_value = {
            "id": "s1", "verification_status": "Verified", "feature_vector": [0.1],
        }
        out = run(AdminService(repo).verify_sighting("s1", "Verified"))
        assert out["id"] == "s1"
        assert "feature_vector" not in out   # internal vector stripped
        repo.update_sighting_verification.assert_called_once_with("s1", "Verified")

    def test_unexpected_error_is_reraised(self):
        repo = _repo()
        repo.update_sighting_verification.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            run(AdminService(repo).verify_sighting("s1", "Dismissed"))


# --------------------------------------------------------------------------- #
# get_sighting_timeline
# --------------------------------------------------------------------------- #
class TestGetSightingTimeline:
    def test_returns_repo_rows(self):
        repo = _repo()
        repo.get_sighting_timeline.return_value = [{"id": "s1"}]
        out = run(AdminService(repo).get_sighting_timeline("pet-1", limit=10, offset=5))
        assert out == [{"id": "s1"}]
        repo.get_sighting_timeline.assert_called_once_with("pet-1", 10, 5)

    def test_error_is_reraised(self):
        repo = _repo()
        repo.get_sighting_timeline.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            run(AdminService(repo).get_sighting_timeline("pet-1"))


# --------------------------------------------------------------------------- #
# resolve_missing_pet — the DB function's known RAISE prefixes -> 400 (ValueError)
# --------------------------------------------------------------------------- #
class TestResolveMissingPet:
    def test_success_returns_repo_result(self):
        repo = _repo()
        repo.resolve_missing_pet.return_value = {"final_hunter_id": "h1"}
        out = run(AdminService(repo).resolve_missing_pet(
            "pet-1", "s1", "slip.jpg", "REF-1", "admin-1"))
        assert out == {"final_hunter_id": "h1"}
        repo.resolve_missing_pet.assert_called_once_with(
            "pet-1", "s1", "slip.jpg", "REF-1", "admin-1")

    @pytest.mark.parametrize("msg", [
        "pet already resolved",
        "final sighting is not a verified Caught sighting",
        "pet not found",
    ])
    def test_known_db_errors_become_valueerror(self, msg):
        repo = _repo()
        repo.resolve_missing_pet.side_effect = Exception(msg)
        with pytest.raises(ValueError):
            run(AdminService(repo).resolve_missing_pet(
                "pet-1", "s1", "slip.jpg", None, "admin-1"))

    def test_unknown_error_propagates_unchanged(self):
        repo = _repo()
        repo.resolve_missing_pet.side_effect = RuntimeError("connection reset")
        with pytest.raises(RuntimeError):
            run(AdminService(repo).resolve_missing_pet(
                "pet-1", "s1", "slip.jpg", None, "admin-1"))


# --------------------------------------------------------------------------- #
# UTC-35  review_report (MD-40, SRS-68) — dismiss or uphold a moderation flag.
#
# As-built note: the test plan writes the constructor as
# `AdminService(report_repo, sighting_repo, user_repo)`. As built there is no
# separate sighting repo — dismissing the flagged sighting reuses the
# `AdminRepository.update_sighting_verification` this service already owns (the
# same call PATCH /admin/sightings/{id}/verification makes), so the constructor
# is `AdminService(repo, report_repo=...)`. There is no user repo either:
# upholding a flag withdraws the SIGHTING and never touches the account.
#
# Ordering matters and is asserted: the flag's own status is written LAST, so a
# failure mid-way leaves it Pending and retryable rather than closed with its
# consequences unapplied.
# --------------------------------------------------------------------------- #
def _moderation_service(flag=None, sighting=None):
    """AdminService wired with both doubles, its report repo returning `flag`
    and its sighting update returning `sighting`."""
    repo = _repo()
    repo.update_sighting_verification.return_value = sighting
    report_repo = MagicMock(spec=ReportRepository)
    report_repo.get_report.return_value = flag
    report_repo.update_report.side_effect = lambda rid, patch: {
        "id": rid, **patch
    }
    service = AdminService(repo, report_repo=report_repo)
    return service, repo, report_repo


_PENDING_FLAG = {"id": "r1", "sighting_id": "s1", "status": "Pending"}


class TestReviewReport:
    def test_unknown_flag_raises_not_found(self):
        """UTC-35-TC-01 — no such flag => ReportNotFound (API maps to 404)."""
        service, _, report_repo = _moderation_service(flag=None)
        with pytest.raises(ReportNotFound):
            run(service.review_report("ghost", "Dismissed", "a1"))
        report_repo.update_report.assert_not_called()

    def test_already_moderated_raises_conflict(self):
        """UTC-35-TC-02 — UD-14 [E1]: a decided flag is never overwritten."""
        service, _, report_repo = _moderation_service(
            flag={"id": "r1", "sighting_id": "s1", "status": "Dismissed"},
        )
        with pytest.raises(ReportAlreadyModerated):
            run(service.review_report("r1", "Reviewed_Ban", "a1"))
        report_repo.update_report.assert_not_called()

    def test_dismiss_writes_status_only(self):
        """UTC-35-TC-03 — dismissing touches the flag and nothing else: the
        sighting stays visible to the owner."""
        service, repo, report_repo = _moderation_service(
            flag=dict(_PENDING_FLAG),
        )

        out = run(service.review_report("r1", "Dismissed", "a1"))

        report_repo.update_report.assert_called_once_with(
            "r1", {"status": "Dismissed"}
        )
        assert out["report"]["status"] == "Dismissed"
        assert out["sighting_dismissed"] is False
        repo.update_sighting_verification.assert_not_called()

    def test_uphold_dismisses_the_sighting(self):
        """UTC-35-TC-04 — upholding withdraws the offending sighting so it
        stops reaching owners, and closes the flag."""
        service, repo, report_repo = _moderation_service(
            flag=dict(_PENDING_FLAG),
            sighting={"id": "s1", "hunter_id": "h9"},
        )

        out = run(service.review_report("r1", "Reviewed and banned", "a1"))

        repo.update_sighting_verification.assert_called_once_with(
            "s1", "Dismissed"
        )
        report_repo.update_report.assert_called_once_with(
            "r1", {"status": "Reviewed_Ban"}
        )
        assert out["sighting_dismissed"] is True

    def test_uphold_leaves_the_hunters_account_alone(self):
        """Despite the Reviewed_Ban enum name, nobody is banned — the account
        system was withdrawn on 2026-08-17. The returned outcome must not claim
        a suspension either, or the admin panel will report one that never
        happened."""
        service, _, _ = _moderation_service(
            flag=dict(_PENDING_FLAG),
            sighting={"id": "s1", "hunter_id": "h9"},
        )

        out = run(service.review_report("r1", "Reviewed_Ban", "a1"))

        assert "suspended_user_id" not in out
        assert set(out) == {"report", "sighting_dismissed"}

    def test_flag_status_is_written_last(self):
        """The idempotency guard only works if the flag closes AFTER its
        consequences: a failed sighting-dismissal must leave it Pending."""
        service, repo, report_repo = _moderation_service(
            flag=dict(_PENDING_FLAG),
            sighting={"id": "s1", "hunter_id": "h9"},
        )
        repo.update_sighting_verification.side_effect = RuntimeError("db down")

        with pytest.raises(RuntimeError):
            run(service.review_report("r1", "Reviewed_Ban", "a1"))

        report_repo.update_report.assert_not_called()

    def test_rejects_decision_outside_the_set(self):
        """A decision that is neither dismiss nor uphold is rejected BEFORE any
        read or write (API maps the ValueError to 400)."""
        service, _, report_repo = _moderation_service(
            flag=dict(_PENDING_FLAG),
        )
        with pytest.raises(ValueError):
            run(service.review_report("r1", "Pending", "a1"))
        report_repo.get_report.assert_not_called()
        report_repo.update_report.assert_not_called()

    def test_uphold_without_a_target_sighting_still_closes_the_flag(self):
        """`reports.sighting_id` is nullable, so a flag can point at nothing.
        That must not crash the queue — close the flag, dismiss nothing."""
        service, repo, report_repo = _moderation_service(
            flag={"id": "r1", "sighting_id": None, "status": "Pending"},
        )

        out = run(service.review_report("r1", "Reviewed_Ban", "a1"))

        repo.update_sighting_verification.assert_not_called()
        assert out["report"]["status"] == "Reviewed_Ban"
        assert out["sighting_dismissed"] is False

    def test_db_error_propagates(self):
        """UTC-35-TC-05 — a repo failure surfaces (API maps it to 500)."""
        service, _, report_repo = _moderation_service(
            flag=dict(_PENDING_FLAG),
        )
        report_repo.update_report.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            run(service.review_report("r1", "Dismissed", "a1"))

    def test_update_returning_no_row_raises_not_found(self):
        """A flag deleted between the read and the write must not report
        success on a row that no longer exists."""
        service, _, report_repo = _moderation_service(
            flag=dict(_PENDING_FLAG),
        )
        report_repo.update_report.side_effect = None
        report_repo.update_report.return_value = None
        with pytest.raises(ReportNotFound):
            run(service.review_report("r1", "Dismissed", "a1"))


# --------------------------------------------------------------------------- #
# UTC-46  list_reports (MD-51) — read the queue that UTC-35 acts from.
#
# The gap this closes: review_report takes a report_id, and until now nothing
# returned one. The queue was writable (MD-39) and decidable (MD-40) but never
# enumerable, so no admin screen could reach a flag.
#
# The contract worth pinning is the filter's THREE-WAY behaviour — a valid
# bucket is forwarded, absence means every bucket, and an unknown value fails
# before any I/O. The last one is what keeps a typo a 400 rather than a failed
# enum cast surfacing as a 500.
# --------------------------------------------------------------------------- #
def _queue_service(rows=None):
    """AdminService whose report repo returns `rows` from list_reports."""
    report_repo = MagicMock(spec=ReportRepository)
    report_repo.list_reports.return_value = rows if rows is not None else []
    return AdminService(_repo(), report_repo=report_repo), report_repo


class TestListReports:
    def test_forwards_a_valid_filter_and_pagination(self):
        """UTC-46-TC-01 — the normalised bucket and the page reach the port."""
        service, report_repo = _queue_service(rows=[{"id": "r1"}])
        out = run(service.list_reports(status="Pending", limit=5, offset=10))
        assert out == [{"id": "r1"}]
        report_repo.list_reports.assert_called_once_with("Pending", 5, 10)

    def test_absent_filter_passes_none_meaning_every_status(self):
        """UTC-46-TC-02 — None must survive to the adapter so the predicate is
        skipped entirely. A string here would silently show one bucket only."""
        service, report_repo = _queue_service()
        run(service.list_reports())
        assert report_repo.list_reports.call_args.args[0] is None

    def test_filter_is_normalised_before_the_query(self):
        """UTC-46-TC-03 — casing from a UI must not reach the enum verbatim."""
        service, report_repo = _queue_service()
        run(service.list_reports(status="  reviewed_ban  "))
        assert report_repo.list_reports.call_args.args[0] == "Reviewed_Ban"

    def test_unknown_filter_raises_before_any_io(self):
        """UTC-46-TC-04 — the 400 happens at the edge; the port is never
        touched, so a bad filter cannot become a database error."""
        service, report_repo = _queue_service()
        with pytest.raises(ValueError):
            run(service.list_reports(status="Banned"))
        report_repo.list_reports.assert_not_called()

    def test_empty_queue_returns_empty_list(self):
        """UTC-46-TC-05 — nothing to moderate is a success, not a 404."""
        service, _ = _queue_service(rows=[])
        assert run(service.list_reports()) == []

    def test_db_error_propagates(self):
        """UTC-46-TC-06 — a repo failure surfaces (API maps it to 500)."""
        service, report_repo = _queue_service()
        report_repo.list_reports.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            run(service.list_reports())
