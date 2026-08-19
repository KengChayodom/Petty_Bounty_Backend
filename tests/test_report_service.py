"""
UTC-41 — flag_sighting (MD-39, SRS-67): a user reporting a sighting for review.

Boundary rule: both DB dependencies are repository ports, doubled with
MagicMock(spec=...) — `ReportRepository` for the `reports` write and
`SightingRepository` for the existence check of the target.

What each test is protecting (per TEST_PLAN §3's "name the defect" rule):
  * validation before I/O — a bad reason must not cost a round-trip and must
    never reach the `report_reason` cast;
  * the insert contract — Pending, and the reporter taken from the JWT, so a
    client cannot file a pre-decided flag or one attributed to someone else;
  * the target check — `reports.sighting_id` is nullable, so without it a flag
    against a ghost id inserts happily and sits in the queue pointing nowhere.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from app.repositories.report_repository import (
    FlagTargetNotFound,
    ReportRepository,
)
from app.repositories.sighting_repository import SightingRepository
from app.services.moderation_logic import FLAG_REASONS
from app.services.report_service import ReportService


def run(coro):
    return asyncio.run(coro)


def _service(sighting={"id": "s1", "hunter_id": "h1"}):
    repo = MagicMock(spec=ReportRepository)
    repo.create_flag.side_effect = lambda payload: {"id": "r1", **payload}
    sighting_repo = MagicMock(spec=SightingRepository)
    sighting_repo.get_sighting.return_value = sighting
    return ReportService(repo, sighting_repo), repo, sighting_repo


class TestFlagSighting:
    def test_rejects_reason_outside_the_set(self):
        """UTC-41-TC-01 — bad reason => ValueError (API maps it to 400), and
        nothing is read or written."""
        service, repo, sighting_repo = _service()

        with pytest.raises(ValueError):
            run(service.flag_sighting("s1", "Ugly", "r1"))

        repo.create_flag.assert_not_called()
        sighting_repo.get_sighting.assert_not_called()

    @pytest.mark.parametrize("reason", FLAG_REASONS)
    def test_accepts_each_valid_reason(self, reason):
        """UTC-41-TC-02 — every permitted reason inserts exactly one flag."""
        service, repo, _ = _service()

        out = run(service.flag_sighting("s1", reason, "r1"))

        assert out["reason"] == reason
        repo.create_flag.assert_called_once()

    @pytest.mark.parametrize("supplied,stored", [
        ("Not a pet", "Not_a_pet"),
        ("Inappropriate image", "Inappropriate_image"),
    ])
    def test_accepts_the_spec_spellings(self, supplied, stored):
        """MD-39 spells two reasons with spaces; the enum does not. They must
        still be accepted, and stored in the enum's spelling."""
        service, repo, _ = _service()

        run(service.flag_sighting("s1", supplied, "r1"))

        assert repo.create_flag.call_args.args[0]["reason"] == stored

    def test_inserts_pending_with_reporter_from_the_token(self):
        """UTC-41-TC-03 — status Pending and reporter_id from the caller."""
        service, repo, _ = _service()

        run(service.flag_sighting("s1", "Spam", "r1"))

        payload = repo.create_flag.call_args.args[0]
        assert payload["status"] == "Pending"
        assert payload["reporter_id"] == "r1"
        assert payload["sighting_id"] == "s1"

    def test_unknown_target_sighting_raises_not_found(self):
        """UTC-41-TC-04 — no such sighting => FlagTargetNotFound (API -> 404),
        and no orphan row is written."""
        service, repo, _ = _service(sighting=None)

        with pytest.raises(FlagTargetNotFound):
            run(service.flag_sighting("ghost", "Spam", "r1"))

        repo.create_flag.assert_not_called()

    def test_db_error_propagates(self):
        """UTC-41-TC-05 — insert failure surfaces (API maps it to 500)."""
        service, repo, _ = _service()
        repo.create_flag.side_effect = RuntimeError("db down")

        with pytest.raises(RuntimeError):
            run(service.flag_sighting("s1", "Spam", "r1"))


class TestFlagNotSaved:
    def test_carries_the_payload_it_failed_to_write(self):
        """The adapter raises this when the INSERT returns no row; keeping the
        payload on the exception is what makes the 500 diagnosable."""
        from app.repositories.report_repository import FlagNotSaved

        payload = {"sighting_id": "s1", "reason": "Spam"}
        err = FlagNotSaved(payload)
        assert err.payload == payload
        assert isinstance(err, ValueError)
