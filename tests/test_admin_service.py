"""
Unit tests for app/services/admin_service.py — verification, timeline, and the
resolve wrapper — all through the AdminRepository port (MagicMock(spec=...)).

Category-Partition highlights:
  * verify_sighting: invalid status [error] / row not found [error] / success /
    unexpected repo error [error]
  * resolve_missing_pet: the DB function's known RAISE prefixes are translated to
    ValueError (400); any other error propagates as-is.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from app.repositories.admin_repository import AdminRepository
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
