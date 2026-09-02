"""
Unit tests for administrator role assignment — UTC-52, UTC-53, UTC-54
(MD-58 to MD-60, SRS-94 to SRS-98, UD-23).

Written against `progress_2/test_plan.md` §3.1.14 Roles Module. The boundary is
the `UserRepository` port, doubled with `MagicMock(spec=...)`: stubbed for return
values, spied on to verify the arguments where making the call *is* the
behaviour under test.

Category-Partition highlights:
  * find_user_by_email: exact hit / near miss [404] / case / unknown [404] /
    malformed [400] / empty [400] / db error
  * assign_user_role: grant / withdraw / bad role [400] / self-demotion [409] /
    last administrator [409] / unknown account [404] / already held (no-op) /
    the acting admin is taken from the JWT / db error
  * list_role_changes: page + total / empty / per-account filter / paging /
    db error

What is deliberately NOT tested here:
  * That a withdrawn role stops working (SRS-98) is a property of
    `require_admin`, which is Feature 1's gate with its own coverage. Nothing in
    MD-59 revokes anything, so there is no behaviour of this method to assert.
  * That the two guards of SRS-96 hold when two administrators act at the same
    moment is a property of the `assign_user_role` procedure and belongs to
    tests/integration/ — same reading as MD-54's rule set.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from app.repositories.admin_repository import AdminRepository
from app.repositories.pagination import Page
from app.repositories.user_repository import (
    RoleAssignmentRefused,
    UserAccountNotFound,
    UserRepository,
)
from app.services.admin_service import AdminService


def run(coro):
    return asyncio.run(coro)


def _service():
    """AdminService wired with a doubled UserRepository (the port under test)."""
    user_repo = MagicMock(spec=UserRepository)
    service = AdminService(
        MagicMock(spec=AdminRepository), user_repo=user_repo,
    )
    return service, user_repo


ACCOUNT = {"id": "u2", "display_name": "Kus", "role": "user"}


# --------------------------------------------------------------------------- #
# UTC-52 — find_user_by_email (MD-58, SRS-94)
# --------------------------------------------------------------------------- #
class TestFindUserByEmail:
    def test_tc01_exact_address_returns_one_account(self):
        service, repo = _service()
        repo.find_by_email.return_value = ACCOUNT

        result = run(service.find_user_by_email("hunter@example.com"))

        assert result == ACCOUNT
        # One account, not a page: the struck account browse must not come back
        # through this method.
        assert not isinstance(result, list)
        assert set(result) >= {"id", "display_name", "role"}

    def test_tc02_address_is_matched_in_full(self):
        service, repo = _service()
        repo.find_by_email.return_value = None  # nothing holds the near miss

        with pytest.raises(UserAccountNotFound):
            run(service.find_user_by_email("hunter@example.co"))

        # The near-matching account is not disclosed: the port was asked for the
        # address as typed, and nothing came back.
        repo.find_by_email.assert_called_once_with("hunter@example.co")

    def test_tc03_case_is_ignored(self):
        service, repo = _service()
        repo.find_by_email.return_value = ACCOUNT

        result = run(service.find_user_by_email("Hunter@Example.com"))

        repo.find_by_email.assert_called_once_with("hunter@example.com")
        assert result == ACCOUNT

    def test_tc04_unknown_address(self):
        service, repo = _service()
        repo.find_by_email.return_value = None

        with pytest.raises(UserAccountNotFound):
            run(service.find_user_by_email("ghost@example.com"))

    @pytest.mark.parametrize(
        "address",
        ["not-an-address", "no-at-sign.com", "@example.com", "a@b",
         "two@@example.com", "spaced address@example.com"],
    )
    def test_tc05_malformed_address_rejected_before_the_lookup(self, address):
        service, repo = _service()

        with pytest.raises(ValueError):
            run(service.find_user_by_email(address))

        repo.find_by_email.assert_not_called()

    @pytest.mark.parametrize("address", ["", "   ", None])
    def test_tc06_empty_address_rejected(self, address):
        service, repo = _service()

        with pytest.raises(ValueError):
            run(service.find_user_by_email(address))

        repo.find_by_email.assert_not_called()

    def test_tc07_database_error_propagates(self):
        service, repo = _service()
        repo.find_by_email.side_effect = RuntimeError("DB connection lost")

        with pytest.raises(RuntimeError):
            run(service.find_user_by_email("hunter@example.com"))


# --------------------------------------------------------------------------- #
# UTC-53 — assign_user_role (MD-59, SRS-95 to SRS-98)
# --------------------------------------------------------------------------- #
class TestAssignUserRole:
    def test_tc01_grants_the_administrator_role(self):
        service, repo = _service()
        repo.assign_user_role.return_value = {
            "changed": True, "id": "u2", "display_name": "Kus",
            "role_before": "user", "role_after": "admin",
        }

        result = run(service.assign_user_role("u2", "admin", "a1"))

        assert result["role_after"] == "admin"
        assert result["role_before"] == "user"
        repo.assign_user_role.assert_called_once_with("u2", "admin", "a1")

    def test_tc02_withdraws_the_administrator_role(self):
        service, repo = _service()
        repo.assign_user_role.return_value = {
            "changed": True, "id": "u2", "display_name": "Kus",
            "role_before": "admin", "role_after": "user",
        }

        result = run(service.assign_user_role("u2", "user", "a1"))

        assert result["role_after"] == "user"
        repo.assign_user_role.assert_called_once_with("u2", "user", "a1")

    @pytest.mark.parametrize("role", ["superuser", "moderator", "", None])
    def test_tc03_role_outside_the_enumeration_is_rejected(self, role):
        service, repo = _service()

        with pytest.raises(ValueError):
            run(service.assign_user_role("u2", role, "a1"))

        repo.assign_user_role.assert_not_called()

    def test_tc04_self_demotion_is_refused(self):
        service, repo = _service()

        with pytest.raises(RoleAssignmentRefused):
            run(service.assign_user_role("a1", "user", "a1"))

        # No round trip: the caller's own identifier is known without a read.
        repo.assign_user_role.assert_not_called()

    def test_tc04b_promoting_yourself_is_not_the_self_demotion_case(self):
        """Re-asserting your own 'admin' is a no-op, not a withdrawal.

        The guard is about WITHDRAWING your own access. Refusing every write
        where target == caller would also refuse this harmless one.
        """
        service, repo = _service()
        repo.assign_user_role.return_value = {
            "changed": False, "id": "a1", "display_name": "Boss",
            "role_before": "admin", "role_after": "admin",
        }

        result = run(service.assign_user_role("a1", "admin", "a1"))

        assert result["changed"] is False
        repo.assign_user_role.assert_called_once_with("a1", "admin", "a1")

    def test_tc05_last_administrator_is_refused(self):
        service, repo = _service()
        repo.assign_user_role.side_effect = RoleAssignmentRefused(
            "This is the only administrator. "
            "Grant the role to another account first."
        )

        with pytest.raises(RoleAssignmentRefused):
            run(service.assign_user_role("a2", "user", "a1"))

    def test_tc06_unknown_account(self):
        service, repo = _service()
        repo.assign_user_role.side_effect = UserAccountNotFound("ghost")

        with pytest.raises(UserAccountNotFound):
            run(service.assign_user_role("ghost", "admin", "a1"))

    def test_tc07_role_already_held_writes_nothing(self):
        service, repo = _service()
        repo.assign_user_role.return_value = {
            "changed": False, "id": "u2", "display_name": "Kus",
            "role_before": "admin", "role_after": "admin",
        }

        result = run(service.assign_user_role("u2", "admin", "a1"))

        assert result["changed"] is False
        # No audit row is claimed: the procedure reports the no-op and the
        # service passes it through unchanged, so a replayed request cannot pad
        # the history.
        assert "role_change_id" not in result

    def test_tc08_the_acting_administrator_is_recorded(self):
        service, repo = _service()
        repo.assign_user_role.return_value = {
            "changed": True, "id": "u2", "display_name": "Kus",
            "role_before": "user", "role_after": "admin",
        }

        run(service.assign_user_role("u2", "admin", "a1"))

        # changed_by comes from the verified caller, never from the body.
        _, _, changed_by = repo.assign_user_role.call_args.args
        assert changed_by == "a1"

    def test_tc09_database_error_propagates(self):
        service, repo = _service()
        repo.assign_user_role.side_effect = RuntimeError("DB connection lost")

        with pytest.raises(RuntimeError):
            run(service.assign_user_role("u2", "admin", "a1"))

    def test_administrator_alias_is_accepted(self):
        """UD-23 writes the actor as "Administrator"; the column holds 'admin'."""
        service, repo = _service()
        repo.assign_user_role.return_value = {
            "changed": True, "id": "u2", "display_name": "Kus",
            "role_before": "user", "role_after": "admin",
        }

        run(service.assign_user_role("u2", "Administrator", "a1"))

        repo.assign_user_role.assert_called_once_with("u2", "admin", "a1")


# --------------------------------------------------------------------------- #
# UTC-54 — list_role_changes (MD-60, SRS-97 reading half)
# --------------------------------------------------------------------------- #
class TestListRoleChanges:
    def test_tc01_returns_the_page_with_its_total(self):
        service, repo = _service()
        rows = [
            {"id": "rc2", "target_user_id": "u2", "changed_by": "a1",
             "role_before": "user", "role_after": "admin"},
            {"id": "rc1", "target_user_id": "u3", "changed_by": "a1",
             "role_before": "admin", "role_after": "user"},
        ]
        repo.list_role_changes.return_value = Page(rows, 7)

        page = run(service.list_role_changes(limit=2, offset=0))

        assert page.items == rows
        assert page.total == 7

    def test_tc02_empty_history(self):
        service, repo = _service()
        repo.list_role_changes.return_value = Page([], 0)

        page = run(service.list_role_changes(limit=50, offset=0))

        assert page.items == []
        assert page.total == 0

    def test_tc03_filters_to_one_account(self):
        service, repo = _service()
        repo.list_role_changes.return_value = Page(
            [{"id": "rc2", "target_user_id": "u2"}], 1,
        )

        page = run(service.list_role_changes(target_user_id="u2"))

        repo.list_role_changes.assert_called_once_with(20, 0, "u2")
        assert all(r["target_user_id"] == "u2" for r in page.items)

    def test_tc04_paging_arguments_are_passed_through(self):
        service, repo = _service()
        repo.list_role_changes.return_value = Page([], 0)

        run(service.list_role_changes(limit=20, offset=40))

        repo.list_role_changes.assert_called_once_with(20, 40, None)

    def test_tc05_database_error_propagates(self):
        service, repo = _service()
        repo.list_role_changes.side_effect = RuntimeError("DB connection lost")

        with pytest.raises(RuntimeError):
            run(service.list_role_changes())
