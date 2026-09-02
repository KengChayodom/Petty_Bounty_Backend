"""
Integration tests for the role-assignment procedures — find_user_by_email() and
assign_user_role() (SRS-94 to SRS-98, MD-58/MD-59,
migrations/2026_09_02_role_assignment.sql).

Everything worth testing here lives in the database and is invisible to the unit
suite, which stops at the repository port:

  * the two guards of SRS-96 — self-demotion and the last administrator — are
    enforced by the procedure, not by the service, because a count read in
    Python can be invalidated before the write lands;
  * atomicity — a refused change leaves the role AND the audit trail untouched,
    so nothing records a grant that did not happen;
  * the no-op — assigning a role an account already holds writes no audit row,
    which is what stops a replayed request padding the history (and is why
    `role_changes` carries a CHECK that the two roles differ);
  * `find_user_by_email` reaching across into `auth.users`, matching the whole
    address and never a prefix — the line between MD-58 and the account search
    struck on 2026-08-21.

The one property NOT covered: that the guards hold when two administrators act
at the same instant. The harness gives every test one connection inside a
transaction that is rolled back, so a second connection cannot see the seeded
rows at all. What is testable here is that the guard is in the procedure rather
than the service, which is the decision that makes the concurrent case safe.

Progress-II SRS traceability: SRS-94, SRS-95, SRS-96, SRS-97.
"""
import json
import uuid

import psycopg
import pytest

pytestmark = pytest.mark.integration


def _assign(conn, *, target, role, changed_by):
    """Mirror SupabaseUserRepository.assign_user_role."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT assign_user_role(%s, %s::user_role, %s)",
            (target, role, changed_by),
        )
        out = cur.fetchone()[0]
    return out if isinstance(out, dict) else json.loads(out)


def _assign_refused(conn, *, target, role, changed_by):
    """Call assign_user_role expecting a RAISE, and return the message.

    The call runs inside a SAVEPOINT (`conn.transaction()` nests one when a
    transaction is already open) because a RAISE aborts the transaction it runs
    in. A bare `conn.rollback()` here would also discard the seeded rows, and
    the assertions that follow — "the role is unchanged", "no audit row was
    written" — would then pass against an empty database, which is the shape of
    a test that cannot fail.
    """
    with pytest.raises(psycopg.errors.RaiseException) as exc:
        with conn.transaction():
            _assign(conn, target=target, role=role, changed_by=changed_by)
    return str(exc.value)


def _lookup(conn, email):
    """Mirror SupabaseUserRepository.find_by_email."""
    with conn.cursor() as cur:
        cur.execute("SELECT find_user_by_email(%s)", (email,))
        out = cur.fetchone()[0]
    if out is None:
        return None
    return out if isinstance(out, dict) else json.loads(out)


def _role(conn, user_id):
    with conn.cursor() as cur:
        cur.execute("SELECT role FROM users WHERE id = %s", (user_id,))
        return cur.fetchone()[0]


def _changes(conn, user_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT role_before, role_after FROM role_changes "
            "WHERE target_user_id = %s ORDER BY created_at",
            (user_id,),
        )
        return cur.fetchall()


def _set_email(conn, user_id, email):
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE auth.users SET email = %s WHERE id = %s", (email, user_id)
        )


# --------------------------------------------------------------------------- #
# find_user_by_email — MD-58 / SRS-94
# --------------------------------------------------------------------------- #
class TestFindUserByEmail:
    def test_resolves_the_exact_address(self, conn, seed):
        uid = seed.user(display_name="Kus", role="user")
        _set_email(conn, uid, "kus@example.com")

        row = _lookup(conn, "kus@example.com")

        assert row is not None
        assert uuid.UUID(row["id"]) == uid
        assert row["display_name"] == "Kus"
        assert row["role"] == "user"

    def test_address_matching_is_case_insensitive(self, conn, seed):
        uid = seed.user()
        _set_email(conn, uid, "kus@example.com")

        assert _lookup(conn, "KUS@Example.COM") is not None

    def test_a_prefix_matches_nothing(self, conn, seed):
        uid = seed.user()
        _set_email(conn, uid, "kus@example.com")

        # The line between this and the struck account search: no partial match,
        # so an administrator cannot walk the address space from a fragment.
        assert _lookup(conn, "kus@example.co") is None
        assert _lookup(conn, "kus") is None
        assert _lookup(conn, "%@example.com") is None

    def test_unknown_address_returns_null(self, conn, seed):
        seed.user()
        assert _lookup(conn, "nobody@example.com") is None

    def test_auth_account_without_a_profile_row_returns_null(self, conn):
        # The join is INNER on purpose: an auth account with no `users` row has
        # no role to report and nothing to assign one to.
        orphan = uuid.uuid4()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO auth.users (id, email) VALUES (%s, %s)",
                (orphan, "orphan@example.com"),
            )

        assert _lookup(conn, "orphan@example.com") is None


# --------------------------------------------------------------------------- #
# assign_user_role — MD-59 / SRS-95, SRS-96, SRS-97
# --------------------------------------------------------------------------- #
class TestAssignUserRole:
    def test_grants_the_role_and_records_the_change(self, conn, seed):
        boss = seed.user(display_name="Boss", role="admin")
        target = seed.user(display_name="Kus", role="user")

        out = _assign(conn, target=target, role="admin", changed_by=boss)

        assert out["changed"] is True
        assert (out["role_before"], out["role_after"]) == ("user", "admin")
        assert _role(conn, target) == "admin"
        assert _changes(conn, target) == [("user", "admin")]

    def test_withdraws_the_role_when_another_admin_remains(self, conn, seed):
        boss = seed.user(role="admin")
        other = seed.user(role="admin")

        out = _assign(conn, target=other, role="user", changed_by=boss)

        assert out["changed"] is True
        assert _role(conn, other) == "user"
        assert _changes(conn, other) == [("admin", "user")]

    def test_refuses_self_demotion(self, conn, seed):
        boss = seed.user(role="admin")
        seed.user(role="admin")  # a second admin, so guard 2 cannot be the cause

        msg = _assign_refused(conn, target=boss, role="user", changed_by=boss)

        assert "own administrator access" in msg
        assert _role(conn, boss) == "admin"

    def test_refuses_removing_the_last_administrator(self, conn, seed):
        only_admin = seed.user(role="admin")

        # Asking for their own withdrawal hits the self-demotion guard first.
        msg = _assign_refused(
            conn, target=only_admin, role="user", changed_by=only_admin,
        )
        assert "own administrator access" in msg

        # With somebody else asking, the count guard is what refuses — and that
        # is the one that keeps the console reachable at all.
        asker = seed.user(role="user")
        msg = _assign_refused(
            conn, target=only_admin, role="user", changed_by=asker,
        )
        assert "last administrator" in msg

        assert _role(conn, only_admin) == "admin"

    def test_a_refused_change_writes_no_audit_row(self, conn, seed):
        only_admin = seed.user(role="admin")
        asker = seed.user(role="user")

        _assign_refused(conn, target=only_admin, role="user", changed_by=asker)

        # Atomicity: nothing records a withdrawal that never happened. The
        # account is still here to be asked about — see `_assign_refused`.
        assert _role(conn, only_admin) == "admin"
        assert _changes(conn, only_admin) == []

    def test_assigning_the_role_already_held_is_a_no_op(self, conn, seed):
        boss = seed.user(role="admin")
        target = seed.user(role="user")

        out = _assign(conn, target=target, role="user", changed_by=boss)

        assert out["changed"] is False
        assert (out["role_before"], out["role_after"]) == ("user", "user")
        # No audit row: a replayed request cannot pad the history.
        assert _changes(conn, target) == []

    def test_unknown_account_raises(self, conn, seed):
        boss = seed.user(role="admin")

        msg = _assign_refused(
            conn, target=uuid.uuid4(), role="admin", changed_by=boss,
        )
        assert "not found" in msg

    def test_role_changes_rejects_a_no_op_row(self, conn, seed):
        # The CHECK is what makes the no-op unrepresentable rather than merely
        # unwritten: nothing can record a change that changed nothing.
        boss = seed.user(role="admin")
        target = seed.user(role="user")

        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO role_changes "
                        "(target_user_id, changed_by, role_before, role_after) "
                        "VALUES (%s, %s, 'user'::user_role, 'user'::user_role)",
                        (target, boss),
                    )

        assert _changes(conn, target) == []

    def test_the_history_survives_the_acting_admin_being_deleted(
        self, conn, seed,
    ):
        # changed_by is ON DELETE SET NULL: removing the administrator who acted
        # must not erase the fact that the change happened.
        boss = seed.user(role="admin")
        seed.user(role="admin")
        target = seed.user(role="user")
        _assign(conn, target=target, role="admin", changed_by=boss)

        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id = %s", (boss,))
            cur.execute(
                "SELECT changed_by, role_before, role_after FROM role_changes "
                "WHERE target_user_id = %s",
                (target,),
            )
            rows = cur.fetchall()

        assert rows == [(None, "user", "admin")]
