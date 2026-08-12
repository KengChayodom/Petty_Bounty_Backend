"""
Integration tests for the device_tokens table (FCM push registration, SRS-FR-12).

The table + its constraints are only exercisable against a real Postgres:
UNIQUE(fcm_token), the platform CHECK, the ON DELETE CASCADE, and — the point of
it all — the upsert-on-fcm_token reassign/refresh contract that
SupabaseDeviceTokenRepository.upsert_device_token relies on so a token moving
accounts reassigns cleanly instead of duplicating.

Progress-I SRS traceability: SRS-19 (register/upsert on fcm_token),
SRS-20 (unregister scoped to the caller's own token).
"""
import psycopg
import pytest

from _query import row_count

pytestmark = pytest.mark.integration


def _tokens_for(conn, fcm_token):
    """Read back every (user_id, platform) row carrying this fcm_token."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT user_id, platform FROM device_tokens WHERE fcm_token = %s",
            (fcm_token,),
        )
        return cur.fetchall()


def _upsert(conn, *, user_id, fcm_token, platform):
    """Mirror SupabaseDeviceTokenRepository.upsert_device_token — the register
    route's upsert with fcm_token as the conflict arbiter."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO device_tokens (user_id, fcm_token, platform) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (fcm_token) DO UPDATE SET "
            "  user_id = EXCLUDED.user_id, "
            "  platform = EXCLUDED.platform, "
            "  updated_at = now()",
            (user_id, fcm_token, platform),
        )


def test_upsert_reassigns_token_and_refreshes_in_place(conn, seed):
    # A phone that logs into account B while still holding account A's row for
    # this token must REASSIGN to B — one row, not two — and refresh platform.
    user_a = seed.user()
    user_b = seed.user()
    seed.device_token(user_id=user_a, fcm_token="tok-xyz", platform="android")

    _upsert(conn, user_id=user_b, fcm_token="tok-xyz", platform="ios")

    rows = _tokens_for(conn, "tok-xyz")
    assert len(rows) == 1                      # reassigned in place, not duplicated
    assert rows[0][0] == user_b                # now owned by B
    assert rows[0][1] == "ios"                 # platform refreshed
    assert row_count(conn, "device_tokens", "user_id", user_a) == 0  # A's row gone


def test_delete_is_scoped_to_user_and_token(conn, seed):
    # unregister must only ever drop the CALLER's own token: a delete scoped to
    # a different user_id removes nothing, even with the right token string.
    user_a = seed.user()
    user_b = seed.user()
    seed.device_token(user_id=user_a, fcm_token="tok-a", platform="android")

    with conn.cursor() as cur:
        # wrong user, right token -> no-op (a client can't drop another's token)
        cur.execute(
            "DELETE FROM device_tokens WHERE user_id = %s AND fcm_token = %s",
            (user_b, "tok-a"),
        )
        assert cur.rowcount == 0
        assert row_count(conn, "device_tokens", "fcm_token", "tok-a") == 1

        # owner-scoped delete removes exactly it
        cur.execute(
            "DELETE FROM device_tokens WHERE user_id = %s AND fcm_token = %s",
            (user_a, "tok-a"),
        )
        assert cur.rowcount == 1
    assert row_count(conn, "device_tokens", "fcm_token", "tok-a") == 0


def test_device_tokens_schema_constraints(conn, seed):
    # Plain column/table constraints, folded into one test via savepoints
    # (conn.transaction()) so a raised error doesn't poison the outer txn.
    user_a = seed.user()
    seed.device_token(user_id=user_a, fcm_token="dup", platform="android")

    # (a) UNIQUE(fcm_token): a second plain insert of the same token is rejected.
    with pytest.raises(psycopg.errors.UniqueViolation):
        with conn.transaction():
            seed.device_token(user_id=user_a, fcm_token="dup", platform="ios")

    # (b) platform CHECK: only 'android' | 'ios' are allowed.
    with pytest.raises(psycopg.errors.CheckViolation):
        with conn.transaction():
            seed.device_token(user_id=user_a, fcm_token="tok-web", platform="web")

    # (c) FK ON DELETE CASCADE: removing the owner drops their tokens.
    assert row_count(conn, "device_tokens", "user_id", user_a) == 1
    with conn.cursor() as cur:
        cur.execute("DELETE FROM users WHERE id = %s", (user_a,))
    assert row_count(conn, "device_tokens", "user_id", user_a) == 0
