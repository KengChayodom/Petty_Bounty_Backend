"""
Integration-test harness: a real, ephemeral Postgres (pgvector + PostGIS).

Lifecycle:
  * session: build the combined pgvector+PostGIS image if absent, start ONE
    disposable container, apply the real schema (prelude -> real Feature#2
    migration -> live match RPC), expose the DSN.
  * per-test: hand out a connection inside an OPEN transaction that is ALWAYS
    rolled back on teardown. Tests seed + call RPCs + read back inside that
    one transaction, so every test starts from the clean schema regardless of
    order — the suite is safe to run 1000× with no data drift.

These tests are gated behind `--integration` (see the root tests/conftest.py),
so the fast unit suite never pays for a Docker spin-up.
"""
import os
import subprocess
import sys
import uuid
from pathlib import Path

import psycopg
import pytest
from testcontainers.postgres import PostgresContainer

# Make sibling modules (e.g. _helpers) importable as top-level from test files.
sys.path.insert(0, os.path.dirname(__file__))

from _helpers import vec_literal  # noqa: E402

INTEG_DIR = Path(__file__).parent
SQL_DIR = INTEG_DIR / "sql"
BACKEND_ROOT = INTEG_DIR.parent.parent
REAL_MIGRATION = BACKEND_ROOT / "migrations" / "2026_05_27_feature2_logging_scoring.sql"
FCM_MIGRATION = BACKEND_ROOT / "migrations" / "2026_06_10_feature_fcm_push.sql"
BYID_MIGRATION = BACKEND_ROOT / "migrations" / "2026_06_10_fix_get_missing_pet_by_id.sql"
SM_UNIQUE_MIGRATION = BACKEND_ROOT / "migrations" / "2026_06_10_fix_sighting_matches_unique.sql"
PENALTY_MIGRATION = BACKEND_ROOT / "migrations" / "2026_08_20_flag_penalty_not_ban.sql"
OWNER_MIGRATION = BACKEND_ROOT / "migrations" / "2026_08_21_owner_driven_resolution.sql"
EXPIRES_AT_MIGRATION = BACKEND_ROOT / "migrations" / "2026-09-01_post_expires_at.sql"
ROLE_MIGRATION = BACKEND_ROOT / "migrations" / "2026_09_02_role_assignment.sql"
OWNER_DETAIL_MIGRATION = BACKEND_ROOT / "migrations" / "2026_09_02_pet_owner_details.sql"
HUNTER_DETAIL_MIGRATION = (
    BACKEND_ROOT / "migrations" / "2026_09_02_sighting_hunter_details.sql"
)
DROP_PATTERN_MIGRATION = (
    BACKEND_ROOT / "migrations" / "2026_09_05_drop_pattern_id.sql"
)
IMAGE_TAG = "petty-bounty-test-pg:pg16"


# --------------------------------------------------------------------------- #
# Image + container + schema (session scope)
# --------------------------------------------------------------------------- #
def _ensure_image() -> None:
    """Build the pgvector+PostGIS image once; reuse it thereafter."""
    present = subprocess.run(
        ["docker", "image", "inspect", IMAGE_TAG],
        capture_output=True,
    ).returncode == 0
    if not present:
        subprocess.run(
            ["docker", "build", "-f", str(INTEG_DIR / "Dockerfile.pg"),
             "-t", IMAGE_TAG, str(INTEG_DIR)],
            check=True,
        )


def _apply_schema(dsn: str) -> None:
    """Apply, in order: prelude -> Feature#2 migration -> FCM migration -> live RPC.

    Each file is sent as one script under autocommit so the migration's own
    BEGIN/COMMIT blocks (it must split the enum-value add into its own
    transaction) are honoured exactly as in production. The FCM, by-id, and
    sighting_matches-UNIQUE migrations are applied verbatim so their constraints
    /RPCs are tested as deployed, not faked. The color/pattern shim adds two
    prod-only columns the by-id RPC projects (see that file).
    """
    files = [
        SQL_DIR / "00_prelude.sql",
        REAL_MIGRATION,              # applied verbatim from the repo — not a copy
        FCM_MIGRATION,               # adds get_nearby_hunters + users.last_location
        SQL_DIR / "10_color_pattern_columns.sql",  # prod-only col by-id projects
        BYID_MIGRATION,              # adds get_missing_pet_by_id (the fixed shape)
        SM_UNIQUE_MIGRATION,         # adds sighting_matches UNIQUE (the upsert arbiter)
        PENALTY_MIGRATION,           # renames Reviewed_Ban, adds score_penalties + RPC
        SQL_DIR / "20_live_match_rpc.sql",
        OWNER_MIGRATION,             # owner_decide_sighting + the de-fanged resolve
        EXPIRES_AT_MIGRATION,        # missing_pets.expires_at; read paths filter it
        ROLE_MIGRATION,              # role_changes + find_user_by_email + assign_user_role
        OWNER_DETAIL_MIGRATION,      # get_missing_pet_by_id projects the owner's contact
        HUNTER_DETAIL_MIGRATION,     # sightings_for_pet projects the hunter's contact
        DROP_PATTERN_MIGRATION,      # drops missing_pets.pattern_id, rebuilding the
                                     # two RPCs that projected it. MUST stay last of
                                     # the two by-id/nearby definitions, or an earlier
                                     # migration re-creates a function selecting a
                                     # column the shim no longer adds.
    ]
    with psycopg.connect(dsn, autocommit=True) as conn:
        for f in files:
            # Force UTF-8: the migration files contain UTF-8 punctuation (em-dash
            # etc.). Path.read_text() otherwise uses the OS default encoding,
            # which is cp874 on a Thai-locale Windows and fails to decode them.
            conn.execute(f.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def pg_dsn():
    _ensure_image()
    with PostgresContainer(IMAGE_TAG, driver=None) as container:
        dsn = container.get_connection_url()  # postgresql://user:pass@host:port/db
        _apply_schema(dsn)
        yield dsn


@pytest.fixture
def conn(pg_dsn):
    """A connection in an open transaction, rolled back after each test."""
    c = psycopg.connect(pg_dsn)  # autocommit=False by default
    try:
        yield c
    finally:
        c.rollback()
        c.close()


# --------------------------------------------------------------------------- #
# Seed factories — build valid rows with overridable fields (skill §5).
# Each returns the new row's UUID. All writes land in the test's transaction.
# --------------------------------------------------------------------------- #
class Seeder:
    def __init__(self, conn):
        self.conn = conn

    def user(self, display_name="Hunter", role="user", total_score=0,
             phone=None, profile_image_url=None) -> uuid.UUID:
        """A profile row (+ its auth.users parent).

        `phone` / `profile_image_url` default to NULL — the state of a real
        account that never filled them in, which is exactly the case the read
        RPCs have to survive.
        """
        uid = uuid.uuid4()
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO auth.users (id, email) VALUES (%s, %s)",
                (uid, f"{uid}@test.local"),
            )
            cur.execute(
                "INSERT INTO users "
                "  (id, display_name, role, total_score, phone, profile_image_url) "
                "VALUES (%s, %s, %s::user_role, %s, %s, %s)",
                (uid, display_name, role, total_score, phone, profile_image_url),
            )
        return uid

    def set_location(self, user_id, *, lat=13.7563, lon=100.5018,
                     age_hours=0) -> None:
        """Set a hunter's last_location + last_location_at (age_hours in the past).

        Drives the get_nearby_hunters freshness + proximity filters. age_hours=0
        is 'just now' (fresh); age_hours=48 is stale relative to a 24h window.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET "
                "  last_location = ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, "
                "  last_location_at = now() - make_interval(hours => %s) "
                "WHERE id = %s",
                (lon, lat, age_hours, user_id),
            )

    def missing_pet(self, *, owner_id=None, species="Cat", status="Searching",
                    lat=13.7563, lon=100.5018, vector=None, bounty=1000,
                    pet_name="Pet", age_days=0) -> uuid.UUID:
        # age_days backdates BOTH created_at and expires_at: the match RPC and
        # the map query filter `expires_at > NOW()` (2026-09-01 migration), and
        # expires_at is granted 7 days from filing, so a post seeded age_days
        # old expires 7 - age_days from now — negative once age_days > 7. This
        # is the only way a test can seed a post that has already aged out.
        pid = uuid.uuid4()
        vec = vec_literal(vector) if vector is not None else None
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO missing_pets "
                "(id, owner_id, pet_name, species, characteristics, bounty_amount, "
                " last_seen_location, last_seen_time, image_url, feature_vector, status, "
                " created_at, expires_at) "
                "VALUES (%s, %s, %s, %s::pet_species, '{}'::jsonb, %s, "
                "        ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, NOW(), "
                "        %s, %s::vector, %s::pet_status, "
                "        NOW() - make_interval(days => %s), "
                "        NOW() - make_interval(days => %s) + INTERVAL '7 days')",
                (pid, owner_id, pet_name, species, bounty, lon, lat,
                 "http://img/pet.jpg", vec, status, age_days, age_days),
            )
        return pid

    def sighting(self, *, hunter_id=None, species="Cat", lat=13.7563, lon=100.5018,
                 vector=None, action="Spotted", verification="Verified",
                 initial_target_pet_id=None, status="Pending_Analysis",
                 created_at=None) -> uuid.UUID:
        # created_at is overridable so tests can control chronological ordering
        # (earliest-verified-sighting F1 rule, and DESC pagination); NULL -> NOW().
        sid = uuid.uuid4()
        vec = vec_literal(vector) if vector is not None else None
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sightings "
                "(id, hunter_id, sighted_location, image_url, detected_species, "
                " feature_vector, initial_target_pet_id, action_type, "
                " sighting_status, verification_status, created_at) "
                "VALUES (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, "
                "        %s, %s::pet_species, %s::vector, %s, %s::action_type, "
                "        %s::sighting_status, %s::verification_status, "
                "        COALESCE(%s::timestamptz, NOW()))",
                (sid, hunter_id, lon, lat, "http://img/s.jpg", species, vec,
                 initial_target_pet_id, action, status, verification, created_at),
            )
        return sid

    def sighting_match(self, *, sighting_id, missing_pet_id, similarity=0.8,
                       owner_status="Pending") -> None:
        # owner_status is the owner's verdict, and since 2026-08-21 it is what
        # scoring reads — so tests need to seed a queue in every state.
        # similarity=None is the targeted shape: a queue row that is not an AI
        # match (sightings_for_pet keys off exactly that).
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sighting_matches "
                "(sighting_id, missing_pet_id, similarity_score, owner_status) "
                "VALUES (%s, %s, %s, %s::owner_decision)",
                (sighting_id, missing_pet_id, similarity, owner_status),
            )

    def report(self, *, sighting_id=None, reporter_id=None, reason="Not_a_pet",
               status="Pending") -> uuid.UUID:
        """A moderation flag — one user's complaint about a sighting."""
        rid = uuid.uuid4()
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO reports (id, reporter_id, sighting_id, reason, status) "
                "VALUES (%s, %s, %s, %s::report_reason, %s::report_status)",
                (rid, reporter_id, sighting_id, reason, status),
            )
        return rid

    def device_token(self, *, user_id, fcm_token="tok", platform="android") -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO device_tokens (user_id, fcm_token, platform) "
                "VALUES (%s, %s, %s)",
                (user_id, fcm_token, platform),
            )


@pytest.fixture
def seed(conn):
    return Seeder(conn)
