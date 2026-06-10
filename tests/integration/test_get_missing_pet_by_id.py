"""
Integration test for the deployed get_missing_pet_by_id RPC.

Locks the contract fixed on 2026-06-10: GET /missing-pets/{id} must return the
SAME shape as get_nearby_missing_pets — numeric latitude/longitude projected via
ST_Y/ST_X (NOT the raw last_seen_location geography) and numeric bounty_amount
(NOT the DECIMAL serialised as a string). The old select('*') path returned the
raw row, which had no lat/lng and a string bounty, crashing the Flutter
MissingPetModel.

Runs against a real pgvector+PostGIS database (see conftest) so the ST_Y/ST_X
projection and numeric typing are exercised exactly as deployed, not faked.
"""
from decimal import Decimal

import pytest

pytestmark = pytest.mark.integration

# These are the columns the Flutter MissingPetModel parses as non-null numbers;
# regressing any of them back to text/geography is the bug this test guards.
_NUMERIC_COLS = ("latitude", "longitude", "bounty_amount")


def _by_id(conn, pet_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, owner_id, pet_name, species, characteristics, "
            "       bounty_amount, latitude, longitude, last_seen_time, "
            "       image_url, status, created_at, primary_color_hex, pattern_id "
            "FROM get_missing_pet_by_id(%s)",
            (pet_id,),
        )
        cols = [d.name for d in cur.description]
        row = cur.fetchone()
        return dict(zip(cols, row)) if row else None


def test_projects_numeric_lat_lng_and_bounty(conn, seed):
    owner = seed.user()
    pid = seed.missing_pet(
        owner_id=owner, species="Cat",
        lat=13.7563, lon=100.5018, bounty=1500, pet_name="Mochi",
    )

    rec = _by_id(conn, pid)

    assert rec is not None
    # latitude/longitude must be real numbers from ST_Y/ST_X — not strings, and
    # not a geography/WKT blob. A revert to raw last_seen_location fails here.
    assert isinstance(rec["latitude"], float)
    assert isinstance(rec["longitude"], float)
    assert rec["latitude"] == pytest.approx(13.7563)
    assert rec["longitude"] == pytest.approx(100.5018)

    # bounty_amount must be numeric (Decimal), not the DECIMAL-as-string the old
    # raw-row path produced.
    assert isinstance(rec["bounty_amount"], Decimal)
    assert not isinstance(rec["bounty_amount"], str)
    assert rec["bounty_amount"] == Decimal("1500")

    # Sanity: identity + species projected, and the prod-only color/pattern
    # columns are present in the result shape (NULL is fine here).
    assert rec["id"] == pid
    assert rec["pet_name"] == "Mochi"
    assert rec["species"] == "Cat"
    assert "primary_color_hex" in rec and "pattern_id" in rec


def test_lat_lng_are_not_swapped(conn, seed):
    # ST_Y is latitude, ST_X is longitude. A swap would still return floats, so
    # an asymmetric point is used to catch a transposed projection.
    owner = seed.user()
    pid = seed.missing_pet(owner_id=owner, lat=1.0, lon=2.0)

    rec = _by_id(conn, pid)

    assert (rec["latitude"], rec["longitude"]) == (pytest.approx(1.0),
                                                   pytest.approx(2.0))


def test_unknown_id_returns_no_row(conn):
    import uuid
    assert _by_id(conn, uuid.uuid4()) is None
