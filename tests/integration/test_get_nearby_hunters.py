"""
Integration tests for the deployed get_nearby_hunters RPC (SRS-FR-12 geo push).

Progress-I SRS traceability: SRS-24 (10 km proximity radius — test_radius_filters_out_far_hunter)
and SRS-25 (owner exclusion — test_excludes_owner_via_exclude_user_id,
test_null_exclude_user_id_returns_everyone).

These run against a real pgvector+PostGIS database (the FCM migration applied
verbatim — see conftest) because the behaviour we care about lives entirely in
SQL: the staleness window, the NULL filters, the owner exclusion, and the
proximity radius. Faking the SQL would test nothing.

The function signature is:
    get_nearby_hunters(center GEOGRAPHY, radius_meters FLOAT,
                       max_age_hours INT = 24, exclude_user_id UUID = NULL)
returning (user_id, distance_meters, last_location_at) ordered by distance.

All hunters are co-located at the BKK default unless a test moves them, so any
exclusion observed is due to the filter under test, not distance — except the
radius test, which is the one that varies position.
"""
import pytest

pytestmark = pytest.mark.integration

# Default center used by the seeder; the RPC is queried at the same point.
_LAT, _LON = 13.7563, 100.5018


def _nearby(conn, *, lat=_LAT, lon=_LON, radius_km=10.0,
            max_age_hours=24, exclude=None):
    """Call the RPC with a WKT center cast to geography (mirrors the service)."""
    center = f"POINT({lon} {lat})"
    with conn.cursor() as cur:
        cur.execute(
            "SELECT user_id, distance_meters, last_location_at "
            "FROM get_nearby_hunters(%s::geography, %s, %s, %s)",
            (center, radius_km * 1000.0, max_age_hours, exclude),
        )
        return cur.fetchall()


def _ids(rows):
    return {r[0] for r in rows}


def test_excludes_stale_hunter(conn, seed):
    # Two co-located hunters; one located 1h ago (fresh), one 48h ago (stale).
    fresh = seed.user()
    stale = seed.user()
    seed.set_location(fresh, age_hours=1)
    seed.set_location(stale, age_hours=48)

    rows = _nearby(conn, max_age_hours=24)

    # Guards the freshness filter: a 2-day-old location must never be "nearby".
    assert _ids(rows) == {fresh}


def test_max_age_boundary_just_inside_and_outside(conn, seed):
    # Boundary value analysis around the 24h window: 23h in, 25h out.
    inside = seed.user()
    outside = seed.user()
    seed.set_location(inside, age_hours=23)
    seed.set_location(outside, age_hours=25)

    rows = _nearby(conn, max_age_hours=24)

    assert _ids(rows) == {inside}


def test_excludes_null_location_and_null_timestamp(conn, seed):
    # Four hunters exercising both arms of the (location IS NOT NULL AND
    # last_location_at IS NOT NULL) filter independently.
    valid = seed.user()
    seed.set_location(valid, age_hours=1)

    both_null = seed.user()  # never shared location -> both columns NULL

    loc_no_ts = seed.user()  # has a location but NULL timestamp
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE users SET "
            "  last_location = ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, "
            "  last_location_at = NULL WHERE id = %s",
            (_LON, _LAT, loc_no_ts),
        )

    ts_no_loc = seed.user()  # has a timestamp but NULL location
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE users SET last_location = NULL, last_location_at = now() "
            "WHERE id = %s",
            (ts_no_loc,),
        )

    rows = _nearby(conn)

    # Only the fully-located hunter is targetable; each NULL arm is excluded.
    assert _ids(rows) == {valid}


def test_excludes_owner_via_exclude_user_id(conn, seed):
    owner = seed.user()
    other = seed.user()
    seed.set_location(owner, age_hours=1)
    seed.set_location(other, age_hours=1)

    rows = _nearby(conn, exclude=owner)

    # The pet owner must not be alerted about their own pet.
    assert _ids(rows) == {other}


def test_null_exclude_user_id_returns_everyone(conn, seed):
    # THE specifically-guarded bug: the filter is
    #   (exclude_user_id IS NULL OR u.id <> exclude_user_id)
    # If the `IS NULL` arm is dropped, `u.id <> NULL` is NULL for every row and
    # the RPC returns ZERO hunters. With exclude=None, BOTH must come back.
    a = seed.user()
    b = seed.user()
    seed.set_location(a, age_hours=1)
    seed.set_location(b, age_hours=1)

    rows = _nearby(conn, exclude=None)

    assert _ids(rows) == {a, b}


def test_radius_filters_out_far_hunter(conn, seed):
    # near ~5.5 km (0.05 deg lat); far ~22 km (0.20 deg lat). Radius 10 km.
    near = seed.user()
    far = seed.user()
    seed.set_location(near, lat=_LAT + 0.05, lon=_LON, age_hours=1)
    seed.set_location(far, lat=_LAT + 0.20, lon=_LON, age_hours=1)

    rows = _nearby(conn, radius_km=10.0)

    assert _ids(rows) == {near}
    # distance is returned in metres; the near hunter is ~5.5 km out.
    assert 5_000 < dict((r[0], r[1]) for r in rows)[near] < 6_000


def test_orders_by_distance_ascending(conn, seed):
    closer = seed.user()
    farther = seed.user()
    seed.set_location(closer, lat=_LAT + 0.02, lon=_LON, age_hours=1)   # ~2.2 km
    seed.set_location(farther, lat=_LAT + 0.06, lon=_LON, age_hours=1)  # ~6.6 km

    rows = _nearby(conn, radius_km=10.0)

    # ORDER BY distance_meters — nearest first.
    assert [r[0] for r in rows] == [closer, farther]
