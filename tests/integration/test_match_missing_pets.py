"""
Integration tests for the deployed matcher match_missing_pets(uuid, int).

Verifies, against a real pgvector+PostGIS database, the five things only this
layer can: cosine ranking, the hardcoded 10 km radius, species + status
filters, the no-threshold NULL-vector fallback ordering, and the missing-vector
RAISE. All vectors are hand-built (see _helpers) so similarities are exact.
"""
import uuid

import psycopg
import pytest
from pytest import approx

from _helpers import HALF_MIX_SIMILARITY, half_mix, unit_vec

pytestmark = pytest.mark.integration


def _match(conn, sighting_id, limit=5):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, pet_name, similarity, distance_meters, status "
            "FROM match_missing_pets(%s, %s)",
            (sighting_id, limit),
        )
        return cur.fetchall()


def test_ranks_by_cosine_similarity_descending(conn, seed):
    owner = seed.user()
    perfect = seed.missing_pet(owner_id=owner, species="Cat", vector=unit_vec(0))
    middle = seed.missing_pet(owner_id=owner, species="Cat", vector=half_mix(0, 1))
    orthogonal = seed.missing_pet(owner_id=owner, species="Cat", vector=unit_vec(1))
    sighting = seed.sighting(species="Cat", vector=unit_vec(0))

    rows = _match(conn, sighting)

    assert [r[0] for r in rows] == [perfect, middle, orthogonal]
    assert rows[0][2] == approx(1.0)
    assert rows[1][2] == approx(HALF_MIX_SIMILARITY)
    assert rows[2][2] == approx(0.0, abs=1e-9)
    assert all(r[3] < 10_000 for r in rows)  # all co-located -> within radius


def test_match_limit_caps_results(conn, seed):
    owner = seed.user()
    perfect = seed.missing_pet(owner_id=owner, species="Cat", vector=unit_vec(0))
    middle = seed.missing_pet(owner_id=owner, species="Cat", vector=half_mix(0, 1))
    seed.missing_pet(owner_id=owner, species="Cat", vector=unit_vec(1))
    sighting = seed.sighting(species="Cat", vector=unit_vec(0))

    rows = _match(conn, sighting, limit=2)

    assert [r[0] for r in rows] == [perfect, middle]  # top-2 by similarity only


def test_filters_out_other_species(conn, seed):
    owner = seed.user()
    cat = seed.missing_pet(owner_id=owner, species="Cat", vector=unit_vec(0))
    # A dog with an IDENTICAL vector + location must still be excluded.
    seed.missing_pet(owner_id=owner, species="Dog", vector=unit_vec(0))
    sighting = seed.sighting(species="Cat", vector=unit_vec(0))

    rows = _match(conn, sighting)

    assert [r[0] for r in rows] == [cat]


def test_filters_out_non_searching_status(conn, seed):
    owner = seed.user()
    searching = seed.missing_pet(owner_id=owner, species="Cat",
                                 vector=unit_vec(0), status="Searching")
    # 'Found' pet with a perfect vector must be excluded (only 'Searching').
    seed.missing_pet(owner_id=owner, species="Cat", vector=unit_vec(0), status="Found")
    sighting = seed.sighting(species="Cat", vector=unit_vec(0))

    rows = _match(conn, sighting)

    assert [r[0] for r in rows] == [searching]


def test_radius_cutoff_at_10km(conn, seed):
    owner = seed.user()
    # sighting at the default coordinate; ~0.05 deg lat north ~= 5.5 km (in),
    # ~0.2 deg lat north ~= 22 km (out). Both have a perfect vector.
    near = seed.missing_pet(owner_id=owner, species="Cat", vector=unit_vec(0),
                            lat=13.7563 + 0.05, lon=100.5018)
    seed.missing_pet(owner_id=owner, species="Cat", vector=unit_vec(0),
                     lat=13.7563 + 0.20, lon=100.5018)
    sighting = seed.sighting(species="Cat", vector=unit_vec(0),
                             lat=13.7563, lon=100.5018)

    rows = _match(conn, sighting)

    assert [r[0] for r in rows] == [near]
    assert 5_000 < rows[0][3] < 6_000  # ~5.5 km


def test_null_vector_pet_is_returned_last_with_zero_similarity(conn, seed):
    # The 2-arg overload has NO similarity threshold, so a Searching, in-range,
    # right-species pet with a NULL feature_vector is still returned — its
    # similarity COALESCEs to 0 and it sorts last via NULLS LAST.
    owner = seed.user()
    perfect = seed.missing_pet(owner_id=owner, species="Cat", vector=unit_vec(0))
    null_vec = seed.missing_pet(owner_id=owner, species="Cat", vector=None)
    sighting = seed.sighting(species="Cat", vector=unit_vec(0))

    rows = _match(conn, sighting)

    assert [r[0] for r in rows] == [perfect, null_vec]
    assert rows[1][2] == approx(0.0)


def test_raises_when_sighting_has_no_vector(conn, seed):
    sighting = seed.sighting(species="Cat", vector=None)  # NULL feature_vector
    with pytest.raises(psycopg.errors.RaiseException):
        with conn.transaction():  # savepoint: keeps the outer txn usable
            _match(conn, sighting)


def test_raises_for_unknown_sighting_id(conn):
    with pytest.raises(psycopg.errors.RaiseException):
        with conn.transaction():
            _match(conn, uuid.uuid4())
