"""
Integration tests for the sighting_matches UNIQUE constraint — schema-enforced
behaviour that only a real Postgres can verify.
"""
import psycopg
import pytest
from pytest import approx

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# sighting_matches UNIQUE(sighting_id, missing_pet_id)
# --------------------------------------------------------------------------- #
def test_duplicate_sighting_match_violates_unique_constraint(conn, seed):
    owner = seed.user()
    pet = seed.missing_pet(owner_id=owner)
    sighting = seed.sighting()
    seed.sighting_match(sighting_id=sighting, missing_pet_id=pet, similarity=0.5)

    with pytest.raises(psycopg.errors.UniqueViolation):
        with conn.transaction():
            seed.sighting_match(sighting_id=sighting, missing_pet_id=pet, similarity=0.9)

    # original row survives unchanged
    with conn.cursor() as cur:
        cur.execute(
            "SELECT similarity_score FROM sighting_matches "
            "WHERE sighting_id = %s AND missing_pet_id = %s", (sighting, pet))
        rows = cur.fetchall()
    assert len(rows) == 1
    assert float(rows[0][0]) == approx(0.5)


def test_upsert_on_conflict_refreshes_in_place(conn, seed):
    # Mirrors the backend _persist_matches upsert (on_conflict=
    # "sighting_id,missing_pet_id"): a re-match refreshes the score, no dupes.
    owner = seed.user()
    pet = seed.missing_pet(owner_id=owner)
    sighting = seed.sighting()
    seed.sighting_match(sighting_id=sighting, missing_pet_id=pet, similarity=0.5)

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sighting_matches (sighting_id, missing_pet_id, similarity_score) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (sighting_id, missing_pet_id) "
            "DO UPDATE SET similarity_score = EXCLUDED.similarity_score",
            (sighting, pet, 0.9))
        cur.execute(
            "SELECT similarity_score FROM sighting_matches "
            "WHERE sighting_id = %s AND missing_pet_id = %s", (sighting, pet))
        rows = cur.fetchall()

    assert len(rows) == 1                       # still one row
    assert float(rows[0][0]) == approx(0.9)     # refreshed in place
