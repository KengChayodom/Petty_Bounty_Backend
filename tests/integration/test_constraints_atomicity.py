"""
Integration tests for resolve_missing_pet + the sighting_matches UNIQUE
constraint — the transactional, schema-enforced behaviours that only a real
Postgres can verify.

resolve_missing_pet is a single-transaction payout: it pays the bounty to the
final (Caught+Verified) hunter, distributes F1 clue points 25/15/10/5/5/… to
every OTHER distinct hunter with a verified sighting, and marks the pet
Resolved — all or nothing.

NOTE on F1 ranks: the function assigns rank in hunter_id (UUID) order, so WHICH
hunter gets 25 vs 15 is non-deterministic. We therefore assert the MULTISET of
points/ranks, never a per-hunter mapping.
"""
from datetime import datetime, timedelta, timezone

import psycopg
import pytest
from pytest import approx

from _query import pet_status, row_count, total_score

pytestmark = pytest.mark.integration

_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _resolve(conn, pet_id, final_sighting_id, verified_by,
             slip="http://slip.jpg", ref="REF-1"):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT resolve_missing_pet(%s, %s, %s, %s, %s)",
            (pet_id, final_sighting_id, slip, ref, verified_by),
        )
        return cur.fetchone()[0]  # jsonb -> dict


def _awards(conn, pet_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT user_id, sighting_id, points, rank "
            "FROM score_awards WHERE missing_pet_id = %s",
            (pet_id,),
        )
        return cur.fetchall()


# --------------------------------------------------------------------------- #
# resolve_missing_pet — happy path
# --------------------------------------------------------------------------- #
def test_resolve_pays_bounty_marks_resolved_and_distributes_f1(conn, seed):
    owner = seed.user()
    admin = seed.user(role="admin")
    pet = seed.missing_pet(owner_id=owner, bounty=5000)

    final_hunter = seed.user()
    final_sighting = seed.sighting(hunter_id=final_hunter, initial_target_pet_id=pet,
                                   action="Caught", verification="Verified")
    clue_hunters = [seed.user() for _ in range(4)]
    for h in clue_hunters:
        seed.sighting(hunter_id=h, initial_target_pet_id=pet,
                      action="Spotted", verification="Verified")

    result = _resolve(conn, pet, final_sighting, admin)

    # 1. pet marked Resolved
    assert pet_status(conn, pet) == "Resolved"

    # 2. exactly one bounty transaction, to the owner, for the bounty amount
    with conn.cursor() as cur:
        cur.execute(
            "SELECT amount, status, owner_id, sighting_id "
            "FROM bounty_transactions WHERE missing_pet_id = %s", (pet,))
        bounty = cur.fetchall()
    assert len(bounty) == 1
    assert float(bounty[0][0]) == approx(5000)
    assert bounty[0][1] == "Verified"
    assert bounty[0][2] == owner
    assert bounty[0][3] == final_sighting

    # 3. F1: one award per OTHER distinct hunter, points multiset {25,15,10,5}
    awards = _awards(conn, pet)
    assert len(awards) == 4
    assert sorted(a[2] for a in awards) == [5, 10, 15, 25]
    assert sorted(a[3] for a in awards) == [1, 2, 3, 4]
    assert final_hunter not in {a[0] for a in awards}  # final hunter excluded

    # 4. each clue hunter's total_score == the points they were awarded
    assert sorted(total_score(conn, h) for h in clue_hunters) == [5, 10, 15, 25]

    # 5. returned JSON summary
    assert result["final_hunter_id"] == str(final_hunter)
    assert float(result["bounty_amount"]) == approx(5000)
    assert len(result["awards"]) == 4


# --------------------------------------------------------------------------- #
# resolve_missing_pet — F1 spam protection
# --------------------------------------------------------------------------- #
def test_f1_awards_each_hunter_once_for_earliest_verified_sighting(conn, seed):
    owner = seed.user()
    admin = seed.user(role="admin")
    pet = seed.missing_pet(owner_id=owner, bounty=1000)

    final_hunter = seed.user()
    final_sighting = seed.sighting(hunter_id=final_hunter, initial_target_pet_id=pet,
                                   action="Caught", verification="Verified")

    # ONE clue hunter with TWO verified sightings for this pet.
    spammer = seed.user()
    earliest = seed.sighting(hunter_id=spammer, initial_target_pet_id=pet,
                             verification="Verified", created_at=_BASE)
    seed.sighting(hunter_id=spammer, initial_target_pet_id=pet,
                  verification="Verified", created_at=_BASE + timedelta(days=1))

    _resolve(conn, pet, final_sighting, admin)

    awards = [a for a in _awards(conn, pet) if a[0] == spammer]
    assert len(awards) == 1                 # awarded once despite two sightings
    assert awards[0][1] == earliest         # credited to the earliest verified one
    assert awards[0][2] == 25               # sole clue hunter -> rank 1 -> 25 pts
    assert total_score(conn, spammer) == 25


# --------------------------------------------------------------------------- #
# resolve_missing_pet — atomicity / failure paths
# --------------------------------------------------------------------------- #
def test_rollback_when_final_sighting_wrong_action_type(conn, seed):
    # Spec example: a non-Caught final sighting must abort with NO residue.
    owner = seed.user()
    admin = seed.user(role="admin")
    pet = seed.missing_pet(owner_id=owner, bounty=1000)
    final_hunter = seed.user()
    bad_final = seed.sighting(hunter_id=final_hunter, initial_target_pet_id=pet,
                              action="Spotted", verification="Verified")  # not Caught
    clue = seed.user()
    seed.sighting(hunter_id=clue, initial_target_pet_id=pet, verification="Verified")

    with pytest.raises(psycopg.errors.RaiseException):
        with conn.transaction():
            _resolve(conn, pet, bad_final, admin)

    assert pet_status(conn, pet) == "Searching"
    assert row_count(conn, "bounty_transactions", "missing_pet_id", pet) == 0
    assert row_count(conn, "score_awards", "missing_pet_id", pet) == 0
    assert total_score(conn, clue) == 0


def test_rollback_during_write_phase_leaves_no_residue(conn, seed):
    # Failure AFTER the final-hunter check passes: an invalid verified_by makes
    # the bounty INSERT (the first write) hit an FK violation. The whole
    # function must roll back — proving the write phase is atomic.
    owner = seed.user()
    pet = seed.missing_pet(owner_id=owner, bounty=1000)
    final_hunter = seed.user()
    final_sighting = seed.sighting(hunter_id=final_hunter, initial_target_pet_id=pet,
                                   action="Caught", verification="Verified")
    clue = seed.user()
    seed.sighting(hunter_id=clue, initial_target_pet_id=pet, verification="Verified")

    import uuid
    bogus_admin = uuid.uuid4()  # not a real users.id -> verified_by FK violation

    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with conn.transaction():
            _resolve(conn, pet, final_sighting, bogus_admin)

    assert pet_status(conn, pet) == "Searching"
    assert row_count(conn, "bounty_transactions", "missing_pet_id", pet) == 0
    assert row_count(conn, "score_awards", "missing_pet_id", pet) == 0
    assert total_score(conn, clue) == 0


def test_resolving_an_already_resolved_pet_raises_and_does_not_double_pay(conn, seed):
    owner = seed.user()
    admin = seed.user(role="admin")
    pet = seed.missing_pet(owner_id=owner, bounty=1000)
    final_hunter = seed.user()
    final_sighting = seed.sighting(hunter_id=final_hunter, initial_target_pet_id=pet,
                                   action="Caught", verification="Verified")

    _resolve(conn, pet, final_sighting, admin)            # first resolve succeeds
    assert pet_status(conn, pet) == "Resolved"

    with pytest.raises(psycopg.errors.RaiseException):
        with conn.transaction():
            _resolve(conn, pet, final_sighting, admin)    # second must reject

    assert row_count(conn, "bounty_transactions", "missing_pet_id", pet) == 1  # paid once


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
