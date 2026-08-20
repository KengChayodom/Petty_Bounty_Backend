"""
Integration tests for resolve_missing_pet + the sighting_matches UNIQUE
constraint — the transactional, schema-enforced behaviours that only a real
Postgres can verify.

resolve_missing_pet settles the MONEY on a case its owner has already closed:
one bounty transaction against the sighting the owner confirmed as the catch,
and the pet from 'Found' to 'Resolved' — all or nothing.

It used to distribute the F1 clue scores as well. Since 2026-08-21 it does not:
the owner distributes them when they confirm the rescue, days before the
transfer, so awarding here too would pay every hunter twice. The scoring tests
that used to live here therefore moved to test_owner_decide_sighting.py, next to
the function that now does the work. What is left here is the failure and
atomicity behaviour of the payment itself.
"""
import psycopg
import pytest
from pytest import approx

from _query import pet_status, row_count, total_score

pytestmark = pytest.mark.integration


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


def _rescued_case(seed, *, bounty=5000, action="Caught",
                  owner_status="Confirmed"):
    """A case its owner has already closed — the only shape the payment accepts.

    'Found' means the animal is home and the scores are already distributed;
    the payment is the administrator catching the money up with a decision that
    was made days earlier. Seeded directly rather than by driving
    owner_decide_sighting, so a failure here points at the payment function and
    not at the one before it.
    """
    owner = seed.user()
    hunter = seed.user()
    pet = seed.missing_pet(owner_id=owner, bounty=bounty, status="Found")
    sighting = seed.sighting(hunter_id=hunter, initial_target_pet_id=pet,
                             action=action)
    seed.sighting_match(sighting_id=sighting, missing_pet_id=pet,
                        similarity=None, owner_status=owner_status)
    return owner, hunter, pet, sighting


# --------------------------------------------------------------------------- #
# resolve_missing_pet — happy path (money only)
# --------------------------------------------------------------------------- #
def test_resolve_pays_the_bounty_marks_resolved_and_awards_nothing(conn, seed):
    admin = seed.user(role="admin")
    owner, hunter, pet, final_sighting = _rescued_case(seed, bounty=5000)

    result = _resolve(conn, pet, final_sighting, admin)

    assert pet_status(conn, pet) == "Resolved"

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

    # The scoring is owner_decide_sighting's, and it already ran. Not one point
    # moves here — the catcher's balance included.
    assert _awards(conn, pet) == []
    assert result["awards"] == []
    assert total_score(conn, hunter) == 0

    assert result["final_hunter_id"] == str(hunter)
    assert float(result["bounty_amount"]) == approx(5000)


def test_a_search_that_is_still_open_cannot_be_paid(conn, seed):
    """The bounty follows the owner's resolution. A pet still 'Searching' has
    not been recovered, whatever an administrator believes."""
    admin = seed.user(role="admin")
    owner, hunter = seed.user(), seed.user()
    pet = seed.missing_pet(owner_id=owner, status="Searching")
    sighting = seed.sighting(hunter_id=hunter, initial_target_pet_id=pet,
                             action="Caught")
    seed.sighting_match(sighting_id=sighting, missing_pet_id=pet,
                        similarity=None, owner_status="Confirmed")

    with pytest.raises(psycopg.errors.RaiseException,
                       match="has not been recovered yet"):
        with conn.transaction():
            _resolve(conn, pet, sighting, admin)

    assert row_count(conn, "bounty_transactions", "missing_pet_id", pet) == 0


def test_a_sighting_the_owner_never_confirmed_cannot_be_paid(conn, seed):
    """Eligibility is the OWNER's confirmation. Nothing writes 'Verified' any
    more, so a check against that column would pay nobody, ever."""
    admin = seed.user(role="admin")
    _, _, pet, sighting = _rescued_case(seed, owner_status="Pending")

    with pytest.raises(psycopg.errors.RaiseException,
                       match="not a confirmed Caught sighting"):
        with conn.transaction():
            _resolve(conn, pet, sighting, admin)

    assert row_count(conn, "bounty_transactions", "missing_pet_id", pet) == 0


# --------------------------------------------------------------------------- #
# resolve_missing_pet — atomicity / failure paths
# --------------------------------------------------------------------------- #
def test_rollback_when_final_sighting_wrong_action_type(conn, seed):
    # Spec example: a non-Caught final sighting must abort with NO residue. The
    # owner confirmed it, but confirming a Spotted card says "yes that is my
    # cat", not "this person brought it home".
    admin = seed.user(role="admin")
    _, hunter, pet, bad_final = _rescued_case(seed, bounty=1000,
                                              action="Spotted")

    with pytest.raises(psycopg.errors.RaiseException):
        with conn.transaction():
            _resolve(conn, pet, bad_final, admin)

    assert pet_status(conn, pet) == "Found"
    assert row_count(conn, "bounty_transactions", "missing_pet_id", pet) == 0
    assert total_score(conn, hunter) == 0


def test_rollback_during_write_phase_leaves_no_residue(conn, seed):
    # Failure AFTER the final-hunter check passes: an invalid verified_by makes
    # the bounty INSERT (the first write) hit an FK violation. The whole
    # function must roll back — proving the write phase is atomic, and in
    # particular that the pet is not left 'Resolved' with no transaction behind
    # it, which would read as a paid case forever.
    import uuid
    _, _, pet, final_sighting = _rescued_case(seed, bounty=1000)
    bogus_admin = uuid.uuid4()  # not a real users.id -> verified_by FK violation

    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with conn.transaction():
            _resolve(conn, pet, final_sighting, bogus_admin)

    assert pet_status(conn, pet) == "Found"
    assert row_count(conn, "bounty_transactions", "missing_pet_id", pet) == 0


def test_resolving_an_already_resolved_pet_raises_and_does_not_double_pay(conn, seed):
    admin = seed.user(role="admin")
    _, _, pet, final_sighting = _rescued_case(seed, bounty=1000)

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
