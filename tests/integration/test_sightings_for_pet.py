"""
Integration tests for sightings_for_pet — the owner/admin read aggregation.

This is where the match_source determination lives. We exercise ALL FOUR
outcomes of that boolean logic (the three positive sources AND the exclusion
of a sighting that is neither matched nor targeted), plus the dismissed gate
and DESC pagination — against the real query.
"""
from datetime import datetime, timedelta, timezone

import pytest
from pytest import approx

pytestmark = pytest.mark.integration

_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _sightings_for_pet(conn, pet_id, *, limit=50, offset=0, include_dismissed=False):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, hunter_id, similarity_score, match_source "
            "FROM sightings_for_pet(%s, %s, %s, %s)",
            (pet_id, limit, offset, include_dismissed),
        )
        return cur.fetchall()


def test_match_source_all_four_branches(conn, seed):
    owner = seed.user()
    hunter = seed.user()
    pet = seed.missing_pet(owner_id=owner)
    other_pet = seed.missing_pet(owner_id=owner)

    # matched only: in sighting_matches for pet, not targeting it
    s_matched = seed.sighting(hunter_id=hunter)
    seed.sighting_match(sighting_id=s_matched, missing_pet_id=pet, similarity=0.8)

    # targeted only: initial_target_pet_id = pet, no AI match row
    s_targeted = seed.sighting(hunter_id=hunter, initial_target_pet_id=pet)

    # both: targeted AND AI-matched
    s_both = seed.sighting(hunter_id=hunter, initial_target_pet_id=pet)
    seed.sighting_match(sighting_id=s_both, missing_pet_id=pet, similarity=0.6)

    # neither: matched to a DIFFERENT pet, never targets `pet`
    s_neither = seed.sighting(hunter_id=hunter)
    seed.sighting_match(sighting_id=s_neither, missing_pet_id=other_pet, similarity=0.9)

    rows = {r[0]: r for r in _sightings_for_pet(conn, pet)}

    # 4th branch: the unrelated sighting is excluded entirely
    assert set(rows) == {s_matched, s_targeted, s_both}
    assert s_neither not in rows

    assert rows[s_matched][3] == "matched"
    assert rows[s_targeted][3] == "targeted"
    assert rows[s_both][3] == "both"

    # similarity_score: MAX over matches, NULL for targeted-only
    assert float(rows[s_matched][2]) == approx(0.8)
    assert rows[s_targeted][2] is None
    assert float(rows[s_both][2]) == approx(0.6)


def test_dismissed_gate(conn, seed):
    owner = seed.user()
    pet = seed.missing_pet(owner_id=owner)
    dismissed = seed.sighting(verification="Dismissed")
    seed.sighting_match(sighting_id=dismissed, missing_pet_id=pet, similarity=0.7)

    # default (owner view): Dismissed hidden
    assert _sightings_for_pet(conn, pet, include_dismissed=False) == []
    # admin view: Dismissed visible
    rows = _sightings_for_pet(conn, pet, include_dismissed=True)
    assert [r[0] for r in rows] == [dismissed]


def test_pagination_orders_by_created_at_desc(conn, seed):
    owner = seed.user()
    hunter = seed.user()
    pet = seed.missing_pet(owner_id=owner)
    oldest = seed.sighting(hunter_id=hunter, initial_target_pet_id=pet, created_at=_BASE)
    middle = seed.sighting(hunter_id=hunter, initial_target_pet_id=pet,
                           created_at=_BASE + timedelta(days=1))
    newest = seed.sighting(hunter_id=hunter, initial_target_pet_id=pet,
                           created_at=_BASE + timedelta(days=2))

    page1 = _sightings_for_pet(conn, pet, limit=2, offset=0)
    page2 = _sightings_for_pet(conn, pet, limit=2, offset=2)

    assert [r[0] for r in page1] == [newest, middle]  # DESC
    assert [r[0] for r in page2] == [oldest]


# --------------------------------------------------------------------------- #
# owner_status — added 2026-08-21, and the reason match_source needed defending
# --------------------------------------------------------------------------- #
def _queue(conn, pet_id, *, include_dismissed=False):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, owner_status, match_source, similarity_score "
            "FROM sightings_for_pet(%s, 50, 0, %s)",
            (pet_id, include_dismissed),
        )
        return {r[0]: r[1:] for r in cur.fetchall()}


def test_returns_the_owners_own_verdict(conn, seed):
    """Without this column no screen can draw the queue: which card is decided,
    which is next, which is still locked. It was the one thing the owner's own
    timeline did not return."""
    owner, hunter = seed.user(), seed.user()
    pet = seed.missing_pet(owner_id=owner)
    undecided = seed.sighting(hunter_id=hunter)
    seed.sighting_match(sighting_id=undecided, missing_pet_id=pet)
    confirmed = seed.sighting(hunter_id=hunter)
    seed.sighting_match(sighting_id=confirmed, missing_pet_id=pet,
                        owner_status="Confirmed")
    rejected = seed.sighting(hunter_id=hunter)
    seed.sighting_match(sighting_id=rejected, missing_pet_id=pet,
                        owner_status="Rejected")

    rows = _queue(conn, pet)

    assert rows[undecided][0] == "Pending"
    assert rows[confirmed][0] == "Confirmed"
    assert rows[rejected][0] == "Rejected"


def test_a_targeted_report_is_still_targeted_once_it_has_a_queue_row(conn, seed):
    """Targeted reports gained a sighting_matches row on 2026-08-21 so their
    owner could rule on them. A queue row is not an AI match, and NULL
    similarity is what separates the two — without that distinction every
    targeted report would start reporting itself as 'both'."""
    owner, hunter = seed.user(), seed.user()
    pet = seed.missing_pet(owner_id=owner)
    targeted = seed.sighting(hunter_id=hunter, initial_target_pet_id=pet)
    seed.sighting_match(sighting_id=targeted, missing_pet_id=pet,
                        similarity=None)

    rows = _queue(conn, pet)

    assert rows[targeted][1] == "targeted"
    assert rows[targeted][2] is None
    assert rows[targeted][0] == "Pending"


# --------------------------------------------------------------------------- #
# hunter contact details — added 2026-09-02
# --------------------------------------------------------------------------- #
def _hunter_details(conn, pet_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, hunter_display_name, hunter_phone, "
            "       hunter_profile_image_url "
            "FROM sightings_for_pet(%s, 50, 0, FALSE)",
            (pet_id,),
        )
        return {r[0]: r[1:] for r in cur.fetchall()}


def test_returns_the_hunters_phone_and_photo(conn, seed):
    """The owner's timeline card says "who reported this". The name alone is
    not enough to act on: the owner has to be able to ring the person holding
    their pet, and the card wants their face, not a grey icon."""
    owner = seed.user()
    hunter = seed.user(
        display_name="Somchai",
        phone="0812345678",
        profile_image_url="https://storage.test/hunters/somchai.jpg",
    )
    pet = seed.missing_pet(owner_id=owner)
    sighting = seed.sighting(hunter_id=hunter, initial_target_pet_id=pet)

    row = _hunter_details(conn, pet)[sighting]

    assert row == (
        "Somchai",
        "0812345678",
        "https://storage.test/hunters/somchai.jpg",
    )


def test_hunter_details_are_null_when_the_profile_never_set_them(conn, seed):
    """Phone and photo are optional on `users`, so the RPC has to answer NULL
    rather than fail — the card then omits the phone line and draws the avatar
    placeholder instead of inventing a number."""
    owner = seed.user()
    hunter = seed.user(display_name="Anon")
    pet = seed.missing_pet(owner_id=owner)
    sighting = seed.sighting(hunter_id=hunter, initial_target_pet_id=pet)

    row = _hunter_details(conn, pet)[sighting]

    assert row == ("Anon", None, None)
