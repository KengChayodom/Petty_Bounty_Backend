"""
Integration tests for owner_decide_sighting — the function the whole product
now turns on (2026-08-21).

Confirming one card can end a search and move every point a case will ever pay,
and none of that is reconstructable afterwards, so every rule lives in SQL and
is tested here against the real function rather than against a double:

  * the queue is ORDERED — oldest card first, and a card an administrator
    dismissed is not in the queue at all (one piece of spam must not jam an
    owner's queue forever);
  * only CONFIRMED cards earn — Pending and Rejected earn nothing, and neither
    does a card that was confirmed and later dismissed by moderation;
  * ranking is NEWEST first, one seat per hunter, 25/15/10/5/5, capped at five;
  * the payout happens EXACTLY ONCE, guarded by the search being closed.

The bounty half is tested alongside it: `resolve_missing_pet` must no longer
award anything, or every hunter would be paid twice — once at rescue, once when
the administrator settles the money days later.
"""
import uuid

import psycopg
import pytest

from _query import pet_status, row_count, total_score

pytestmark = pytest.mark.integration


def _decide(conn, pet_id, sighting_id, owner_id, decision="Confirmed"):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT owner_decide_sighting(%s, %s, %s, %s)",
            (pet_id, sighting_id, owner_id, decision),
        )
        return cur.fetchone()[0]


def _owner_status(conn, sighting_id, pet_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT owner_status FROM sighting_matches "
            "WHERE sighting_id = %s AND missing_pet_id = %s",
            (sighting_id, pet_id),
        )
        row = cur.fetchone()
        return row[0] if row else None


def _sighting_status(conn, sighting_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT sighting_status FROM sightings WHERE id = %s", (sighting_id,)
        )
        return cur.fetchone()[0]


def _awards(conn, pet_id):
    """(user_id, points, rank) for a pet, best rank first."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT user_id, points, rank FROM score_awards "
            "WHERE missing_pet_id = %s ORDER BY rank",
            (pet_id,),
        )
        return cur.fetchall()


def _card(seed, pet, *, hunter, created_at, action="Spotted",
          verification="Pending", owner_status="Pending"):
    """One queue card: a sighting plus its sighting_matches row."""
    sid = seed.sighting(
        hunter_id=hunter, action=action, verification=verification,
        created_at=created_at,
    )
    seed.sighting_match(
        sighting_id=sid, missing_pet_id=pet, owner_status=owner_status,
    )
    return sid


# --------------------------------------------------------------------------- #
# Ownership and the shape of "not on your queue"
# --------------------------------------------------------------------------- #
class TestAccess:
    def test_a_stranger_cannot_decide_and_learns_nothing(self, conn, seed):
        """404-not-403 expressed in SQL: the message must not distinguish "not
        yours" from "no such pet", or it becomes a way to enumerate reports."""
        owner, stranger, hunter = seed.user(), seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)
        card = _card(seed, pet, hunter=hunter, created_at="2026-01-01")

        with pytest.raises(
            psycopg.errors.RaiseException, match="not found or not owned by you",
        ):
            with conn.transaction():
                _decide(conn, pet, card, stranger)

        assert _owner_status(conn, card, pet) == "Pending"

    def test_unknown_pet_reads_the_same_as_someone_elses(self, conn, seed):
        owner, hunter = seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)
        card = _card(seed, pet, hunter=hunter, created_at="2026-01-01")

        with pytest.raises(
            psycopg.errors.RaiseException, match="not found or not owned by you",
        ):
            with conn.transaction():
                _decide(conn, uuid.uuid4(), card, owner)

    def test_a_sighting_with_no_queue_row_is_not_on_the_queue(self, conn, seed):
        """A sighting that was never matched to this pet — deciding it would be
        ruling on somebody else's timeline."""
        owner, hunter = seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)
        loose = seed.sighting(hunter_id=hunter)

        with pytest.raises(
            psycopg.errors.RaiseException, match="is not a match for pet",
        ):
            with conn.transaction():
                _decide(conn, pet, loose, owner)

    def test_a_dismissed_card_cannot_be_decided(self, conn, seed):
        """Moderation withdrew it. It is not on the queue, so it reads exactly
        like a sighting that was never there."""
        owner, hunter = seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)
        card = _card(seed, pet, hunter=hunter, created_at="2026-01-01",
                     verification="Dismissed")

        with pytest.raises(
            psycopg.errors.RaiseException, match="is not a match for pet",
        ):
            with conn.transaction():
                _decide(conn, pet, card, owner)


# --------------------------------------------------------------------------- #
# The queue is ordered
# --------------------------------------------------------------------------- #
class TestOrdering:
    def test_skipping_ahead_is_refused(self, conn, seed):
        """The whole point of the order: an owner who could jump straight to the
        rescue card would close the case with everyone who helped still Pending,
        and Pending earns nothing."""
        owner, h1, h2 = seed.user(), seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)
        _card(seed, pet, hunter=h1, created_at="2026-01-01")
        newer = _card(seed, pet, hunter=h2, created_at="2026-01-02")

        with pytest.raises(
            psycopg.errors.RaiseException, match="is out of order",
        ):
            with conn.transaction():
                _decide(conn, pet, newer, owner)

    def test_deciding_the_oldest_unlocks_the_next(self, conn, seed):
        owner, h1, h2 = seed.user(), seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)
        older = _card(seed, pet, hunter=h1, created_at="2026-01-01")
        newer = _card(seed, pet, hunter=h2, created_at="2026-01-02")

        _decide(conn, pet, older, owner, "Rejected")
        _decide(conn, pet, newer, owner, "Confirmed")

        assert _owner_status(conn, older, pet) == "Rejected"
        assert _owner_status(conn, newer, pet) == "Confirmed"

    def test_a_dismissed_card_does_not_block_the_queue(self, conn, seed):
        """Otherwise one piece of spam an administrator withdrew would freeze
        the owner's queue permanently — nothing after it could ever be decided."""
        owner, h1, h2 = seed.user(), seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)
        _card(seed, pet, hunter=h1, created_at="2026-01-01",
              verification="Dismissed")
        later = _card(seed, pet, hunter=h2, created_at="2026-01-02")

        _decide(conn, pet, later, owner, "Confirmed")

        assert _owner_status(conn, later, pet) == "Confirmed"

    def test_a_card_cannot_be_decided_twice(self, conn, seed):
        """A verdict is what scoring reads. Flipping one after the fact would
        silently change who gets paid."""
        owner, hunter = seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)
        card = _card(seed, pet, hunter=hunter, created_at="2026-01-01")

        _decide(conn, pet, card, owner, "Confirmed")
        with pytest.raises(
            psycopg.errors.RaiseException, match="has already been decided",
        ):
            with conn.transaction():
                _decide(conn, pet, card, owner, "Rejected")

    def test_ties_on_created_at_still_have_one_order(self, conn, seed):
        """Two sightings can share a timestamp. The order falls back to the id,
        so exactly one of them is next and the queue cannot deadlock."""
        owner, h1, h2 = seed.user(), seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)
        a = _card(seed, pet, hunter=h1, created_at="2026-01-01T00:00:00Z")
        b = _card(seed, pet, hunter=h2, created_at="2026-01-01T00:00:00Z")

        first, second = sorted([a, b], key=str)
        with pytest.raises(
            psycopg.errors.RaiseException, match="is out of order",
        ):
            with conn.transaction():
                _decide(conn, pet, second, owner)

        _decide(conn, pet, first, owner, "Confirmed")
        _decide(conn, pet, second, owner, "Confirmed")


# --------------------------------------------------------------------------- #
# A confirmed Spotted card decides nothing else
# --------------------------------------------------------------------------- #
class TestSpottedConfirmation:
    def test_confirming_a_spotted_card_does_not_close_the_search(
        self, conn, seed,
    ):
        """The animal is not home yet. The post must stay Searching — the match
        RPC and the map both filter on that status, so closing here would pull
        the report out of circulation while it is still lost."""
        owner, hunter = seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)
        card = _card(seed, pet, hunter=hunter, created_at="2026-01-01")

        out = _decide(conn, pet, card, owner, "Confirmed")

        assert out["search_closed"] is False
        assert out["awards"] == []
        assert pet_status(conn, pet) == "Searching"
        assert _sighting_status(conn, card) == "Confirmed"
        assert row_count(conn, "score_awards", "missing_pet_id", pet) == 0

    def test_rejecting_pays_nothing_and_touches_nothing_else(self, conn, seed):
        owner, hunter = seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)
        card = _card(seed, pet, hunter=hunter, created_at="2026-01-01")

        out = _decide(conn, pet, card, owner, "Rejected")

        assert out["search_closed"] is False
        assert _sighting_status(conn, card) == "Pending_Analysis"
        assert total_score(conn, hunter) == 0


# --------------------------------------------------------------------------- #
# The rescue: one confirmation ends the search and pays everybody
# --------------------------------------------------------------------------- #
class TestRescuePayout:
    def test_the_full_case_end_to_end(self, conn, seed):
        """Four cards, decided oldest-first, closing on the catch. Ranking runs
        NEWEST first, so the hunter who caught the animal ranks 1st — the old
        admin-side function excluded the catcher from scoring entirely and
        ranked the rest oldest-first, i.e. exactly backwards."""
        owner = seed.user()
        h_a, h_b, h_c, h_d = (seed.user() for _ in range(4))
        pet = seed.missing_pet(owner_id=owner)

        a = _card(seed, pet, hunter=h_a, created_at="2026-01-01")
        b = _card(seed, pet, hunter=h_b, created_at="2026-01-02")
        c = _card(seed, pet, hunter=h_c, created_at="2026-01-03")
        d = _card(seed, pet, hunter=h_d, created_at="2026-01-04", action="Caught")

        _decide(conn, pet, a, owner, "Confirmed")
        _decide(conn, pet, b, owner, "Rejected")     # not my cat
        _decide(conn, pet, c, owner, "Confirmed")
        out = _decide(conn, pet, d, owner, "Confirmed")

        assert out["search_closed"] is True
        assert out["pet_status"] == "Found"
        assert pet_status(conn, pet) == "Found"

        assert _awards(conn, pet) == [
            (h_d, 25, 1),   # the catch, newest
            (h_c, 15, 2),
            (h_a, 10, 3),
        ]
        assert total_score(conn, h_d) == 25
        assert total_score(conn, h_c) == 15
        assert total_score(conn, h_a) == 10
        assert total_score(conn, h_b) == 0           # rejected earns nothing

    def test_pending_cards_earn_nothing(self, conn, seed):
        """A card the owner never ruled on is not evidence of anything. It can
        only happen for cards NEWER than the catch, which the ordering rule
        allows to be left behind."""
        owner, h_late, h_catch = seed.user(), seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)

        catch = _card(seed, pet, hunter=h_catch, created_at="2026-01-03",
                      action="Caught")
        # arrived after the catch, from someone who had not heard
        _card(seed, pet, hunter=h_late, created_at="2026-01-04")

        _decide(conn, pet, catch, owner, "Confirmed")

        assert [a[0] for a in _awards(conn, pet)] == [h_catch]
        assert total_score(conn, h_late) == 0

    def test_a_dismissed_card_earns_nothing_even_if_it_was_confirmed(
        self, conn, seed,
    ):
        """Order of events: the owner confirms, someone flags it, the admin
        upholds. The confirmation is still on the row, so scoring has to check
        the moderation ruling too."""
        owner, h_bad, h_catch = seed.user(), seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)

        _card(seed, pet, hunter=h_bad, created_at="2026-01-01",
              verification="Dismissed", owner_status="Confirmed")
        catch = _card(seed, pet, hunter=h_catch, created_at="2026-01-02",
                      action="Caught")

        _decide(conn, pet, catch, owner, "Confirmed")

        assert [a[0] for a in _awards(conn, pet)] == [h_catch]
        assert total_score(conn, h_bad) == 0

    def test_one_hunter_takes_one_seat_at_their_newest_card(self, conn, seed):
        """Three photos from the same person is one person helping. Without this
        they would take three of the five seats."""
        owner, prolific, h_catch = seed.user(), seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)

        _card(seed, pet, hunter=prolific, created_at="2026-01-01")
        _card(seed, pet, hunter=prolific, created_at="2026-01-02")
        newest = _card(seed, pet, hunter=prolific, created_at="2026-01-03")
        catch = _card(seed, pet, hunter=h_catch, created_at="2026-01-04",
                      action="Caught")

        for card in _pending_in_order(conn, pet):
            _decide(conn, pet, card, owner, "Confirmed")

        awards = _awards(conn, pet)
        assert [(a[0], a[1]) for a in awards] == [(h_catch, 25), (prolific, 15)]
        # the seat is taken at their NEWEST card, not their first
        assert [a for a in awards if a[0] == prolific][0][0] == prolific
        with conn.cursor() as cur:
            cur.execute(
                "SELECT sighting_id FROM score_awards "
                "WHERE missing_pet_id = %s AND user_id = %s", (pet, prolific),
            )
            assert cur.fetchone()[0] == newest
        assert catch is not None

    def test_the_ladder_caps_at_five_hunters(self, conn, seed):
        """25/15/10/5/5 and nothing after. The previous function had no cap at
        all: rank 6, 7, 20 each earned 5 points forever."""
        owner = seed.user()
        hunters = [seed.user() for _ in range(7)]
        pet = seed.missing_pet(owner_id=owner)

        for day, hunter in enumerate(hunters, start=1):
            _card(seed, pet, hunter=hunter, created_at=f"2026-01-{day:02d}",
                  action="Caught" if hunter is hunters[-1] else "Spotted")

        for card in _pending_in_order(conn, pet):
            _decide(conn, pet, card, owner, "Confirmed")

        awards = _awards(conn, pet)
        assert [a[1] for a in awards] == [25, 15, 10, 5, 5]
        # the two oldest contributors are outside the ladder
        assert total_score(conn, hunters[0]) == 0
        assert total_score(conn, hunters[1]) == 0

    def test_the_remaining_sightings_are_closed(self, conn, seed):
        """The animal is home; nothing on this timeline is a live lead any more.
        Same rule the owner's End Search button applies."""
        owner, h1, h2 = seed.user(), seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)
        spotted = _card(seed, pet, hunter=h1, created_at="2026-01-01")
        catch = _card(seed, pet, hunter=h2, created_at="2026-01-02",
                      action="Caught")

        _decide(conn, pet, spotted, owner, "Confirmed")
        _decide(conn, pet, catch, owner, "Confirmed")

        assert _sighting_status(conn, spotted) == "Closed"
        assert _sighting_status(conn, catch) == "Confirmed"

    def test_the_payout_cannot_run_twice(self, conn, seed):
        """The closed search is the guard. A second rescue card — a rival claim
        that arrived later — must not re-open the till."""
        owner, h1, h2 = seed.user(), seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)
        first = _card(seed, pet, hunter=h1, created_at="2026-01-01",
                      action="Caught")
        second = _card(seed, pet, hunter=h2, created_at="2026-01-02",
                       action="Caught")

        _decide(conn, pet, first, owner, "Confirmed")
        with pytest.raises(
            psycopg.errors.RaiseException, match="is already closed",
        ):
            with conn.transaction():
                _decide(conn, pet, second, owner, "Confirmed")

        assert total_score(conn, h1) == 25
        assert total_score(conn, h2) == 0

    def test_rival_claims_are_settled_by_rejecting_the_first(self, conn, seed):
        """No tie-break logic anywhere: the older claim is offered first, and
        rejecting it advances the queue to the newer one."""
        owner, liar, hero = seed.user(), seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)
        false_claim = _card(seed, pet, hunter=liar, created_at="2026-01-01",
                            action="Caught")
        real_claim = _card(seed, pet, hunter=hero, created_at="2026-01-02",
                           action="Caught")

        _decide(conn, pet, false_claim, owner, "Rejected")
        out = _decide(conn, pet, real_claim, owner, "Confirmed")

        assert out["search_closed"] is True
        assert total_score(conn, hero) == 25
        assert total_score(conn, liar) == 0


def _pending_in_order(conn, pet_id):
    """Every undecided card for a pet, oldest first — the order the owner must
    follow. Read once up front: deciding rewrites the rows this selects."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT s.id FROM sighting_matches sm "
            "  JOIN sightings s ON s.id = sm.sighting_id "
            " WHERE sm.missing_pet_id = %s AND sm.owner_status = 'Pending' "
            "   AND s.verification_status <> 'Dismissed' "
            " ORDER BY s.created_at, s.id",
            (pet_id,),
        )
        return [r[0] for r in cur.fetchall()]


# --------------------------------------------------------------------------- #
# resolve_missing_pet — money only, and only after the owner has closed
# --------------------------------------------------------------------------- #
class TestBountySettlement:
    def _rescued_case(self, conn, seed):
        owner, helper, catcher = seed.user(), seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner, bounty=5000)
        helped = _card(seed, pet, hunter=helper, created_at="2026-01-01")
        catch = _card(seed, pet, hunter=catcher, created_at="2026-01-02",
                      action="Caught")
        _decide(conn, pet, helped, owner, "Confirmed")
        _decide(conn, pet, catch, owner, "Confirmed")
        return pet, catch, catcher, helper

    def _resolve(self, conn, pet, sighting, admin):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT resolve_missing_pet(%s, %s, %s, %s, %s)",
                (pet, sighting, "http://img/slip.jpg", "REF-1", admin),
            )
            return cur.fetchone()[0]

    def test_settlement_pays_the_bounty_and_awards_nothing(self, conn, seed):
        """The scores were distributed at rescue time, days earlier. Awarding
        here as well — which the old function did — would pay everyone twice."""
        admin = seed.user(role="admin")
        pet, catch, catcher, helper = self._rescued_case(conn, seed)
        before = {u: total_score(conn, u) for u in (catcher, helper)}

        out = self._resolve(conn, pet, catch, admin)

        assert out["awards"] == []
        assert out["bounty_amount"] == 5000
        assert pet_status(conn, pet) == "Resolved"
        assert row_count(conn, "bounty_transactions", "missing_pet_id", pet) == 1
        assert {u: total_score(conn, u) for u in (catcher, helper)} == before

    def test_a_search_the_owner_has_not_closed_cannot_be_paid(self, conn, seed):
        """The bounty follows the owner's resolution. Paying first would settle
        a case nobody has said is over."""
        admin, owner, hunter = seed.user(role="admin"), seed.user(), seed.user()
        pet = seed.missing_pet(owner_id=owner)
        catch = _card(seed, pet, hunter=hunter, created_at="2026-01-01",
                      action="Caught")

        with pytest.raises(
            psycopg.errors.RaiseException, match="has not been recovered yet",
        ):
            with conn.transaction():
                self._resolve(conn, pet, catch, admin)

    def test_paying_a_sighting_the_owner_did_not_confirm_is_refused(
        self, conn, seed,
    ):
        """Eligibility is the owner's confirmation now, not an administrator's
        own verification — nothing writes 'Verified' any more."""
        admin = seed.user(role="admin")
        pet, catch, _, helper = self._rescued_case(conn, seed)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT s.id FROM sightings s WHERE s.hunter_id = %s", (helper,)
            )
            helper_sighting = cur.fetchone()[0]

        with pytest.raises(
            psycopg.errors.RaiseException, match="not a confirmed Caught sighting",
        ):
            with conn.transaction():
                self._resolve(conn, pet, helper_sighting, admin)
        assert catch is not None

    def test_settling_twice_is_refused(self, conn, seed):
        admin = seed.user(role="admin")
        pet, catch, _, _ = self._rescued_case(conn, seed)

        self._resolve(conn, pet, catch, admin)
        with pytest.raises(
            psycopg.errors.RaiseException, match="already resolved",
        ):
            with conn.transaction():
                self._resolve(conn, pet, catch, admin)
