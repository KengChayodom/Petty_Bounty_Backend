"""
Unit tests for app/services/pet_logic.py — the pure rule behind the badge an
owner sees on their report card (decided 2026-08-17).

No I/O at all (TEST_PLAN §3 layer L1). The defects these catch are the ones a
status model gets wrong in practice:

  * a recovered pet still shouting SPOTTED because sightings were counted after
    the search was closed;
  * a fresh report crashing or vanishing from the list because nobody has
    reported a sighting for it yet;
  * the badge being computed in two places (server and client) and drifting.
"""
from datetime import datetime, timedelta, timezone

import pytest

from app.services.pet_logic import (
    POST_STATUS_EXPIRED,
    POST_STATUS_PENDING,
    POST_STATUS_RESCUED,
    POST_STATUS_SPOTTED,
    attach_sighting_counts,
    derive_post_status,
    is_post_expired,
)

# A fixed "now" so the boundary is a value in the test, not the clock. The
# timestamps below are `expires_at` values (an absolute instant), not ages.
NOW = datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)
FRESH = NOW + timedelta(days=1)                       # expires tomorrow — live
JUST_INSIDE = NOW + timedelta(seconds=1)              # expires in 1s — still live
ON_THE_BOUNDARY = NOW                                 # expires_at == now — out
LONG_EXPIRED = NOW - timedelta(days=30)               # expired a month ago


class TestDerivePostStatus:
    @pytest.mark.parametrize("pet_status,count,expected", [
        # search still open — the sighting count decides
        ("Searching", 0, POST_STATUS_PENDING),
        ("Searching", 1, POST_STATUS_SPOTTED),
        ("Searching", 3, POST_STATUS_SPOTTED),
        # closed — wins regardless of how many sightings came in
        ("Found", 0, POST_STATUS_RESCUED),
        ("Found", 3, POST_STATUS_RESCUED),
        ("Resolved", 0, POST_STATUS_RESCUED),
        ("Resolved", 7, POST_STATUS_RESCUED),
    ])
    def test_the_whole_table(self, pet_status, count, expected):
        assert derive_post_status(pet_status, count) == expected

    def test_closed_check_is_case_and_space_insensitive(self):
        """The column is an enum, but rows written by older code paths and by
        hand in the SQL editor have arrived with odd casing before."""
        assert derive_post_status("found", 0) == POST_STATUS_RESCUED
        assert derive_post_status("  RESOLVED ", 0) == POST_STATUS_RESCUED

    def test_legacy_spotted_column_value_is_ignored(self):
        """`pet_status` still has a 'Spotted' member from the old model. It is
        no longer written, and it must NOT be read as "the search is over" — a
        row carrying it is still an open search."""
        assert derive_post_status("Spotted", 0) == POST_STATUS_PENDING
        assert derive_post_status("Spotted", 2) == POST_STATUS_SPOTTED

    def test_missing_status_is_treated_as_an_open_search(self):
        """A null status is not a recovery. Reading it as one would tell an
        owner their pet is home."""
        assert derive_post_status(None, 0) == POST_STATUS_PENDING
        assert derive_post_status("", 1) == POST_STATUS_SPOTTED


class TestIsPostExpired:
    """SRS-87. The predicate must agree with the SQL exactly, because the SQL
    is what actually stops a post reaching hunters — disagreeing means the
    badge says one thing and the map does another."""

    @pytest.mark.parametrize("expires_at,expected", [
        (FRESH, False),
        (JUST_INSIDE, False),
        # The read paths keep a post while expires_at > NOW(), so the boundary
        # instant itself is already out.
        (ON_THE_BOUNDARY, True),
        (LONG_EXPIRED, True),
    ])
    def test_the_boundary(self, expires_at, expected):
        assert is_post_expired(expires_at, now=NOW) is expected

    def test_reads_postgrest_iso_strings(self):
        """PostgREST hands timestamps back as strings, including the 'Z' form
        that fromisoformat rejected before Python 3.11."""
        assert is_post_expired(LONG_EXPIRED.isoformat(), now=NOW) is True
        assert is_post_expired("2026-07-01T00:00:00Z", now=NOW) is True
        assert is_post_expired("2026-09-01T00:00:00Z", now=NOW) is False

    def test_naive_timestamps_are_read_as_utc(self):
        """The column is `timestamp with time zone` written by NOW(), but a
        driver may hand back a naive value. Reading it as local time would move
        the boundary by the machine's offset."""
        assert is_post_expired(
            LONG_EXPIRED.replace(tzinfo=None), now=NOW
        ) is True

    def test_a_naive_now_is_also_read_as_utc(self):
        """Callers pass an aware `now`, but a naive one must still answer
        rather than raising on a naive/aware comparison."""
        assert is_post_expired(
            LONG_EXPIRED, now=NOW.replace(tzinfo=None)
        ) is True
        assert is_post_expired(FRESH, now=NOW.replace(tzinfo=None)) is False

    @pytest.mark.parametrize("bad", [None, "", "   ", "not-a-date", 12345])
    def test_an_unreadable_timestamp_is_not_an_expiry(self, bad):
        """Greying out a live search because one timestamp arrived in an odd
        shape hides a findable pet — the worse of the two failures."""
        assert is_post_expired(bad, now=NOW) is False


class TestDerivePostStatusExpiry:
    def test_an_aged_out_post_with_nothing_to_show_reads_expired(self):
        assert derive_post_status(
            "Searching", 0, LONG_EXPIRED, now=NOW
        ) == POST_STATUS_EXPIRED

    def test_spotted_outranks_expired(self):
        """An expired post that collected sightings still has a queue its owner
        must work through — the row is untouched by expiry and the case can
        still be closed and paid. Badging it EXPIRED would grey out the one
        report that needs action."""
        assert derive_post_status(
            "Searching", 2, LONG_EXPIRED, now=NOW
        ) == POST_STATUS_SPOTTED

    def test_rescued_outranks_expired(self):
        """A pet recovered after its post aged out is still home."""
        assert derive_post_status(
            "Found", 0, LONG_EXPIRED, now=NOW
        ) == POST_STATUS_RESCUED

    def test_a_live_post_is_unaffected(self):
        assert derive_post_status(
            "Searching", 0, FRESH, now=NOW
        ) == POST_STATUS_PENDING

    def test_callers_without_a_timestamp_keep_the_old_behaviour(self):
        """`created_at` is optional so a caller that does not have it gets the
        pre-expiry answer rather than a wrong one."""
        assert derive_post_status("Searching", 0) == POST_STATUS_PENDING


class TestAttachSightingCounts:
    def test_attaches_count_and_status_per_pet(self):
        pets = [
            {"id": "p1", "status": "Searching"},
            {"id": "p2", "status": "Searching"},
            {"id": "p3", "status": "Found"},
        ]
        out = attach_sighting_counts(pets, {"p2": 3, "p3": 1})

        assert [p["post_status"] for p in out] == [
            POST_STATUS_PENDING, POST_STATUS_SPOTTED, POST_STATUS_RESCUED,
        ]
        assert [p["sighting_count"] for p in out] == [0, 3, 1]

    def test_pet_missing_from_the_counts_reads_zero(self):
        """Pets with no sightings are simply absent from the count query's
        result — that must be 0, not a KeyError and not a dropped row."""
        out = attach_sighting_counts([{"id": "p1", "status": "Searching"}], {})
        assert out[0]["sighting_count"] == 0
        assert out[0]["post_status"] == POST_STATUS_PENDING

    def test_original_rows_are_not_mutated(self):
        """The rows come straight from the repository; mutating them in place
        would make the enrichment order-dependent if it ever ran twice."""
        pets = [{"id": "p1", "status": "Searching"}]
        attach_sighting_counts(pets, {"p1": 2})
        assert pets == [{"id": "p1", "status": "Searching"}]

    def test_other_fields_survive(self):
        out = attach_sighting_counts(
            [{"id": "p1", "status": "Searching", "pet_name": "Mochi",
              "bounty_amount": 2000}],
            {"p1": 1},
        )
        assert out[0]["pet_name"] == "Mochi"
        assert out[0]["bounty_amount"] == 2000

    def test_empty_list(self):
        assert attach_sighting_counts([], {}) == []

    def test_expiry_is_attached_per_row(self):
        pets = [
            {"id": "p1", "status": "Searching", "expires_at": FRESH},
            {"id": "p2", "status": "Searching", "expires_at": LONG_EXPIRED},
            {"id": "p3", "status": "Searching", "expires_at": LONG_EXPIRED},
        ]
        out = attach_sighting_counts(pets, {"p3": 2}, now=NOW)
        assert [p["post_status"] for p in out] == [
            POST_STATUS_PENDING, POST_STATUS_EXPIRED, POST_STATUS_SPOTTED,
        ]

    def test_now_is_sampled_once_for_the_whole_list(self):
        """Two reports expiring in the same second must not land on opposite
        sides of the boundary because the clock ticked mid-loop."""
        pets = [
            {"id": f"p{i}", "status": "Searching", "expires_at": ON_THE_BOUNDARY}
            for i in range(3)
        ]
        out = attach_sighting_counts(pets, {}, now=NOW)
        assert {p["post_status"] for p in out} == {POST_STATUS_EXPIRED}


class TestCountSightings:
    """The rule for what counts as a sighting of a pet.

    It lives in pure logic rather than in the query on purpose: it is a product
    decision, and only here can a test pin it down — the adapter runs solely
    against a real database.
    """

    def test_counts_per_pet(self):
        from app.services.pet_logic import count_sightings

        links = [
            {"pet_id": "p1", "sighting_id": "s1", "owner_status": None},
            {"pet_id": "p1", "sighting_id": "s2", "owner_status": "Pending"},
            {"pet_id": "p2", "sighting_id": "s3", "owner_status": "Confirmed"},
        ]
        assert count_sightings(links) == {"p1": 2, "p2": 1}

    def test_one_sighting_counts_once_across_both_sources(self):
        """A hunter can report a pet from its detail page AND have the photo
        match it — one person, one sighting, two rows."""
        from app.services.pet_logic import count_sightings

        links = [
            {"pet_id": "p1", "sighting_id": "s1", "owner_status": None},  # matched
            {"pet_id": "p1", "sighting_id": "s1", "owner_status": None},  # targeted
        ]
        assert count_sightings(links) == {"p1": 1}

    def test_rejected_matches_do_not_count(self):
        """The owner said it is not their pet. Counting it anyway would keep
        the report reading "someone has seen your pet" on a match they already
        dismissed."""
        from app.services.pet_logic import count_sightings

        links = [
            {"pet_id": "p1", "sighting_id": "s1", "owner_status": "Rejected"},
            {"pet_id": "p1", "sighting_id": "s2", "owner_status": "Confirmed"},
        ]
        assert count_sightings(links) == {"p1": 1}

    def test_a_pet_whose_every_match_was_rejected_disappears(self):
        """…which is what takes the card back to PENDING."""
        from app.services.pet_logic import count_sightings

        links = [
            {"pet_id": "p1", "sighting_id": "s1", "owner_status": "Rejected"},
        ]
        assert count_sightings(links) == {}

    def test_confirmed_still_counts(self):
        from app.services.pet_logic import count_sightings

        links = [{"pet_id": "p1", "sighting_id": "s1", "owner_status": "Confirmed"}]
        assert count_sightings(links) == {"p1": 1}

    @pytest.mark.parametrize("bad", [
        {"pet_id": None, "sighting_id": "s1", "owner_status": None},
        {"pet_id": "p1", "sighting_id": None, "owner_status": None},
        {},
    ])
    def test_incomplete_rows_are_skipped_not_fatal(self, bad):
        """`sighting_matches` allows NULL on both foreign keys, so a half-empty
        row is representable — it must not take the whole list down."""
        from app.services.pet_logic import count_sightings

        assert count_sightings([bad]) == {}

    def test_empty(self):
        from app.services.pet_logic import count_sightings

        assert count_sightings([]) == {}
