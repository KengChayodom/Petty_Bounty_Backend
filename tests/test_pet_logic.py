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
import pytest

from app.services.pet_logic import (
    POST_STATUS_PENDING,
    POST_STATUS_RESCUED,
    POST_STATUS_SPOTTED,
    attach_sighting_counts,
    derive_post_status,
)


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
