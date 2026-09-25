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
    BROWSE_SPECIES_FILTERS,
    BROWSE_STATUS_FILTERS,
    POST_STATUS_EXPIRED,
    POST_STATUS_PENDING,
    POST_STATUS_RESCUED,
    POST_STATUS_SPOTTED,
    attach_sighting_counts,
    build_missing_pet_payload,
    derive_post_status,
    is_post_expired,
    normalize_browse_species,
    normalize_browse_status,
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


# --------------------------------------------------------------------------- #
# build_missing_pet_payload — the missing_pets INSERT contract (MD-63)
#
# Everything the report is stored as is decided here and nowhere else: the
# projection the spatial queries read, the status that makes a report
# matchable, and the shape of the timestamp. The service that calls this only
# supplies the vector, so a defect here is invisible in a service test that
# asserts on the vector alone.
# --------------------------------------------------------------------------- #
class _Pet:
    """The fields build_missing_pet_payload reads off a MissingPetCreate."""

    def __init__(self, **kw):
        self.owner_id = kw.get("owner_id", "owner-1")
        self.pet_name = kw.get("pet_name", "Luna")
        self.species = kw.get("species", "Dog")
        self.characteristics = kw.get("characteristics", {"color": "Golden"})
        self.bounty_amount = kw.get("bounty_amount", 1000.0)
        self.latitude = kw.get("latitude", 13.7563)
        self.longitude = kw.get("longitude", 100.5018)
        self.last_seen_time = kw.get(
            "last_seen_time", datetime(2025, 1, 12, 10, 30, tzinfo=timezone.utc)
        )
        self.image_url = kw.get("image_url", "https://example.com/luna.jpg")
        self.primary_color_hex = kw.get("primary_color_hex", "#AABBCC")


class TestBuildMissingPetPayload:
    def test_the_location_is_a_point_with_longitude_first(self):
        """UTC-63-TC-01 — SRS-62, the projection the spatial queries read.

        PostGIS takes x then y, which is longitude then latitude, and getting
        the pair the wrong way round puts every report in the wrong hemisphere
        without failing anything.
        """
        out = build_missing_pet_payload(
            _Pet(latitude=13.7563, longitude=100.5018), feature_vector=[0.1]
        )

        assert out["last_seen_location"] == "POINT(100.5018 13.7563)"

    def test_the_last_seen_time_is_stored_as_an_iso_timestamp(self):
        """UTC-63-TC-02 — SRS-62, the time half."""
        out = build_missing_pet_payload(
            _Pet(last_seen_time=datetime(2025, 1, 12, 10, 30, tzinfo=timezone.utc)),
            feature_vector=[0.1],
        )

        assert out["last_seen_time"] == "2025-01-12T10:30:00+00:00"

    def test_a_new_report_opens_at_searching(self):
        """UTC-63-TC-03 — SRS-64.

        'Searching' is the status the matching RPC filters on, so a report
        stored at any other value is created and then never matched.
        """
        out = build_missing_pet_payload(_Pet(), feature_vector=[0.1])

        assert out["status"] == "Searching"

    def test_the_vector_the_caller_supplies_is_the_one_stored(self):
        """UTC-63-TC-04 — SRS-63's half that this function owns."""
        out = build_missing_pet_payload(_Pet(), feature_vector=[0.25, 0.5])

        assert out["feature_vector"] == [0.25, 0.5]

    def test_every_other_field_travels_unchanged(self):
        """UTC-63-TC-05 — the payload carries the report as it was given."""
        pet = _Pet(
            owner_id="owner-9",
            pet_name="Mochi",
            species="Cat",
            characteristics={"size": "small"},
            bounty_amount=250.0,
            image_url="https://example.com/mochi.jpg",
            primary_color_hex="#112233",
        )

        out = build_missing_pet_payload(pet, feature_vector=[0.1])

        assert out["owner_id"] == "owner-9"
        assert out["pet_name"] == "Mochi"
        assert out["species"] == "Cat"
        assert out["characteristics"] == {"size": "small"}
        assert out["bounty_amount"] == 250.0
        assert out["image_url"] == "https://example.com/mochi.jpg"
        assert out["primary_color_hex"] == "#112233"

    def test_the_owners_colour_wins_over_the_measured_one(self):
        """UTC-63-TC-07 — the measured colour is only a default. An owner who
        chose a colour, including by changing the default, keeps it."""
        out = build_missing_pet_payload(
            _Pet(primary_color_hex="#8A7560"),
            feature_vector=[0.1], measured_color_hex="#726860",
        )

        assert out["primary_color_hex"] == "#8A7560"

    def test_no_owner_colour_falls_back_to_the_measured_one(self):
        """UTC-63-TC-08 — when the default never reached the form (analyze
        failed, or an older client), the colour measured at registration is
        stored, so the report still takes part in colour matching."""
        out = build_missing_pet_payload(
            _Pet(primary_color_hex=None),
            feature_vector=[0.1], measured_color_hex="#726860",
        )

        assert out["primary_color_hex"] == "#726860"

    def test_no_colour_on_either_side_stores_none(self):
        """UTC-63-TC-09 — nothing chosen and nothing measured (a full-frame
        fallback) leaves the colour empty, and the report matches on CLIP only."""
        out = build_missing_pet_payload(
            _Pet(primary_color_hex=None), feature_vector=[0.1],
        )

        assert out["primary_color_hex"] is None

    def test_the_expiry_is_left_to_the_column_default(self):
        """UTC-63-TC-06 — SRS-86 is granted by the database, not here.

        `expires_at` has a column DEFAULT of NOW() + 7 days, and that DEFAULT is
        the single source of the grant. A payload that carried the key would
        move the date silently, so its absence is the assertion.
        """
        out = build_missing_pet_payload(_Pet(), feature_vector=[0.1])

        assert "expires_at" not in out


# --------------------------------------------------------------------------- #
# normalize_browse_status / normalize_browse_species — the admin browse filter
# (MD-66). Both run before any I/O so an unrecognised value is a clean 400
# rather than a failed enumeration cast surfacing as a 500.
# --------------------------------------------------------------------------- #
class TestNormalizeBrowseStatus:
    """MD-73. The status half of the administrator browse filter."""

    def test_no_filter_means_every_report(self):
        """UTC-68-TC-01 — absence is not a value to refuse."""
        assert normalize_browse_status(None) is None

    @pytest.mark.parametrize("stored", BROWSE_STATUS_FILTERS)
    def test_each_permitted_status_passes_through(self, stored):
        """UTC-68-TC-02 — every value the column holds is accepted."""
        assert normalize_browse_status(stored) == stored

    def test_casing_and_surrounding_space_are_normalised(self):
        """UTC-68-TC-03 — the console sends what the user typed."""
        assert normalize_browse_status("  searching  ") == "Searching"

    @pytest.mark.parametrize("bad", ["Pending", "Expired", "Rescued", "Banned", ""])
    def test_a_bucket_the_column_does_not_hold_is_refused(self, bad):
        """UTC-68-TC-04 — the three derived badge names are refused too.

        Pending, Expired and Rescued are `derive_post_status` names, not values
        the `status` column holds, so filtering on them would return nothing
        rather than fail. They are refused so the caller learns why.
        """
        with pytest.raises(ValueError):
            normalize_browse_status(bad)

    def test_the_refusal_names_the_permitted_values(self):
        """UTC-68-TC-05 — the message is what the 400 carries to the console."""
        with pytest.raises(ValueError) as exc:
            normalize_browse_status("Banned")

        for permitted in BROWSE_STATUS_FILTERS:
            assert permitted in str(exc.value)


class TestNormalizeBrowseSpecies:
    """MD-74. The species half of the administrator browse filter."""

    def test_no_filter_means_every_report(self):
        """UTC-69-TC-01 — absence is not a value to refuse."""
        assert normalize_browse_species(None) is None

    @pytest.mark.parametrize("stored", BROWSE_SPECIES_FILTERS)
    def test_each_permitted_species_passes_through(self, stored):
        """UTC-69-TC-02 — 'Other' is still a browse filter, because reports
        created before 2026-09-10 hold it even though no new one may."""
        assert normalize_browse_species(stored) == stored

    def test_casing_and_surrounding_space_are_normalised(self):
        """UTC-69-TC-03 — the console sends what the user typed."""
        assert normalize_browse_species("  cat  ") == "Cat"

    @pytest.mark.parametrize("bad", ["Dragon", "Rabbit", ""])
    def test_a_species_outside_the_set_is_refused(self, bad):
        """UTC-69-TC-04."""
        with pytest.raises(ValueError):
            normalize_browse_species(bad)

    def test_the_refusal_names_the_permitted_values(self):
        """UTC-69-TC-05 — the message is what the 400 carries to the console."""
        with pytest.raises(ValueError) as exc:
            normalize_browse_species("Dragon")

        for permitted in BROWSE_SPECIES_FILTERS:
            assert permitted in str(exc.value)
