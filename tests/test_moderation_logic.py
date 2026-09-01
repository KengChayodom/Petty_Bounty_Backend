"""
Unit tests for app/services/moderation_logic.py — the pure normalisers that
reconcile the spec's spellings with the database enums.

These have no I/O at all (TEST_PLAN §3 layer L1). The defect they catch is
concrete: a reason or decision that reaches the DB in the spec's spacing
("Not a pet", "Reviewed and banned") is not a member of the `report_reason` /
`report_status` enums, so the INSERT/UPDATE fails at the cast — a 500 for what
should be either a clean write or a 400.
"""
import pytest

from app.services.moderation_logic import (
    DECISION_DISMISS,
    DECISION_UPHOLD,
    FLAG_REASONS,
    FLAG_STATUSES,
    build_flag_payload,
    normalize_flag_decision,
    normalize_flag_reason,
    normalize_flag_status_filter,
)


class TestNormalizeFlagReason:
    @pytest.mark.parametrize("supplied,expected", [
        # the enum spellings (UD-16's Input Specification)
        ("Spam", "Spam"),
        ("Not_a_pet", "Not_a_pet"),
        ("Inappropriate_image", "Inappropriate_image"),
        # the prose spellings (MD-39's parameter table)
        ("Not a pet", "Not_a_pet"),
        ("Inappropriate image", "Inappropriate_image"),
        # user-interface casing / padding
        ("  spam  ", "Spam"),
        ("INAPPROPRIATE IMAGE", "Inappropriate_image"),
    ])
    def test_maps_onto_the_enum(self, supplied, expected):
        assert normalize_flag_reason(supplied) == expected
        assert expected in FLAG_REASONS

    @pytest.mark.parametrize("bad", ["Ugly", "", "   ", None, "Pending"])
    def test_rejects_anything_else(self, bad):
        with pytest.raises(ValueError):
            normalize_flag_reason(bad)


class TestNormalizeFlagDecision:
    @pytest.mark.parametrize("supplied,expected", [
        ("Dismissed", DECISION_DISMISS),
        ("Dismiss Flag", DECISION_DISMISS),          # UD-16's Action wording
        ("Reviewed_Penalty", DECISION_UPHOLD),
        ("Reviewed_Ban", DECISION_UPHOLD),           # pre-2026-08-20 enum name
        ("Reviewed and banned", DECISION_UPHOLD),    # MD-40's wording
        ("Uphold and Ban User", DECISION_UPHOLD),    # UD-16's Action wording
        ("Uphold and Penalise User", DECISION_UPHOLD),
    ])
    def test_maps_onto_the_enum(self, supplied, expected):
        assert normalize_flag_decision(supplied) == expected

    @pytest.mark.parametrize("bad", ["Pending", "Banned", "", None])
    def test_rejects_anything_else(self, bad):
        """'Pending' is the notable one: it is a real report_status value but
        it is a starting state, not a decision, so it must not be writable."""
        with pytest.raises(ValueError):
            normalize_flag_decision(bad)


class TestNormalizeFlagStatusFilter:
    @pytest.mark.parametrize("supplied,expected", [
        ("Pending", "Pending"),
        ("Dismissed", DECISION_DISMISS),
        ("Reviewed_Penalty", DECISION_UPHOLD),
        ("  pending  ", "Pending"),          # user-interface casing / padding
        ("REVIEWED_PENALTY", DECISION_UPHOLD),
    ])
    def test_maps_onto_the_enum(self, supplied, expected):
        assert normalize_flag_status_filter(supplied) == expected
        assert expected in FLAG_STATUSES

    def test_pending_is_accepted_here_unlike_a_decision(self):
        """The one deliberate difference from normalize_flag_decision: Pending
        is not a decision an admin may write, but it IS the bucket they read
        the queue to find, so it must survive as a filter."""
        assert normalize_flag_status_filter("Pending") == "Pending"
        with pytest.raises(ValueError):
            normalize_flag_decision("Pending")

    def test_none_means_every_status_not_a_null_status(self):
        """None passes through untouched so the adapter can skip the predicate
        entirely — the MD-37 convention. Returning a string here would filter
        the queue down to one bucket by accident."""
        assert normalize_flag_status_filter(None) is None

    @pytest.mark.parametrize("bad", ["Banned", "Reviewed", "", "   ", "Spam"])
    def test_rejects_anything_else(self, bad):
        """Rejecting at the edge is what keeps an unknown filter a 400 instead
        of a failed PostgREST enum cast surfacing as a 500. 'Spam' is the trap:
        a real enum value, but of report_reason, not report_status."""
        with pytest.raises(ValueError):
            normalize_flag_status_filter(bad)

    def test_decision_aliases_are_not_filter_values(self):
        """'uphold' is an instruction, not a bucket; accepting it here would
        make the filter's vocabulary quietly differ from what it returns."""
        for alias in ("uphold", "Dismiss Flag", "Reviewed and banned"):
            with pytest.raises(ValueError):
                normalize_flag_status_filter(alias)


class TestBuildFlagPayload:
    def test_insert_contract(self):
        """A flag is always born Pending and always attributed to the verified
        caller — neither is something the request body can set."""
        payload = build_flag_payload("s1", "Not a pet", "r1")
        assert payload == {
            "sighting_id": "s1",
            "reason": "Not_a_pet",
            "reporter_id": "r1",
            "status": "Pending",
        }

    def test_bad_reason_raises_before_a_payload_exists(self):
        with pytest.raises(ValueError):
            build_flag_payload("s1", "Ugly", "r1")
