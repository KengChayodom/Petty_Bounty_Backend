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
    MAX_PENALTY_POINTS,
    PENALTY_POINTS_BY_REASON,
    build_flag_payload,
    normalize_flag_decision,
    normalize_flag_reason,
    normalize_flag_status_filter,
    resolve_penalty_points,
)


class TestNormalizeFlagReason:
    @pytest.mark.parametrize("supplied,expected", [
        # the enum spellings (UD-16's Input Specification)
        ("Spam", "Spam"),
        ("Not_a_pet", "Not_a_pet"),
        ("Inappropriate_image", "Inappropriate_image"),
        # the prose spellings (MD-43's parameter table)
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
        ("Reviewed and banned", DECISION_UPHOLD),    # MD-44's wording
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
        entirely — the MD-41 convention. Returning a string here would filter
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


class TestResolvePenaltyPoints:
    """UTC-55 — how many points an upheld flag costs its hunter.

    These were framed through AdminService.review_report until 09/09/2026, which
    meant standing up two repository stubs, a Pending flag and a sighting
    carrying a hunter just to read back a number this function returns on its
    own. The rules live here; UTC-40 keeps only the seam, meaning that whatever
    this returns is what reaches the deduction.
    """

    @pytest.mark.parametrize("reason,expected", sorted(
        PENALTY_POINTS_BY_REASON.items()
    ))
    def test_each_reason_charges_its_own_tariff(self, reason, expected):
        """Omitting the figure falls back to the tariff for the reason, and the
        three reasons are not interchangeable: each selects a different default,
        which is the whole point of keeping a table rather than one constant."""
        assert resolve_penalty_points(reason, None) == expected

    @pytest.mark.parametrize("custom", [1, 7, MAX_PENALTY_POINTS])
    def test_an_administrators_figure_overrides_the_tariff(self, custom):
        """The ruling an administrator actually made wins over the default, for
        every reason, so the tariff is a starting point and not a cap."""
        for reason in FLAG_REASONS:
            assert resolve_penalty_points(reason, custom) == custom

    def test_zero_is_a_figure_and_not_an_omission(self):
        """0 must survive as a deliberate ruling: uphold the flag, withdraw the
        sighting, charge nothing. A truthiness test here would read 0 as "not
        supplied" and silently charge the tariff instead, which is the one
        substitution the caller cannot detect."""
        for reason in FLAG_REASONS:
            assert resolve_penalty_points(reason, 0) == 0
            assert PENALTY_POINTS_BY_REASON[reason] != 0

    @pytest.mark.parametrize("bad", [-1, -100, MAX_PENALTY_POINTS + 1, 10_000])
    def test_a_figure_outside_the_range_is_refused(self, bad):
        """Both bounds are refused. The cap exists so a mistyped extra digit
        cannot wipe a hunter's whole history, and the floor because a negative
        deduction would be an award."""
        with pytest.raises(ValueError):
            resolve_penalty_points("Spam", bad)

    @pytest.mark.parametrize("unknown", ["Harassment", "", None, "spam"])
    def test_a_reason_the_table_does_not_hold_charges_the_mildest_figure(
        self, unknown
    ):
        """It does not raise. The reason was validated by normalize_flag_reason
        when the flag was created, so an unknown value here means the enum grew
        a member this table has not caught up with, and refusing to moderate the
        queue is a worse failure than under-charging one hunter. Note "spam" in
        the wrong casing is among these: this function does not normalise, it
        looks up, so the mildest figure is what a caller who skipped the
        normaliser gets."""
        assert resolve_penalty_points(unknown, None) == min(
            PENALTY_POINTS_BY_REASON.values()
        )
