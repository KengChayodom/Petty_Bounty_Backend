"""Pydantic schemas for admin-only operations (Feature #2)."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.services.moderation_logic import (
    MAX_PENALTY_POINTS,
    PENALTY_POINTS_BY_REASON,
)


class VerifySightingRequest(BaseModel):
    """Body of PATCH /admin/sightings/{id}/verification."""

    verification_status: Literal["Verified", "Dismissed"] = Field(
        ...,
        description="New verification state. 'Pending' is the default at "
                    "creation time and is not settable here.",
    )
    note: Optional[str] = Field(
        None, max_length=500,
        description="Optional admin note (not currently persisted; reserved "
                    "for an audit trail in a later iteration).",
    )


class ResolveMissingPetRequest(BaseModel):
    """
    Body of POST /admin/missing-pets/{id}/resolve.

    The admin picks which Caught sighting becomes the bounty-paying
    'final_sighting' — they may need to disambiguate when more than one
    hunter claims to have caught the pet.
    """

    final_sighting_id: str = Field(
        ...,
        description="UUID of the Caught + Verified sighting whose hunter "
                    "receives the full bounty payout.",
    )
    slip_image_url: str = Field(
        ...,
        description="URL of the bank-transfer slip image — proof of payout.",
    )
    reference_no: Optional[str] = Field(
        None, max_length=255,
        description="Transfer reference number (optional but useful for audit).",
    )


class ReviewReportRequest(BaseModel):
    """
    Body of PATCH /admin/reports/{report_id} (MD-40).

    `decision` is a plain string rather than a Literal so an unrecognised value
    comes back as the 400 the spec calls for (raised by
    `moderation_logic.normalize_flag_decision`) instead of FastAPI's 422, and so
    the UD-14 wording ("Dismiss Flag" / "Uphold and Ban User") is accepted
    alongside the enum values.
    """

    decision: str = Field(
        ...,
        description="'Dismissed' or 'Reviewed_Penalty' (also accepts 'Dismiss "
                    "Flag', 'Uphold and Penalise User', and the legacy ban "
                    "wording — upholding deducts score, it never bans).",
    )
    # Deliberately unconstrained here: the range is enforced in
    # `moderation_logic.resolve_penalty_points`, which raises ValueError and so
    # surfaces as the 400 this endpoint documents. Declaring ge/le would make
    # Pydantic reject it first as a 422, contradicting `decision` right above.
    penalty_points: Optional[int] = Field(
        None,
        description=(
            "Score to deduct from the reported hunter. Omit to charge the "
            "per-reason default ("
            + ", ".join(
                f"{r}={p}" for r, p in PENALTY_POINTS_BY_REASON.items()
            )
            + f"). Must be 0-{MAX_PENALTY_POINTS}. 0 upholds the flag and "
            "withdraws the sighting without any deduction. Ignored when "
            "`decision` is 'Dismissed'."
        ),
    )
