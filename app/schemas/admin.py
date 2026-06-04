"""Pydantic schemas for admin-only operations (Feature #2)."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


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
