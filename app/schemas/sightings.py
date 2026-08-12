# schemas/sightings.py
"""
Pydantic schemas for sighting-related operations.

2-step flow (mandated by wiki/api_flow.md Flow 3, step 6 — user must be
able to override YOLO's species guess):

    POST /sightings/analyze  →  server returns {species, confidence, bbox}
                                  AND caches the pre-computed CLIP vector
    POST /sightings/         →  client sends user-CONFIRMED species; server
                                  pulls the cached vector and saves
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    """Request schema for POST /sightings/analyze."""

    image_url: str = Field(
        ...,
        description="Public URL of the image to analyze",
    )

    class Config:
        json_schema_extra = {
            "example": {"image_url": "https://example.com/photos/sighting.jpg"}
        }


class SightingCreate(BaseModel):
    """
    Request schema for POST /sightings/.

    `detected_species` is the user's confirmed species (possibly a manual
    override of YOLO's guess from the /analyze step). The server stores
    this value verbatim and uses it to filter the pgvector match query —
    YOLO's guess is NOT trusted automatically.
    """

    hunter_id: str = Field(
        ...,
        description="ID of the bounty hunter (currently overridden server-side by TEST_USER_ID).",
    )
    image_url: str = Field(
        ...,
        description="Public URL of the sighting photo. Must match the URL "
                    "sent to /sightings/analyze for the cache to hit.",
    )
    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="GPS latitude where the pet was sighted (-90 to 90)",
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="GPS longitude where the pet was sighted (-180 to 180)",
    )
    detected_species: str = Field(
        ..., min_length=1, max_length=50,
        description="User-confirmed species (Cat / Dog / Bird / Other). "
                    "Validator normalises to title-case.",
    )
    action_type: Literal["Spotted", "Caught"] = Field(
        "Spotted",
        description="Whether the hunter only spotted the pet or actually "
                    "caught/returned it. 'Caught' is the action that makes a "
                    "sighting eligible to be the bounty-paying final sighting.",
    )
    notes: Optional[str] = Field(
        None, max_length=500,
        description="Optional free-form notes from the hunter",
    )

    @field_validator("detected_species")
    @classmethod
    def species_must_be_valid(cls, v: str) -> str:
        valid_species = {"Cat", "Dog", "Bird", "Other"}
        normalized = v.capitalize()
        if normalized not in valid_species:
            raise ValueError(f"Species must be one of {valid_species}")
        return normalized

    class Config:
        json_schema_extra = {
            "example": {
                "hunter_id": "550e8400-e29b-41d4-a716-446655440000",
                "image_url": "https://example.com/photos/sighting.jpg",
                "latitude": 13.7563,
                "longitude": 100.5018,
                "detected_species": "Dog",
                "action_type": "Spotted",
                "notes": "Spotted near the park entrance",
            }
        }


class TargetedSightingCreate(BaseModel):
    """
    Request schema for POST /sightings/targeted.

    The targeted flow: the hunter is looking at ONE known missing-pet listing
    and reporting it straight to that pet's owner. There is no AI species
    analysis and no similarity matching — the pet is already chosen by eye —
    so this path has no `feature_vector` and never runs the match RPC. That is
    why it is a separate endpoint from the discovery `POST /sightings/` rather
    than a mode flag on it: `target_pet_id` is REQUIRED here, which makes the
    "targeted but no target" nonsense state unrepresentable.
    """

    hunter_id: str = Field(
        ...,
        description="ID of the bounty hunter (overridden server-side from the JWT).",
    )
    image_url: str = Field(
        ...,
        description="Public URL of the sighting photo.",
    )
    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="GPS latitude where the pet was sighted (-90 to 90)",
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="GPS longitude where the pet was sighted (-180 to 180)",
    )
    detected_species: str = Field(
        ..., min_length=1, max_length=50,
        description="Species carried over from the targeted pet (no AI detection "
                    "runs in this flow). Validator normalises to title-case.",
    )
    target_pet_id: str = Field(
        ...,
        description="UUID of the missing-pet listing being reported against. "
                    "Required — the targeted flow always has a chosen pet.",
    )
    action_type: Literal["Spotted", "Caught"] = Field(
        "Spotted",
        description="Whether the hunter only spotted the pet or actually "
                    "caught/returned it.",
    )
    notes: Optional[str] = Field(
        None, max_length=500,
        description="Optional free-form notes from the hunter",
    )

    @field_validator("detected_species")
    @classmethod
    def species_must_be_valid(cls, v: str) -> str:
        valid_species = {"Cat", "Dog", "Bird", "Other"}
        normalized = v.capitalize()
        if normalized not in valid_species:
            raise ValueError(f"Species must be one of {valid_species}")
        return normalized

    class Config:
        json_schema_extra = {
            "example": {
                "hunter_id": "550e8400-e29b-41d4-a716-446655440000",
                "image_url": "https://example.com/photos/sighting.jpg",
                "latitude": 13.7563,
                "longitude": 100.5018,
                "detected_species": "Dog",
                "target_pet_id": "660e8400-e29b-41d4-a716-446655440111",
                "action_type": "Spotted",
                "notes": "Reported straight to the owner from the pet's page",
            }
        }


class SightingResponse(BaseModel):
    """Typed view of a stored sighting (feature_vector intentionally omitted)."""

    id: str
    hunter_id: str
    image_url: str
    sighted_location: str
    detected_species: str
    action_type: str
    sighting_status: str
    created_at: str
    distance_meters: Optional[float] = None
    similarity: Optional[float] = None
    # Coat colour auto-extracted from the sighting photo ('#RRGGBB'); NULL when
    # unreadable (near-black subject / full-frame fallback). Used server-side to
    # exclude clearly-wrong-colour matches and to re-rank the rest.
    primary_color_hex: Optional[str] = None

    class Config:
        from_attributes = True
