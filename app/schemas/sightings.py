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
from typing import Optional

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
                "notes": "Spotted near the park entrance",
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

    class Config:
        from_attributes = True
