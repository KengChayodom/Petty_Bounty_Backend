# schemas/missing_pets.py

"""
Pydantic schemas for missing pet operations.

This module provides request and response models for:
- Creating missing pet reports
- Updating missing pet information
- Missing pet responses
"""
from typing import Optional
from datetime import datetime
import re
from pydantic import BaseModel, Field, field_validator


class MissingPetCreate(BaseModel):
    """
    Request schema for creating a new missing pet report.

    The pet's photo will be analyzed to extract a feature vector
    for similarity matching.
    """

    owner_id: str = Field(
        ...,
        description="ID of the pet owner"
    )
    pet_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the missing pet"
    )
    species: str = Field(
        ...,
        description="Species of the pet (Cat, Dog, Bird, Other)"
    )
    characteristics: dict = Field(
        ...,
        description="Pet characteristics as JSON (e.g., color, size, markings)"
    )
    bounty_amount: float = Field(
        ...,
        ge=0,
        description="Bounty amount in local currency"
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="GPS longitude where pet was last seen (-180 to 180)"
    )
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="GPS latitude where pet was last seen (-90 to 90)"
    )
    last_seen_time: datetime = Field(
        ...,
        description="Timestamp when the pet was last seen"
    )
    image_url: str = Field(
        ...,
        description="Public URL of the pet's photo"
    )
    primary_color_hex: Optional[str] = Field(
        None,
        description="Primary coat color in hex format (e.g., #000000)"
    )
    pattern_id: Optional[str] = Field(
        None,
        description="Coat pattern identifier (solid, tabby, calico, etc.)"
    )

    @field_validator('species')
    @classmethod
    def species_must_be_valid(cls, v: str) -> str:
        """Validate and normalize the species value."""
        valid_species = {'Cat', 'Dog', 'Bird', 'Other'}
        normalized = v.capitalize()
        if normalized not in valid_species:
            raise ValueError(f"Species must be one of {valid_species}")
        return normalized

    @field_validator('characteristics')
    @classmethod
    def characteristics_must_not_be_empty(cls, v: dict) -> dict:
        """Validate that characteristics dictionary is not empty."""
        if not v:
            raise ValueError("Characteristics cannot be empty")
        return v

    @field_validator('primary_color_hex')
    @classmethod
    def validate_hex_color(cls, v: Optional[str]) -> Optional[str]:
        """Validate hex color format."""
        if v is None:
            return v
        if not re.match(r'^#[0-9A-Fa-f]{6}$', v):
            raise ValueError("Invalid hex color format. Use #RRGGBB format.")
        return v.upper()

    @field_validator('pattern_id')
    @classmethod
    def validate_pattern_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate pattern identifier."""
        if v is None:
            return v
        valid_patterns = {'solid', 'tabby', 'calico', 'tuxedo', 'spotted',
                          'striped', 'bicolor', 'tricolor', 'merle', 'brindle'}
        if v.lower() not in valid_patterns:
            raise ValueError(f"Invalid pattern. Must be one of {valid_patterns}")
        return v.lower()

    class Config:
        json_schema_extra = {
            "example": {
                "owner_id": "550e8400-e29b-41d4-a716-446655440000",
                "pet_name": "Luna",
                "species": "Dog",
                "characteristics": {
                    "color": "Golden",
                    "size": "Medium",
                    "markings": "White patch on chest"
                },
                "bounty_amount": 1000.00,
                "longitude": 100.5018,
                "latitude": 13.7563,
                "last_seen_time": "2025-01-12T10:30:00Z",
                "image_url": "https://example.com/photos/luna.jpg"
            }
        }


class MissingPetUpdate(BaseModel):
    """
    Request schema for updating a missing pet report.

    All fields are optional to allow partial updates.
    """

    pet_name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Updated name of the missing pet"
    )
    status: Optional[str] = Field(
        None,
        description="Updated status (Searching, Spotted, Found)"
    )
    bounty_amount: Optional[float] = Field(
        None,
        ge=0,
        description="Updated bounty amount"
    )
    characteristics: Optional[dict] = Field(
        None,
        description="Updated pet characteristics"
    )
    primary_color_hex: Optional[str] = Field(
        None,
        description="Updated primary coat color in hex format"
    )
    pattern_id: Optional[str] = Field(
        None,
        description="Updated coat pattern identifier"
    )

    @field_validator('status')
    @classmethod
    def status_must_be_valid(cls, v: Optional[str]) -> Optional[str]:
        """Validate that status is one of the allowed values."""
        if v is None:
            return v
        valid_statuses = {'Searching', 'Spotted', 'Found'}
        normalized = v.capitalize()
        if normalized not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        return normalized

    class Config:
        json_schema_extra = {
            "example": {
                "status": "Spotted",
                "bounty_amount": 1500.00
            }
        }


class MissingPetResponse(BaseModel):
    """
    Response schema for missing pet data.

    Includes all pet information with optional computed fields.
    """

    id: str
    owner_id: str
    pet_name: str
    species: str
    characteristics: dict
    bounty_amount: float
    latitude: float
    longitude: float    
    last_seen_time: datetime
    image_url: str
    feature_vector: list[float]
    status: str
    created_at: datetime

    # Optional color and pattern fields
    primary_color_hex: Optional[str] = None
    pattern_id: Optional[str] = None

    # Optional computed fields
    distance_meters: Optional[float] = None
    similarity: Optional[float] = None

    class Config:
        from_attributes = True
