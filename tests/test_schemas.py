"""
Unit tests for the Pydantic request-schema validators — the normalize/reject
branches the service tests never exercise (they only ever build valid payloads).

Covered: species normalization + rejection (missing-pet + both sighting flows),
empty-characteristics rejection, hex-colour and pattern-id validation, and the
missing-pet status validator.
"""
from datetime import datetime

import pytest
from pydantic import ValidationError

from app.schemas.missing_pets import MissingPetCreate, MissingPetUpdate
from app.schemas.sightings import SightingCreate, TargetedSightingCreate


def _create(**over):
    data = dict(
        pet_name="Luna", species="Dog", characteristics={"color": "brown"},
        bounty_amount=100.0, longitude=100.5, latitude=13.7,
        last_seen_time=datetime(2026, 1, 1), image_url="http://x/y.jpg",
    )
    data.update(over)
    return data


class TestMissingPetCreateValidators:
    def test_species_normalized_to_titlecase(self):
        assert MissingPetCreate(**_create(species="dog")).species == "Dog"

    def test_species_outside_set_rejected(self):
        with pytest.raises(ValidationError):
            MissingPetCreate(**_create(species="dragon"))

    def test_empty_characteristics_rejected(self):
        with pytest.raises(ValidationError):
            MissingPetCreate(**_create(characteristics={}))

    def test_hex_none_passes_through(self):
        assert MissingPetCreate(**_create(primary_color_hex=None)).primary_color_hex is None

    def test_hex_valid_is_uppercased(self):
        assert MissingPetCreate(
            **_create(primary_color_hex="#abc123")
        ).primary_color_hex == "#ABC123"

    def test_hex_malformed_rejected(self):
        with pytest.raises(ValidationError):
            MissingPetCreate(**_create(primary_color_hex="red"))

    def test_pattern_none_passes_through(self):
        assert MissingPetCreate(**_create(pattern_id=None)).pattern_id is None

    def test_pattern_valid_is_lowercased(self):
        assert MissingPetCreate(**_create(pattern_id="Tabby")).pattern_id == "tabby"

    def test_pattern_unknown_rejected(self):
        with pytest.raises(ValidationError):
            MissingPetCreate(**_create(pattern_id="polkadot"))


class TestMissingPetUpdateValidators:
    def test_status_none_passes_through(self):
        assert MissingPetUpdate(status=None).status is None

    def test_status_normalized(self):
        assert MissingPetUpdate(status="found").status == "Found"

    def test_status_outside_set_rejected(self):
        with pytest.raises(ValidationError):
            MissingPetUpdate(status="lost")


def _sighting(**over):
    data = dict(
        hunter_id="h1", image_url="http://x/y.jpg", latitude=13.7, longitude=100.5,
        detected_species="Dog", action_type="Spotted",
    )
    data.update(over)
    return data


class TestSightingSpeciesValidators:
    def test_discovery_species_normalized(self):
        assert SightingCreate(
            **_sighting(detected_species="dog")
        ).detected_species == "Dog"

    def test_discovery_species_rejected(self):
        with pytest.raises(ValidationError):
            SightingCreate(**_sighting(detected_species="dragon"))

    def test_targeted_species_normalized(self):
        s = TargetedSightingCreate(
            **_sighting(detected_species="cat", target_pet_id="p1")
        )
        assert s.detected_species == "Cat"

    def test_targeted_species_rejected(self):
        with pytest.raises(ValidationError):
            TargetedSightingCreate(
                **_sighting(detected_species="dragon", target_pet_id="p1")
            )
