"""
Unit tests for the Pydantic request-schema validators — the normalize/reject
branches the service tests never exercise (they only ever build valid payloads).

Covered: species normalization + rejection (missing-pet + both sighting flows),
empty-characteristics rejection, hex-colour and pattern-id validation, the
missing-pet status validator, and the action_type validator that routes both
create flows through `sighting_logic.normalize_action_type`.
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

    def test_reopening_a_search_is_allowed(self):
        assert MissingPetUpdate(status="Searching").status == "Searching"

    def test_spotted_is_refused_even_though_the_enum_has_it(self):
        """`pet_status` still carries 'Spotted' from the old model, but the
        column is the MATCHING FILTER — match_missing_pets and
        get_nearby_missing_pets both select `status = 'Searching'` — so writing
        it here would pull a still-lost pet out of matching and off the map.
        "Someone has seen your pet" is the derived `post_status`, never stored.
        """
        with pytest.raises(ValidationError):
            MissingPetUpdate(status="Spotted")

    def test_resolved_is_refused_because_only_the_settlement_writes_it(self):
        """'Resolved' means the bounty has been paid — the administrator's
        resolve_missing_pet writes it, and only after the search is 'Found'."""
        with pytest.raises(ValidationError):
            MissingPetUpdate(status="Resolved")

    def test_the_rejection_names_the_values_in_a_stable_order(self):
        """A set would render the permitted values in whatever order it
        iterated, so the message an API client sees would change run to run."""
        with pytest.raises(ValidationError) as ei:
            MissingPetUpdate(status="Spotted")
        assert "Status must be one of Searching, Found" in str(ei.value)


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


# The create flows used to declare action_type as Literal["Spotted", "Caught"] —
# a second copy of the enum vocabulary that 422'd on "Rescue", the exact word
# the final-review button says and the exact word PATCH /sightings/{id}/action
# accepts. Both flows now defer to sighting_logic.normalize_action_type, so the
# three write paths answer the same way.
class TestSightingActionTypeValidators:
    def test_discovery_accepts_ui_wording(self):
        assert SightingCreate(**_sighting(action_type="Rescue")).action_type == "Caught"

    def test_discovery_is_case_insensitive(self):
        assert SightingCreate(**_sighting(action_type="RESCUE")).action_type == "Caught"

    def test_discovery_stored_value_passes_through(self):
        assert SightingCreate(**_sighting(action_type="caught")).action_type == "Caught"

    def test_discovery_defaults_to_spotted(self):
        data = _sighting()
        del data["action_type"]
        assert SightingCreate(**data).action_type == "Spotted"

    def test_discovery_rejects_nonsense(self):
        with pytest.raises(ValidationError):
            SightingCreate(**_sighting(action_type="banana"))

    def test_targeted_accepts_ui_wording(self):
        s = TargetedSightingCreate(**_sighting(action_type="Rescue", target_pet_id="p1"))
        assert s.action_type == "Caught"

    def test_targeted_rejects_nonsense(self):
        with pytest.raises(ValidationError):
            TargetedSightingCreate(**_sighting(action_type="banana", target_pet_id="p1"))
