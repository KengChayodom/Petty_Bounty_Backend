"""
Unit tests for app/services/pet_service.py:

  * UTC-12  get_nearby_missing_pets (MD-15, SRS-24) — Home Map proximity query:
    WKT centre in POINT(lng lat) form, km->m conversion, limit passthrough,
    empty-result normalisation, transport-error re-raise.
  * UTC-13  register_missing_pet (MD-11, SRS-47, trigger of SRS-21) — owner
    report runs the same mask-isolate + CLIP path as the live sighting save
    (full-frame fallback on a YOLO miss), builds the PostGIS point, inserts with
    status "Searching" and the feature vector; a no-row insert raises.

Boundary rule (per db-testing-seams): the DB is reached only through the
MissingPetRepository port, so we double THAT with MagicMock(spec=...). The AI
pipeline is mocked at the AIManager class boundary. We assert on the payload the
service hands the repo (the insert contract) and on the semantic arguments to
the nearby query — the calls that ARE the behaviour — plus the returned rows.
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.repositories.missing_pet_repository import (
    MissingPetNotSaved,
    MissingPetRepository,
)
from app.schemas.missing_pets import MissingPetCreate
from app.services.ai_service import AIManager
from app.services.pet_service import PetService


def run(coro):
    return asyncio.run(coro)


def _repo():
    return MagicMock(spec=MissingPetRepository)


# --------------------------------------------------------------------------- #
# UTC-12  get_nearby_missing_pets
# --------------------------------------------------------------------------- #
class TestGetNearbyMissingPets:
    def test_builds_wkt_centre_metre_radius_and_limit(self):
        repo = _repo()
        repo.get_nearby_missing_pets.return_value = [{"id": "pet-1"}]

        rows = run(PetService.get_nearby_missing_pets(
            repo, latitude=13.7563, longitude=100.5018, radius_km=5.0, limit=7,
        ))

        assert rows == [{"id": "pet-1"}]
        # WKT is POINT(lng lat); km -> metres; limit passed through verbatim.
        repo.get_nearby_missing_pets.assert_called_once_with(
            "POINT(100.5018 13.7563)", 5000.0, 7
        )

    def test_defaults_radius_and_limit(self):
        repo = _repo()
        repo.get_nearby_missing_pets.return_value = []

        run(PetService.get_nearby_missing_pets(repo, latitude=0.0, longitude=0.0))

        repo.get_nearby_missing_pets.assert_called_once_with(
            "POINT(0.0 0.0)", 10000.0, 20
        )

    def test_no_rows_returns_empty_list(self):
        repo = _repo()
        repo.get_nearby_missing_pets.return_value = []
        assert run(PetService.get_nearby_missing_pets(repo, 1.0, 2.0)) == []

    def test_rpc_error_is_reraised(self):
        repo = _repo()
        repo.get_nearby_missing_pets.side_effect = RuntimeError("RPC transport error")
        with pytest.raises(RuntimeError):
            run(PetService.get_nearby_missing_pets(repo, 1.0, 2.0))


# --------------------------------------------------------------------------- #
# UTC-13  register_missing_pet
# --------------------------------------------------------------------------- #
def _make_pet(**overrides):
    data = {
        "owner_id": "owner-1",
        "pet_name": "Luna",
        "species": "Dog",
        "characteristics": {"color": "brown"},
        "bounty_amount": 1500,
        "latitude": 13.7563,
        "longitude": 100.5018,
        "last_seen_time": datetime(2026, 6, 1, 12, 0, 0),
        "image_url": "https://img.example/pet.jpg",
    }
    data.update(overrides)
    return MissingPetCreate(**data)


def _patch_ai(monkeypatch, *, isolate_return, vector):
    """Mock the AIManager pipeline at the class boundary.
    download/yolo/clip are awaited; isolate_subject is synchronous."""
    monkeypatch.setattr(AIManager, "download_image", AsyncMock(return_value="SRC_IMG"))
    monkeypatch.setattr(AIManager, "run_yolo_seg", AsyncMock(return_value="YOLO_RESULTS"))
    monkeypatch.setattr(AIManager, "isolate_subject", MagicMock(return_value=isolate_return))
    monkeypatch.setattr(AIManager, "clip_encode", AsyncMock(return_value=vector))


class TestRegisterMissingPet:
    def test_success_inserts_isolated_vector(self, monkeypatch):
        vector = [0.1, 0.2, 0.3]
        _patch_ai(
            monkeypatch,
            isolate_return=("CROP_IMG", "Dog", 0.9, [1.0, 2.0, 3.0, 4.0]),
            vector=vector,
        )
        repo = _repo()
        repo.insert_missing_pet.return_value = {"id": "pet-xyz"}

        created = run(PetService.register_missing_pet(repo, _make_pet(species="Dog")))

        assert created == {"id": "pet-xyz"}
        repo.insert_missing_pet.assert_called_once()
        payload = repo.insert_missing_pet.call_args.args[0]
        assert payload["feature_vector"] == vector
        assert payload["status"] == "Searching"
        assert payload["species"] == "Dog"
        assert payload["last_seen_location"] == "POINT(100.5018 13.7563)"
        # isolate constrained to the user-confirmed species; CLIP encodes the crop
        AIManager.isolate_subject.assert_called_once_with(
            "SRC_IMG", "YOLO_RESULTS", expected_species="Dog"
        )
        AIManager.clip_encode.assert_awaited_once_with("CROP_IMG")

    def test_yolo_miss_falls_back_to_full_frame(self, monkeypatch):
        vector = [0.4, 0.5]
        _patch_ai(monkeypatch, isolate_return=None, vector=vector)
        repo = _repo()
        repo.insert_missing_pet.return_value = {"id": "pet-xyz"}

        run(PetService.register_missing_pet(repo, _make_pet()))

        # On a YOLO miss CLIP encodes the original downloaded frame, and that
        # vector is what gets inserted.
        AIManager.clip_encode.assert_awaited_once_with("SRC_IMG")
        assert repo.insert_missing_pet.call_args.args[0]["feature_vector"] == vector

    def test_insert_returning_no_row_raises_valueerror(self, monkeypatch):
        _patch_ai(
            monkeypatch,
            isolate_return=("CROP_IMG", "Dog", 0.9, [1.0, 2.0, 3.0, 4.0]),
            vector=[0.1],
        )
        repo = _repo()
        # The adapter raises MissingPetNotSaved (a ValueError) on an empty insert.
        repo.insert_missing_pet.side_effect = MissingPetNotSaved({})

        with pytest.raises(ValueError):
            run(PetService.register_missing_pet(repo, _make_pet()))

    def test_ai_pipeline_error_is_wrapped_with_context(self, monkeypatch):
        # A NON-ValueError from the AI pipeline is caught and re-raised as a
        # generic Exception carrying context (distinct from the ValueError path).
        monkeypatch.setattr(
            AIManager, "download_image", AsyncMock(side_effect=RuntimeError("net")))
        repo = _repo()

        with pytest.raises(Exception) as ei:
            run(PetService.register_missing_pet(repo, _make_pet()))

        assert "Failed to register missing pet" in str(ei.value)
        repo.insert_missing_pet.assert_not_called()


# --------------------------------------------------------------------------- #
# get_missing_pet_by_id — thin delegate; a repo error must propagate.
# --------------------------------------------------------------------------- #
class TestGetMissingPetById:
    def test_returns_repo_row(self):
        repo = _repo()
        repo.get_missing_pet_by_id.return_value = {"id": "pet-1", "latitude": 13.7}
        assert run(PetService.get_missing_pet_by_id(repo, "pet-1")) == {
            "id": "pet-1", "latitude": 13.7,
        }

    def test_error_is_reraised(self):
        repo = _repo()
        repo.get_missing_pet_by_id.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            run(PetService.get_missing_pet_by_id(repo, "pet-1"))


# --------------------------------------------------------------------------- #
# get_sightings_for_pet — owner timeline; forces include_dismissed=False and
# propagates repo errors.
# --------------------------------------------------------------------------- #
class TestGetSightingsForPet:
    def test_returns_rows_and_forces_include_dismissed_false(self):
        repo = _repo()
        repo.sightings_for_pet.return_value = [{"id": "s1"}]
        out = run(PetService.get_sightings_for_pet(repo, "pet-1", limit=10, offset=5))
        assert out == [{"id": "s1"}]
        # the owner never sees Dismissed reports
        repo.sightings_for_pet.assert_called_once_with(
            "pet-1", 10, 5, include_dismissed=False
        )

    def test_error_is_reraised(self):
        repo = _repo()
        repo.sightings_for_pet.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            run(PetService.get_sightings_for_pet(repo, "pet-1"))
