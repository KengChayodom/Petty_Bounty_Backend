"""
Unit tests for app/services/pet_service.py:

  * UTC-12  get_nearby_missing_pets (MD-15, SRS-24) — Home Map proximity query:
    WKT centre in POINT(lng lat) form, km->m conversion, limit passthrough,
    empty-result normalisation, transport-error re-raise.
  * UTC-13  register_missing_pet (MD-11, trigger of SRS-21) — owner report runs
    the same mask-isolate + CLIP path as the live sighting save (full-frame
    fallback on a YOLO miss), builds the PostGIS point, inserts with status
    "Searching" and the feature vector; a no-row insert raises.

The Supabase client is the FakeSupabase boundary double; the AIManager pipeline
is mocked at the class boundary (download/yolo/isolate/clip).
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas.missing_pets import MissingPetCreate
from app.services.ai_service import AIManager
from app.services.pet_service import PetService


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# UTC-12  get_nearby_missing_pets
# --------------------------------------------------------------------------- #
class _RaisingRpcDB:
    """Minimal Supabase double whose rpc(...).execute() raises."""
    def rpc(self, *_a, **_k):
        class _C:
            def execute(self_inner):
                raise RuntimeError("RPC transport error")
        return _C()


class TestGetNearbyMissingPets:
    def test_builds_wkt_centre_metre_radius_and_limit(self, fake_db):
        fake_db.set_rpc_result("get_nearby_missing_pets", [{"id": "pet-1"}])

        rows = run(PetService.get_nearby_missing_pets(
            fake_db, latitude=13.7563, longitude=100.5018, radius_km=5.0, limit=7,
        ))

        assert rows == [{"id": "pet-1"}]
        name, params = fake_db.rpc_calls[-1]
        assert name == "get_nearby_missing_pets"
        assert params["center_location"] == "POINT(100.5018 13.7563)"
        assert params["radius_meters"] == 5000.0
        assert params["limit"] == 7

    def test_defaults_radius_and_limit(self, fake_db):
        fake_db.set_rpc_result("get_nearby_missing_pets", [])

        run(PetService.get_nearby_missing_pets(fake_db, latitude=0.0, longitude=0.0))

        _name, params = fake_db.rpc_calls[-1]
        assert params["radius_meters"] == 10000.0
        assert params["limit"] == 20

    def test_no_rows_returns_empty_list(self, fake_db):
        fake_db.set_rpc_result("get_nearby_missing_pets", None)
        assert run(PetService.get_nearby_missing_pets(fake_db, 1.0, 2.0)) == []

    def test_rpc_error_is_reraised(self):
        with pytest.raises(RuntimeError):
            run(PetService.get_nearby_missing_pets(_RaisingRpcDB(), 1.0, 2.0))


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
    def test_success_inserts_isolated_vector(self, fake_db, monkeypatch):
        vector = [0.1, 0.2, 0.3]
        _patch_ai(
            monkeypatch,
            isolate_return=("CROP_IMG", "Dog", 0.9, [1.0, 2.0, 3.0, 4.0]),
            vector=vector,
        )
        fake_db.set_table_result("missing_pets", "insert", data=[{"id": "pet-xyz"}])

        created = run(PetService.register_missing_pet(fake_db, _make_pet(species="Dog")))

        assert created == {"id": "pet-xyz"}
        payload = fake_db.payload_for("missing_pets", "insert")
        assert payload["feature_vector"] == vector
        assert payload["status"] == "Searching"
        assert payload["species"] == "Dog"
        assert payload["last_seen_location"] == "POINT(100.5018 13.7563)"
        # isolate constrained to the user-confirmed species; CLIP encodes the crop
        AIManager.isolate_subject.assert_called_once_with(
            "SRC_IMG", "YOLO_RESULTS", expected_species="Dog"
        )
        AIManager.clip_encode.assert_awaited_once_with("CROP_IMG")

    def test_yolo_miss_falls_back_to_full_frame(self, fake_db, monkeypatch):
        vector = [0.4, 0.5]
        _patch_ai(monkeypatch, isolate_return=None, vector=vector)
        fake_db.set_table_result("missing_pets", "insert", data=[{"id": "pet-xyz"}])

        run(PetService.register_missing_pet(fake_db, _make_pet()))

        # On a YOLO miss CLIP encodes the original downloaded frame, and that
        # vector is what gets inserted.
        AIManager.clip_encode.assert_awaited_once_with("SRC_IMG")
        assert fake_db.payload_for("missing_pets", "insert")["feature_vector"] == vector

    def test_insert_returning_no_row_raises_valueerror(self, fake_db, monkeypatch):
        _patch_ai(
            monkeypatch,
            isolate_return=("CROP_IMG", "Dog", 0.9, [1.0, 2.0, 3.0, 4.0]),
            vector=[0.1],
        )
        # fake_db default insert result is data=[] -> the "no data returned" guard.
        with pytest.raises(ValueError):
            run(PetService.register_missing_pet(fake_db, _make_pet()))
