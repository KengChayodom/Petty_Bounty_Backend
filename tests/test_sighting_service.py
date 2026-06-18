"""
Unit tests for app/services/sighting_service.py.

Scope: the deterministic business logic — status validation, match-row
mapping, similarity-threshold filtering, the get_matches precondition guards,
and the headline invariant that the *user-confirmed* species (not YOLO's
guess) and the *cached* vector are what get persisted on the hot path.

The Supabase client is faked at the boundary (see conftest.FakeSupabase);
the AI manager is a MagicMock that the cache-hit path must never touch.

Progress-I SRS traceability:
  * SRS-31 — sighting save returns the ranked matches (TestProcessAndSaveCacheHit,
    TestGetMatches).
  * SRS-38 — the sighting row stores the USER-confirmed species + the cached
    vector, and candidate pets are linked via sighting_matches; the explicit
    Pet ID is stored only for the targeted flow (TestPersistMatches,
    TestProcessAndSaveCacheHit::{test_persists_user_species_and_cached_vector_not_yolo,
    test_persists_matches_into_sighting_matches,
    test_target_pet_id_is_threaded_into_insert_payload,
    test_initial_target_pet_id_omitted_when_no_target, test_targeted_path_skips_vector_and_match}).
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas.sightings import SightingCreate
from app.services.ai_cache import AnalyzeCache
from app.services.sighting_service import SightingService


def run(coro):
    """Drive an async coroutine without pulling in pytest-asyncio."""
    return asyncio.run(coro)


def make_sighting(**overrides):
    data = {
        "hunter_id": "hunter-1",
        "image_url": "https://img.example/sighting.jpg",
        "latitude": 13.7563,
        "longitude": 100.5018,
        "detected_species": "Dog",
        "action_type": "Spotted",
    }
    data.update(overrides)
    return SightingCreate(**data)


def wire_save_path(fake_db, stored_vector, similarity=0.88):
    """Program the DB responses the post-vector tail of process_and_save needs:
    the INSERT echo, the get_matches re-read, and the match RPC result."""
    fake_db.set_table_result("sightings", "insert", data=[{
        "id": "s1",
        "hunter_id": "hunter-1",
        "detected_species": "Dog",
        "feature_vector": str(stored_vector),   # Supabase stores it as text
        "sighted_location": "POINT(100.5018 13.7563)",
    }])
    fake_db.set_table_result("sightings", "select", data=[{
        "id": "s1",
        "feature_vector": stored_vector,
        "detected_species": "Dog",
        "sighted_location": "POINT(100.5018 13.7563)",
    }])
    fake_db.set_rpc_result("match_missing_pets", [{"id": "pet-1", "similarity": similarity}])


# --------------------------------------------------------------------------- #
# update_sighting_status — pure allow-list validation + not-found handling.
# --------------------------------------------------------------------------- #
class TestUpdateSightingStatus:
    def test_rejects_status_outside_allowlist(self, fake_db):
        svc = SightingService(fake_db, ai_manager=None)
        with pytest.raises(ValueError):
            run(svc.update_sighting_status("s1", "Teleported"))
        # invalid status must be caught *before* any DB write
        assert fake_db.recorded == []

    def test_accepts_valid_status_and_returns_row(self, fake_db):
        fake_db.set_table_result(
            "sightings", "update",
            data=[{"id": "s1", "sighting_status": "Confirmed"}],
        )
        svc = SightingService(fake_db, ai_manager=None)
        row = run(svc.update_sighting_status("s1", "Confirmed"))
        assert row["sighting_status"] == "Confirmed"

    def test_unknown_sighting_id_raises(self, fake_db):
        fake_db.set_table_result("sightings", "update", data=[])  # nothing updated
        svc = SightingService(fake_db, ai_manager=None)
        with pytest.raises(ValueError):
            run(svc.update_sighting_status("ghost", "Confirmed"))


# --------------------------------------------------------------------------- #
# _persist_matches — maps RPC rows -> sighting_matches rows.
# --------------------------------------------------------------------------- #
class TestPersistMatches:
    def test_empty_matches_writes_nothing(self, fake_db):
        svc = SightingService(fake_db, ai_manager=None)
        svc._persist_matches("s1", [])
        assert fake_db.recorded == []

    def test_maps_id_and_similarity_to_columns(self, fake_db):
        svc = SightingService(fake_db, ai_manager=None)
        svc._persist_matches("s1", [
            {"id": "pet-a", "similarity": 0.91},
            {"id": "pet-b", "similarity": 0.42},
        ])
        rows = fake_db.payload_for("sighting_matches", "upsert")
        assert rows == [
            {"sighting_id": "s1", "missing_pet_id": "pet-a", "similarity_score": 0.91},
            {"sighting_id": "s1", "missing_pet_id": "pet-b", "similarity_score": 0.42},
        ]

    def test_drops_matches_without_an_id(self, fake_db):
        svc = SightingService(fake_db, ai_manager=None)
        svc._persist_matches("s1", [
            {"similarity": 0.9},               # no id -> skipped
            {"id": "pet-b", "similarity": 0.4},
        ])
        rows = fake_db.payload_for("sighting_matches", "upsert")
        assert rows == [
            {"sighting_id": "s1", "missing_pet_id": "pet-b", "similarity_score": 0.4},
        ]

    def test_all_rows_idless_writes_nothing(self, fake_db):
        svc = SightingService(fake_db, ai_manager=None)
        svc._persist_matches("s1", [{"similarity": 0.9}, {"foo": "bar"}])
        assert fake_db.recorded == []

    def test_missing_similarity_becomes_none(self, fake_db):
        svc = SightingService(fake_db, ai_manager=None)
        svc._persist_matches("s1", [{"id": "pet-a"}])
        rows = fake_db.payload_for("sighting_matches", "upsert")
        assert rows == [
            {"sighting_id": "s1", "missing_pet_id": "pet-a", "similarity_score": None},
        ]

    def test_upserts_on_unique_constraint(self, fake_db):
        svc = SightingService(fake_db, ai_manager=None)
        svc._persist_matches("s1", [{"id": "pet-a", "similarity": 0.9}])
        # idempotency contract: refresh-in-place, not duplicate rows
        _t, op, _payload, on_conflict = fake_db.recorded[-1]
        assert op == "upsert"
        assert on_conflict == "sighting_id,missing_pet_id"


# --------------------------------------------------------------------------- #
# get_matches — precondition guards + threshold filtering.
# --------------------------------------------------------------------------- #
class TestGetMatches:
    def _seed_valid_sighting(self, fake_db):
        fake_db.set_table_result("sightings", "select", data=[{
            "id": "s1",
            "feature_vector": [0.1, 0.2],
            "detected_species": "Dog",
            "sighted_location": "POINT(100.5 13.75)",
        }])

    def test_threshold_zero_returns_all_rpc_rows(self, fake_db):
        self._seed_valid_sighting(fake_db)
        fake_db.set_rpc_result("match_missing_pets", [
            {"id": "p1", "similarity": 0.9},
            {"id": "p2", "similarity": 0.3},
        ])
        svc = SightingService(fake_db, ai_manager=None)
        matches = run(svc.get_matches("s1", threshold=0.0))
        assert [m["id"] for m in matches] == ["p1", "p2"]

    def test_threshold_filters_below_cutoff(self, fake_db):
        self._seed_valid_sighting(fake_db)
        fake_db.set_rpc_result("match_missing_pets", [
            {"id": "p1", "similarity": 0.9},
            {"id": "p2", "similarity": 0.3},
        ])
        svc = SightingService(fake_db, ai_manager=None)
        matches = run(svc.get_matches("s1", threshold=0.5))
        assert [m["id"] for m in matches] == ["p1"]

    def test_threshold_keeps_value_exactly_at_cutoff(self, fake_db):
        # boundary: similarity == threshold must be retained (>=)
        self._seed_valid_sighting(fake_db)
        fake_db.set_rpc_result("match_missing_pets", [{"id": "p1", "similarity": 0.5}])
        svc = SightingService(fake_db, ai_manager=None)
        matches = run(svc.get_matches("s1", threshold=0.5))
        assert [m["id"] for m in matches] == ["p1"]

    def test_null_similarity_treated_as_zero_and_filtered(self, fake_db):
        self._seed_valid_sighting(fake_db)
        fake_db.set_rpc_result("match_missing_pets", [{"id": "p1", "similarity": None}])
        svc = SightingService(fake_db, ai_manager=None)
        matches = run(svc.get_matches("s1", threshold=0.1))
        assert matches == []

    def test_unknown_sighting_raises(self, fake_db):
        fake_db.set_table_result("sightings", "select", data=[])
        svc = SightingService(fake_db, ai_manager=None)
        with pytest.raises(ValueError):
            run(svc.get_matches("ghost"))

    @pytest.mark.parametrize("missing_field", [
        "feature_vector", "detected_species", "sighted_location",
    ])
    def test_incomplete_sighting_row_raises(self, fake_db, missing_field):
        row = {
            "id": "s1",
            "feature_vector": [0.1],
            "detected_species": "Dog",
            "sighted_location": "POINT(0 0)",
        }
        del row[missing_field]
        fake_db.set_table_result("sightings", "select", data=[row])
        svc = SightingService(fake_db, ai_manager=None)
        with pytest.raises(ValueError):
            run(svc.get_matches("s1"))


# --------------------------------------------------------------------------- #
# process_and_save_sighting — the hot path's headline invariants on a cache HIT.
# --------------------------------------------------------------------------- #
class TestProcessAndSaveCacheHit:
    def test_persists_user_species_and_cached_vector_not_yolo(self, fake_db):
        cached_vector = [0.1, 0.2, 0.3, 0.4]
        sighting = make_sighting(detected_species="Dog")
        # /analyze cached a CLIP vector AND a *different* YOLO guess (Cat).
        AnalyzeCache.set(str(sighting.image_url), {
            "species": "Cat",                 # YOLO's (wrong) guess
            "feature_vector": cached_vector,
        })
        wire_save_path(fake_db, cached_vector)
        ai = MagicMock()
        svc = SightingService(fake_db, ai_manager=ai)

        result = run(svc.process_and_save_sighting(sighting))

        insert_payload = fake_db.payload_for("sightings", "insert")
        # user-confirmed species wins over YOLO's cached guess
        assert insert_payload["detected_species"] == "Dog"
        # the pre-computed cached vector is reused verbatim
        assert insert_payload["feature_vector"] == cached_vector
        assert insert_payload["sighting_status"] == "Pending_Analysis"
        # cache hit must short-circuit the heavy pipeline entirely
        ai.download_image.assert_not_called()
        ai.run_yolo_seg.assert_not_called()
        ai.clip_encode.assert_not_called()

    def test_strips_feature_vector_from_response_and_returns_matches(self, fake_db):
        cached_vector = [0.5, 0.6]
        sighting = make_sighting()
        AnalyzeCache.set(str(sighting.image_url), {
            "species": "Dog", "feature_vector": cached_vector,
        })
        wire_save_path(fake_db, cached_vector)
        svc = SightingService(fake_db, ai_manager=MagicMock())

        result = run(svc.process_and_save_sighting(sighting))

        assert "feature_vector" not in result["sighting"]   # not leaked to client
        assert result["matches"] == [{"id": "pet-1", "similarity": 0.88}]

    # The following exercise the shared save-tail (persist + payload assembly)
    # via the cache-hit path; the behaviour is branch-agnostic.
    def test_persists_matches_into_sighting_matches(self, fake_db):
        # Returning the matches is not enough — they must also be written through
        # to sighting_matches so the owner timeline / F1 scoring can read them.
        cached_vector = [0.1, 0.2]
        sighting = make_sighting()
        AnalyzeCache.set(str(sighting.image_url), {
            "species": "Dog", "feature_vector": cached_vector,
        })
        wire_save_path(fake_db, cached_vector)  # rpc -> [{"id":"pet-1","similarity":0.88}]
        svc = SightingService(fake_db, ai_manager=MagicMock())

        run(svc.process_and_save_sighting(sighting))

        persisted = fake_db.payload_for("sighting_matches", "upsert")
        assert persisted == [
            {"sighting_id": "s1", "missing_pet_id": "pet-1", "similarity_score": 0.88},
        ]

    def test_target_pet_id_is_threaded_into_insert_payload(self, fake_db):
        cached_vector = [0.1, 0.2]
        sighting = make_sighting(target_pet_id="target-pet-1")
        AnalyzeCache.set(str(sighting.image_url), {
            "species": "Dog", "feature_vector": cached_vector,
        })
        wire_save_path(fake_db, cached_vector)
        svc = SightingService(fake_db, ai_manager=MagicMock())

        run(svc.process_and_save_sighting(sighting))

        insert_payload = fake_db.payload_for("sightings", "insert")
        assert insert_payload["initial_target_pet_id"] == "target-pet-1"

    def test_initial_target_pet_id_omitted_when_no_target(self, fake_db):
        # The NULL branch: a stray report (no target_pet_id) must NOT write an
        # initial_target_pet_id key at all (not even None).
        cached_vector = [0.1, 0.2]
        sighting = make_sighting()  # target_pet_id defaults to None
        AnalyzeCache.set(str(sighting.image_url), {
            "species": "Dog", "feature_vector": cached_vector,
        })
        wire_save_path(fake_db, cached_vector)
        svc = SightingService(fake_db, ai_manager=MagicMock())

        run(svc.process_and_save_sighting(sighting))

        insert_payload = fake_db.payload_for("sightings", "insert")
        assert "initial_target_pet_id" not in insert_payload

    def test_targeted_path_skips_vector_and_match(self, fake_db):
        # skip_matching (pet-detail targeted report): the hunter chose the pet
        # by eye, so the service must NOT pull/compute a vector, must NOT call
        # the match RPC, and must persist against initial_target_pet_id with
        # feature_vector left out of the INSERT. matches comes back empty.
        sighting = make_sighting(
            target_pet_id="target-pet-1", skip_matching=True,
        )
        fake_db.set_table_result("sightings", "insert", data=[{
            "id": "s1",
            "hunter_id": "hunter-1",
            "detected_species": "Dog",
            "sighted_location": "POINT(100.5018 13.7563)",
        }])
        # No AnalyzeCache entry and a MagicMock AI that would explode if touched
        # — proves the targeted path never reaches the AI pipeline.
        svc = SightingService(fake_db, ai_manager=MagicMock())

        result = run(svc.process_and_save_sighting(sighting))

        insert_payload = fake_db.payload_for("sightings", "insert")
        assert insert_payload["initial_target_pet_id"] == "target-pet-1"
        assert "feature_vector" not in insert_payload
        assert all(name != "match_missing_pets" for name, _ in fake_db.rpc_calls)
        assert result["matches"] == []


# --------------------------------------------------------------------------- #
# process_and_save_sighting — the cache-MISS branch (TTL expiry / worker restart)
# must transparently re-run the heavy pipeline. download_image / run_yolo_seg /
# clip_encode are awaited (AsyncMock); isolate_subject is synchronous (MagicMock).
# --------------------------------------------------------------------------- #
class TestProcessAndSaveCacheMiss:
    @staticmethod
    def _make_ai(*, recomputed_vector, isolate_return,
                 image="PIL_IMAGE", results="YOLO_RESULTS"):
        ai = MagicMock()
        ai.download_image = AsyncMock(return_value=image)
        ai.run_yolo_seg = AsyncMock(return_value=results)
        ai.isolate_subject = MagicMock(return_value=isolate_return)  # sync, not awaited
        ai.clip_encode = AsyncMock(return_value=recomputed_vector)
        return ai

    def test_miss_reruns_pipeline_and_inserts_recomputed_vector(self, fake_db):
        recomputed = [0.7, 0.8, 0.9]
        sighting = make_sighting(detected_species="Dog")
        # No AnalyzeCache.set -> the autouse fixture leaves the cache empty -> MISS.
        wire_save_path(fake_db, recomputed)
        ai = self._make_ai(
            recomputed_vector=recomputed,
            # YOLO re-detects a Cat; the user said Dog. The vector follows YOLO,
            # the stored species must follow the user.
            isolate_return=("ISOLATED_IMG", "Cat", 0.91, [0, 0, 10, 10]),
        )
        svc = SightingService(fake_db, ai_manager=ai)

        result = run(svc.process_and_save_sighting(sighting))

        # full pipeline actually ran, in order, on the real inputs
        ai.download_image.assert_awaited_once_with("https://img.example/sighting.jpg")
        ai.run_yolo_seg.assert_awaited_once_with("PIL_IMAGE")
        ai.clip_encode.assert_awaited_once_with("ISOLATED_IMG")  # the isolated frame, not the tuple

        insert_payload = fake_db.payload_for("sightings", "insert")
        assert insert_payload["feature_vector"] == recomputed       # re-encoded, not cached
        assert insert_payload["detected_species"] == "Dog"          # user wins over YOLO's "Cat"
        assert "feature_vector" not in result["sighting"]           # still stripped from response

    def test_miss_isolate_runs_without_species_constraint(self, fake_db):
        # The "forgiving re-run" contract: isolate_subject is called on the raw
        # YOLO results with NO expected_species, so the vector represents whatever
        # animal pixels YOLO finds regardless of the user's species choice.
        recomputed = [0.1, 0.2]
        sighting = make_sighting(detected_species="Bird")
        wire_save_path(fake_db, recomputed)
        ai = self._make_ai(
            recomputed_vector=recomputed,
            isolate_return=("ISOLATED_IMG", "Dog", 0.8, [1, 2, 3, 4]),
        )
        svc = SightingService(fake_db, ai_manager=ai)

        run(svc.process_and_save_sighting(sighting))

        ai.isolate_subject.assert_called_once_with("PIL_IMAGE", "YOLO_RESULTS")

    def test_miss_yolo_finds_nothing_raises_and_skips_insert(self, fake_db):
        # SRS-30: when no cat/dog/bird is detected the server refuses to save a
        # sighting (the UI surfaces "No target animal detected, please try again").
        # isolate_subject returns None (YOLO miss) -> ValueError, no row written,
        # and CLIP must not be invoked on a non-existent subject.
        sighting = make_sighting()
        ai = self._make_ai(recomputed_vector=[0.0], isolate_return=None)
        svc = SightingService(fake_db, ai_manager=ai)

        with pytest.raises(ValueError, match="No target animal detected"):
            run(svc.process_and_save_sighting(sighting))

        ai.clip_encode.assert_not_awaited()
        assert fake_db.payload_for("sightings", "insert") is None


# --------------------------------------------------------------------------- #
# analyze_sighting_image (UTC-15, SRS-30/31) — the heavy first step. We test its
# COORDINATION logic with a mocked AI manager + the real AnalyzeCache: the
# success path returns the verify-screen payload and caches the full result; a
# YOLO miss returns not_found without caching or encoding; a pipeline error
# re-raises without caching. (The real YOLO/CLIP detection accuracy is manual.)
# --------------------------------------------------------------------------- #
class TestAnalyzeSightingImage:
    @staticmethod
    def _make_ai(*, isolate_return, vector=None, download_exc=None,
                 image="PIL_IMAGE", results="YOLO_RESULTS"):
        ai = MagicMock()
        if download_exc is not None:
            ai.download_image = AsyncMock(side_effect=download_exc)
        else:
            ai.download_image = AsyncMock(return_value=image)
        ai.run_yolo_seg = AsyncMock(return_value=results)
        ai.isolate_subject = MagicMock(return_value=isolate_return)  # sync
        ai.clip_encode = AsyncMock(return_value=vector)
        return ai

    def test_success_caches_and_returns_payload(self, fake_db):
        url = "https://img.example/analyze.jpg"
        bbox = [1.0, 2.0, 3.0, 4.0]
        ai = self._make_ai(
            isolate_return=("ISOLATED_IMG", "Dog", 0.8825, bbox),
            vector=[0.11, 0.22],
        )
        svc = SightingService(fake_db, ai_manager=ai)

        out = run(svc.analyze_sighting_image(url))

        assert out["status"] == "success"
        assert out["data"]["species"] == "Dog"
        assert out["data"]["confidence"] == 88.25      # round(0.8825 * 100, 2)
        assert out["data"]["bbox"] == bbox

        cached = AnalyzeCache.get(url)
        assert cached is not None
        assert cached["feature_vector"] == [0.11, 0.22]
        assert cached["confidence"] == 0.8825          # raw, not the percentage
        assert cached["species"] == "Dog"
        assert cached["pil_image"] == "PIL_IMAGE"
        assert cached["isolated_image"] == "ISOLATED_IMG"

    def test_no_target_animal_returns_not_found_without_caching(self, fake_db):
        url = "https://img.example/empty-scene.jpg"
        ai = self._make_ai(isolate_return=None)
        svc = SightingService(fake_db, ai_manager=ai)

        out = run(svc.analyze_sighting_image(url))

        assert out["status"] == "not_found"
        assert AnalyzeCache.get(url) is None
        ai.clip_encode.assert_not_awaited()

    def test_pipeline_error_reraises_without_caching(self, fake_db):
        url = "https://img.example/broken.jpg"
        ai = self._make_ai(
            isolate_return=None, download_exc=RuntimeError("download failed"),
        )
        svc = SightingService(fake_db, ai_manager=ai)

        with pytest.raises(RuntimeError):
            run(svc.analyze_sighting_image(url))
        assert AnalyzeCache.get(url) is None


# --------------------------------------------------------------------------- #
# get_sighting_by_id (UTC-19) — single read with the internal feature vector
# stripped; an unknown id returns None (not an error); a DB error re-raises.
# --------------------------------------------------------------------------- #
class _RaisingSelectDB:
    """Supabase double whose table(...).select(...).eq(...).execute() raises."""
    def table(self, _name):
        return self

    def select(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    def execute(self):
        raise RuntimeError("DB connection lost")


class TestGetSightingById:
    def test_returns_row_with_vector_stripped(self, fake_db):
        fake_db.set_table_result("sightings", "select", data=[{
            "id": "s1",
            "detected_species": "Dog",
            "feature_vector": [0.1, 0.2, 0.3],
        }])
        svc = SightingService(fake_db, ai_manager=MagicMock())

        row = run(svc.get_sighting_by_id("s1"))

        assert row["id"] == "s1"
        assert "feature_vector" not in row

    def test_unknown_id_returns_none(self, fake_db):
        # default fake select result is data=[] -> None
        svc = SightingService(fake_db, ai_manager=MagicMock())
        assert run(svc.get_sighting_by_id("ghost")) is None

    def test_db_error_is_reraised(self):
        svc = SightingService(_RaisingSelectDB(), ai_manager=MagicMock())
        with pytest.raises(RuntimeError):
            run(svc.get_sighting_by_id("s1"))
