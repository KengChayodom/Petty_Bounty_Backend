"""
Unit tests for app/services/sighting_service.py.

Scope: the deterministic business logic — status validation, match-row
mapping, similarity-threshold filtering, the get_matches precondition guards,
and the headline invariant that the *user-confirmed* species (not YOLO's
guess) and the *cached* vector are what get persisted on the hot path.

Boundary rule (per db-testing-seams): the DB is reached only through the
SightingRepository port owned by this codebase, so we double THAT with
MagicMock(spec=...) — never a hand-rolled Supabase client. The AI manager is a
MagicMock that the cache-hit path must never touch. DB-engine semantics (e.g.
the sighting_matches on-conflict/idempotency contract) are NOT asserted here —
they belong to the adapter integration suite (TEST_PLAN §4 #5).

Progress-I SRS traceability:
  * SRS-30 — the server refuses to save when no cat/dog/bird is detected
    (TestProcessAndSaveCacheMiss::test_miss_yolo_finds_nothing_raises_and_skips_insert).
  * SRS-31 — the sighting save persists the USER-confirmed species + the cached
    vector, returns the ranked matches (with their cosine similarity scores,
    SRS-35), and links candidate pets via sighting_matches
    (TestProcessAndSaveCacheHit::{test_persists_user_species_and_cached_vector_not_yolo,
    test_persists_matches_into_sighting_matches}, TestGetMatches, TestPersistMatches).
  * SRS-50 — the TARGETED flow: the hunter reports one known pet straight to its
    owner (pet-detail "Report Sighting"). It stores initial_target_pet_id, skips
    the CLIP vector + match RPC, and goes through the dedicated
    save_targeted_sighting method (its own endpoint POST /sightings/targeted),
    NOT a skip_matching flag (TestSaveTargetedSighting).
  * SRS-43 — the species-correction dropdown ("No" -> pick correct species) is a
    DISCOVERY re-submit: the client calls createSightingWithMatch, which goes
    through process_and_save_sighting (matching runs again with the corrected
    species). It has no target pet, so it is distinct from SRS-50
    (TestProcessAndSaveCacheHit::test_persists_user_species_and_cached_vector_not_yolo).
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.repositories.sighting_repository import (
    SightingNotSaved,
    SightingRepository,
)
from app.schemas.sightings import SightingCreate, TargetedSightingCreate
from app.services.ai_cache import AnalyzeCache
from app.services.sighting_service import SightingService

POINT = "POINT(100.5018 13.7563)"


def run(coro):
    """Drive an async coroutine without pulling in pytest-asyncio."""
    return asyncio.run(coro)


def _repo():
    return MagicMock(spec=SightingRepository)


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


def make_targeted_sighting(**overrides):
    data = {
        "hunter_id": "hunter-1",
        "image_url": "https://img.example/sighting.jpg",
        "latitude": 13.7563,
        "longitude": 100.5018,
        "detected_species": "Dog",
        "action_type": "Spotted",
        "target_pet_id": "target-pet-1",
    }
    data.update(overrides)
    return TargetedSightingCreate(**data)


def wire_save_path(repo, stored_vector, similarity=0.88):
    """Program the repo responses the post-vector tail of process_and_save needs:
    the INSERT echo (feature_vector included, as Supabase stores it — text — so
    the strip-from-response behaviour is exercised), the get_matches re-read, and
    the match RPC result."""
    repo.insert_sighting.return_value = {
        "id": "s1",
        "hunter_id": "hunter-1",
        "detected_species": "Dog",
        "feature_vector": str(stored_vector),
        "sighted_location": POINT,
    }
    repo.get_sighting_for_match.return_value = {
        "id": "s1",
        "feature_vector": stored_vector,
        "detected_species": "Dog",
        "sighted_location": POINT,
    }
    repo.match_missing_pets.return_value = [{"id": "pet-1", "similarity": similarity}]


# --------------------------------------------------------------------------- #
# _persist_matches — maps RPC rows -> sighting_matches rows via the repo upsert.
# (The on-conflict/idempotency contract itself is a DB concern: adapter suite.)
# --------------------------------------------------------------------------- #
class TestPersistMatches:
    def test_empty_matches_writes_nothing(self):
        repo = _repo()
        svc = SightingService(repo, ai_manager=None)
        svc._persist_matches("s1", [])
        repo.upsert_sighting_matches.assert_not_called()

    def test_maps_id_and_similarity_to_columns(self):
        repo = _repo()
        svc = SightingService(repo, ai_manager=None)
        svc._persist_matches("s1", [
            {"id": "pet-a", "similarity": 0.91},
            {"id": "pet-b", "similarity": 0.42},
        ])
        repo.upsert_sighting_matches.assert_called_once_with([
            {"sighting_id": "s1", "missing_pet_id": "pet-a", "similarity_score": 0.91},
            {"sighting_id": "s1", "missing_pet_id": "pet-b", "similarity_score": 0.42},
        ])

    def test_drops_matches_without_an_id(self):
        repo = _repo()
        svc = SightingService(repo, ai_manager=None)
        svc._persist_matches("s1", [
            {"similarity": 0.9},               # no id -> skipped
            {"id": "pet-b", "similarity": 0.4},
        ])
        repo.upsert_sighting_matches.assert_called_once_with([
            {"sighting_id": "s1", "missing_pet_id": "pet-b", "similarity_score": 0.4},
        ])

    def test_all_rows_idless_writes_nothing(self):
        repo = _repo()
        svc = SightingService(repo, ai_manager=None)
        svc._persist_matches("s1", [{"similarity": 0.9}, {"foo": "bar"}])
        repo.upsert_sighting_matches.assert_not_called()

    def test_missing_similarity_becomes_none(self):
        repo = _repo()
        svc = SightingService(repo, ai_manager=None)
        svc._persist_matches("s1", [{"id": "pet-a"}])
        repo.upsert_sighting_matches.assert_called_once_with([
            {"sighting_id": "s1", "missing_pet_id": "pet-a", "similarity_score": None},
        ])


# --------------------------------------------------------------------------- #
# get_matches — precondition guards + threshold filtering.
# --------------------------------------------------------------------------- #
class TestGetMatches:
    def _valid_sighting(self, repo):
        repo.get_sighting_for_match.return_value = {
            "id": "s1",
            "feature_vector": [0.1, 0.2],
            "detected_species": "Dog",
            "sighted_location": "POINT(100.5 13.75)",
        }

    def test_threshold_zero_returns_all_rpc_rows(self):
        repo = _repo()
        self._valid_sighting(repo)
        repo.match_missing_pets.return_value = [
            {"id": "p1", "similarity": 0.9},
            {"id": "p2", "similarity": 0.3},
        ]
        svc = SightingService(repo, ai_manager=None)
        matches = run(svc.get_matches("s1", threshold=0.0))
        assert [m["id"] for m in matches] == ["p1", "p2"]

    def test_threshold_filters_below_cutoff(self):
        repo = _repo()
        self._valid_sighting(repo)
        repo.match_missing_pets.return_value = [
            {"id": "p1", "similarity": 0.9},
            {"id": "p2", "similarity": 0.3},
        ]
        svc = SightingService(repo, ai_manager=None)
        matches = run(svc.get_matches("s1", threshold=0.5))
        assert [m["id"] for m in matches] == ["p1"]

    def test_threshold_keeps_value_exactly_at_cutoff(self):
        # boundary: similarity == threshold must be retained (>=)
        repo = _repo()
        self._valid_sighting(repo)
        repo.match_missing_pets.return_value = [{"id": "p1", "similarity": 0.5}]
        svc = SightingService(repo, ai_manager=None)
        matches = run(svc.get_matches("s1", threshold=0.5))
        assert [m["id"] for m in matches] == ["p1"]

    def test_null_similarity_treated_as_zero_and_filtered(self):
        repo = _repo()
        self._valid_sighting(repo)
        repo.match_missing_pets.return_value = [{"id": "p1", "similarity": None}]
        svc = SightingService(repo, ai_manager=None)
        matches = run(svc.get_matches("s1", threshold=0.1))
        assert matches == []

    def test_unknown_sighting_raises(self):
        repo = _repo()
        repo.get_sighting_for_match.return_value = None
        svc = SightingService(repo, ai_manager=None)
        with pytest.raises(ValueError):
            run(svc.get_matches("ghost"))

    @pytest.mark.parametrize("missing_field", [
        "feature_vector", "detected_species", "sighted_location",
    ])
    def test_incomplete_sighting_row_raises(self, missing_field):
        row = {
            "id": "s1",
            "feature_vector": [0.1],
            "detected_species": "Dog",
            "sighted_location": "POINT(0 0)",
        }
        del row[missing_field]
        repo = _repo()
        repo.get_sighting_for_match.return_value = row
        svc = SightingService(repo, ai_manager=None)
        with pytest.raises(ValueError):
            run(svc.get_matches("s1"))


# --------------------------------------------------------------------------- #
# process_and_save_sighting — the hot path's headline invariants on a cache HIT.
# --------------------------------------------------------------------------- #
class TestProcessAndSaveCacheHit:
    def test_persists_user_species_and_cached_vector_not_yolo(self):
        cached_vector = [0.1, 0.2, 0.3, 0.4]
        sighting = make_sighting(detected_species="Dog")
        # /analyze cached a CLIP vector AND a *different* YOLO guess (Cat).
        AnalyzeCache.set(str(sighting.image_url), {
            "species": "Cat",                 # YOLO's (wrong) guess
            "feature_vector": cached_vector,
        })
        repo = _repo()
        wire_save_path(repo, cached_vector)
        ai = MagicMock()
        svc = SightingService(repo, ai_manager=ai)

        run(svc.process_and_save_sighting(sighting))

        insert_payload = repo.insert_sighting.call_args.args[0]
        # user-confirmed species wins over YOLO's cached guess
        assert insert_payload["detected_species"] == "Dog"
        # the pre-computed cached vector is reused verbatim
        assert insert_payload["feature_vector"] == cached_vector
        assert insert_payload["sighting_status"] == "Pending_Analysis"
        # cache hit must short-circuit the heavy pipeline entirely
        ai.download_image.assert_not_called()
        ai.run_yolo_seg.assert_not_called()
        ai.clip_encode.assert_not_called()

    def test_strips_feature_vector_from_response_and_returns_matches(self):
        cached_vector = [0.5, 0.6]
        sighting = make_sighting()
        AnalyzeCache.set(str(sighting.image_url), {
            "species": "Dog", "feature_vector": cached_vector,
        })
        repo = _repo()
        wire_save_path(repo, cached_vector)
        svc = SightingService(repo, ai_manager=MagicMock())

        result = run(svc.process_and_save_sighting(sighting))

        assert "feature_vector" not in result["sighting"]   # not leaked to client
        assert result["matches"] == [{"id": "pet-1", "similarity": 0.88}]

    # The following exercise the shared save-tail (persist + payload assembly)
    # via the cache-hit path; the behaviour is branch-agnostic.
    def test_persists_matches_into_sighting_matches(self):
        # Returning the matches is not enough — they must also be written through
        # to sighting_matches so the owner timeline / F1 scoring can read them.
        cached_vector = [0.1, 0.2]
        sighting = make_sighting()
        AnalyzeCache.set(str(sighting.image_url), {
            "species": "Dog", "feature_vector": cached_vector,
        })
        repo = _repo()
        wire_save_path(repo, cached_vector)  # match -> [{"id":"pet-1","similarity":0.88}]
        svc = SightingService(repo, ai_manager=MagicMock())

        run(svc.process_and_save_sighting(sighting))

        repo.upsert_sighting_matches.assert_called_once_with([
            {"sighting_id": "s1", "missing_pet_id": "pet-1", "similarity_score": 0.88},
        ])

    def test_discovery_leaves_initial_target_pet_id_unset(self):
        # Discovery never targets a specific pet, so the INSERT must NOT write
        # an initial_target_pet_id key at all (that column is targeted-only).
        cached_vector = [0.1, 0.2]
        sighting = make_sighting()
        AnalyzeCache.set(str(sighting.image_url), {
            "species": "Dog", "feature_vector": cached_vector,
        })
        repo = _repo()
        wire_save_path(repo, cached_vector)
        svc = SightingService(repo, ai_manager=MagicMock())

        run(svc.process_and_save_sighting(sighting))

        assert "initial_target_pet_id" not in repo.insert_sighting.call_args.args[0]


# --------------------------------------------------------------------------- #
# save_targeted_sighting — the pet-detail TARGETED path (UTC-28, MD-33). The
# hunter reports ONE known pet straight to its owner: no CLIP vector, no match
# RPC, initial_target_pet_id set, matches always empty. Its own method + its
# own endpoint (POST /sightings/targeted), NOT a skip_matching flag.
# --------------------------------------------------------------------------- #
class TestSaveTargetedSighting:
    def test_persists_target_pet_id_and_omits_vector(self):
        sighting = make_targeted_sighting(target_pet_id="target-pet-1")
        repo = _repo()
        repo.insert_sighting.return_value = {
            "id": "s1",
            "hunter_id": "hunter-1",
            "detected_species": "Dog",
            "sighted_location": POINT,
        }
        svc = SightingService(repo, ai_manager=MagicMock())

        run(svc.save_targeted_sighting(sighting))

        insert_payload = repo.insert_sighting.call_args.args[0]
        assert insert_payload["initial_target_pet_id"] == "target-pet-1"
        assert "feature_vector" not in insert_payload

    def test_skips_match_rpc_and_ai_pipeline_returns_empty_matches(self):
        # The hunter chose the pet by eye — no AnalyzeCache entry, and a
        # MagicMock AI that would explode if any pipeline call were made.
        sighting = make_targeted_sighting()
        repo = _repo()
        repo.insert_sighting.return_value = {
            "id": "s1",
            "hunter_id": "hunter-1",
            "detected_species": "Dog",
            "sighted_location": POINT,
        }
        ai = MagicMock()
        svc = SightingService(repo, ai_manager=ai)

        result = run(svc.save_targeted_sighting(sighting))

        # no similarity search of any kind
        repo.get_sighting_for_match.assert_not_called()
        repo.match_missing_pets.assert_not_called()
        assert result["matches"] == []
        # never reaches the AI pipeline
        ai.download_image.assert_not_called()
        ai.run_yolo_seg.assert_not_called()
        ai.clip_encode.assert_not_called()
        # feature_vector stripped from the returned row
        assert "feature_vector" not in result["sighting"]

    def test_insert_valueerror_propagates(self):
        # An empty INSERT (SightingNotSaved, a ValueError) propagates unchanged.
        sighting = make_targeted_sighting()
        repo = _repo()
        repo.insert_sighting.side_effect = SightingNotSaved({})
        svc = SightingService(repo, ai_manager=MagicMock())
        with pytest.raises(ValueError):
            run(svc.save_targeted_sighting(sighting))

    def test_unexpected_insert_error_propagates(self):
        # A non-ValueError (transport) hits the generic handler and re-raises.
        sighting = make_targeted_sighting()
        repo = _repo()
        repo.insert_sighting.side_effect = RuntimeError("DB connection lost")
        svc = SightingService(repo, ai_manager=MagicMock())
        with pytest.raises(RuntimeError):
            run(svc.save_targeted_sighting(sighting))


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

    def test_miss_reruns_pipeline_and_inserts_recomputed_vector(self):
        recomputed = [0.7, 0.8, 0.9]
        sighting = make_sighting(detected_species="Dog")
        # No AnalyzeCache.set -> the autouse fixture leaves the cache empty -> MISS.
        repo = _repo()
        wire_save_path(repo, recomputed)
        ai = self._make_ai(
            recomputed_vector=recomputed,
            # YOLO re-detects a Cat; the user said Dog. The vector follows YOLO,
            # the stored species must follow the user.
            isolate_return=("ISOLATED_IMG", "Cat", 0.91, [0, 0, 10, 10]),
        )
        svc = SightingService(repo, ai_manager=ai)

        result = run(svc.process_and_save_sighting(sighting))

        # full pipeline actually ran, in order, on the real inputs
        ai.download_image.assert_awaited_once_with("https://img.example/sighting.jpg")
        ai.run_yolo_seg.assert_awaited_once_with("PIL_IMAGE")
        ai.clip_encode.assert_awaited_once_with("ISOLATED_IMG")  # the isolated frame, not the tuple

        insert_payload = repo.insert_sighting.call_args.args[0]
        assert insert_payload["feature_vector"] == recomputed       # re-encoded, not cached
        assert insert_payload["detected_species"] == "Dog"          # user wins over YOLO's "Cat"
        assert "feature_vector" not in result["sighting"]           # still stripped from response

    def test_miss_isolate_runs_without_species_constraint(self):
        # The "forgiving re-run" contract: isolate_subject is called on the raw
        # YOLO results with NO expected_species, so the vector represents whatever
        # animal pixels YOLO finds regardless of the user's species choice.
        recomputed = [0.1, 0.2]
        sighting = make_sighting(detected_species="Bird")
        repo = _repo()
        wire_save_path(repo, recomputed)
        ai = self._make_ai(
            recomputed_vector=recomputed,
            isolate_return=("ISOLATED_IMG", "Dog", 0.8, [1, 2, 3, 4]),
        )
        svc = SightingService(repo, ai_manager=ai)

        run(svc.process_and_save_sighting(sighting))

        ai.isolate_subject.assert_called_once_with("PIL_IMAGE", "YOLO_RESULTS")

    def test_miss_yolo_finds_nothing_raises_and_skips_insert(self):
        # SRS-30: when no cat/dog/bird is detected the server refuses to save a
        # sighting (the UI surfaces "No target animal detected, please try again").
        # isolate_subject returns None (YOLO miss) -> ValueError, no row written,
        # and CLIP must not be invoked on a non-existent subject.
        sighting = make_sighting()
        repo = _repo()
        ai = self._make_ai(recomputed_vector=[0.0], isolate_return=None)
        svc = SightingService(repo, ai_manager=ai)

        with pytest.raises(ValueError, match="No target animal detected"):
            run(svc.process_and_save_sighting(sighting))

        ai.clip_encode.assert_not_awaited()
        repo.insert_sighting.assert_not_called()


# --------------------------------------------------------------------------- #
# process_and_save_sighting — the error/edge FRAMES (Category-Partition):
# the row is saved but the downstream save-tail degrades gracefully. These are
# the swallow-and-continue branches the happy-path tests never exercise.
#   A. match RPC raises        -> matches=[] (row still saved)   [swallowed]
#   B. match RPC returns []     -> `if matches:` False, no persist
#   C. persist raises           -> matches still returned         [swallowed]
#   D. insert returns no row    -> SightingNotSaved propagates    [error]
# All run on the cache-HIT branch so the AI pipeline is out of the picture.
# --------------------------------------------------------------------------- #
class TestProcessAndSaveErrorFrames:
    @staticmethod
    def _cache_hit(sighting, vector):
        AnalyzeCache.set(str(sighting.image_url), {
            "species": "Dog", "feature_vector": vector,
        })

    def test_match_lookup_failure_is_swallowed_and_row_still_saved(self):
        # Frame A: get_matches raises (the RPC is down). The sighting row is
        # already saved, so we swallow, return matches=[], and never persist.
        cached_vector = [0.1, 0.2]
        sighting = make_sighting()
        self._cache_hit(sighting, cached_vector)
        repo = _repo()
        repo.insert_sighting.return_value = {
            "id": "s1", "detected_species": "Dog",
            "feature_vector": str(cached_vector),
        }
        repo.get_sighting_for_match.return_value = {
            "id": "s1", "feature_vector": cached_vector,
            "detected_species": "Dog", "sighted_location": POINT,
        }
        repo.match_missing_pets.side_effect = RuntimeError("match RPC down")
        svc = SightingService(repo, ai_manager=MagicMock())

        result = run(svc.process_and_save_sighting(sighting))

        assert result["sighting"]["id"] == "s1"          # row still saved
        assert result["matches"] == []                   # failure swallowed
        repo.upsert_sighting_matches.assert_not_called()  # nothing to persist

    def test_no_matches_skips_persist(self):
        # Frame B: matching succeeds but finds nothing -> `if matches:` is False,
        # so the persist block is skipped entirely.
        cached_vector = [0.1, 0.2]
        sighting = make_sighting()
        self._cache_hit(sighting, cached_vector)
        repo = _repo()
        repo.insert_sighting.return_value = {
            "id": "s1", "detected_species": "Dog",
            "feature_vector": str(cached_vector),
        }
        repo.get_sighting_for_match.return_value = {
            "id": "s1", "feature_vector": cached_vector,
            "detected_species": "Dog", "sighted_location": POINT,
        }
        repo.match_missing_pets.return_value = []   # no candidates
        svc = SightingService(repo, ai_manager=MagicMock())

        result = run(svc.process_and_save_sighting(sighting))

        assert result["matches"] == []
        repo.upsert_sighting_matches.assert_not_called()

    def test_persist_failure_is_swallowed_and_matches_still_returned(self):
        # Frame C: the sighting_matches upsert blows up (e.g. missing UNIQUE).
        # It MUST NOT clobber the matches already fetched for the verify screen,
        # nor the saved row — the failure is logged (ERROR) and swallowed.
        cached_vector = [0.1, 0.2]
        sighting = make_sighting()
        self._cache_hit(sighting, cached_vector)
        repo = _repo()
        wire_save_path(repo, cached_vector)   # match -> [{"id":"pet-1","similarity":0.88}]
        repo.upsert_sighting_matches.side_effect = RuntimeError("persist failed")
        svc = SightingService(repo, ai_manager=MagicMock())

        result = run(svc.process_and_save_sighting(sighting))

        assert result["matches"] == [{"id": "pet-1", "similarity": 0.88}]  # unaffected
        assert result["sighting"]["id"] == "s1"                            # still saved
        repo.upsert_sighting_matches.assert_called_once()                  # it was attempted

    def test_insert_returning_no_row_raises(self):
        # Frame D: the INSERT comes back empty -> the adapter raises
        # SightingNotSaved (a ValueError) which propagates; matching is never
        # reached and nothing is persisted.
        cached_vector = [0.1, 0.2]
        sighting = make_sighting()
        self._cache_hit(sighting, cached_vector)
        repo = _repo()
        repo.insert_sighting.side_effect = SightingNotSaved({})
        svc = SightingService(repo, ai_manager=MagicMock())

        with pytest.raises(ValueError):
            run(svc.process_and_save_sighting(sighting))

        repo.match_missing_pets.assert_not_called()
        repo.upsert_sighting_matches.assert_not_called()

    def test_unexpected_insert_error_propagates_after_logging(self):
        # Frame D': a NON-ValueError from the write (transport/connection) is
        # not swallowed — it hits the outer generic handler, is logged, and
        # re-raised (distinct from the ValueError re-raise branch above).
        cached_vector = [0.1, 0.2]
        sighting = make_sighting()
        self._cache_hit(sighting, cached_vector)
        repo = _repo()
        repo.insert_sighting.side_effect = RuntimeError("DB connection lost")
        svc = SightingService(repo, ai_manager=MagicMock())

        with pytest.raises(RuntimeError):
            run(svc.process_and_save_sighting(sighting))


# --------------------------------------------------------------------------- #
# analyze_sighting_image (UTC-15, SRS-29/30) — the heavy first step. We test its
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

    def test_success_caches_and_returns_payload(self):
        url = "https://img.example/analyze.jpg"
        bbox = [1.0, 2.0, 3.0, 4.0]
        ai = self._make_ai(
            isolate_return=("ISOLATED_IMG", "Dog", 0.8825, bbox),
            vector=[0.11, 0.22],
        )
        svc = SightingService(_repo(), ai_manager=ai)

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

    def test_no_target_animal_returns_not_found_without_caching(self):
        url = "https://img.example/empty-scene.jpg"
        ai = self._make_ai(isolate_return=None)
        svc = SightingService(_repo(), ai_manager=ai)

        out = run(svc.analyze_sighting_image(url))

        assert out["status"] == "not_found"
        assert AnalyzeCache.get(url) is None
        ai.clip_encode.assert_not_awaited()

    def test_pipeline_error_reraises_without_caching(self):
        url = "https://img.example/broken.jpg"
        ai = self._make_ai(
            isolate_return=None, download_exc=RuntimeError("download failed"),
        )
        svc = SightingService(_repo(), ai_manager=ai)

        with pytest.raises(RuntimeError):
            run(svc.analyze_sighting_image(url))
        assert AnalyzeCache.get(url) is None


# --------------------------------------------------------------------------- #
# get_sighting_by_id (UTC-19) — single read with the internal feature vector
# stripped; an unknown id returns None (not an error); a DB error re-raises.
# --------------------------------------------------------------------------- #
class TestGetSightingById:
    def test_returns_row_with_vector_stripped(self):
        repo = _repo()
        repo.get_sighting.return_value = {
            "id": "s1",
            "detected_species": "Dog",
            "feature_vector": [0.1, 0.2, 0.3],
        }
        svc = SightingService(repo, ai_manager=MagicMock())

        row = run(svc.get_sighting_by_id("s1"))

        assert row["id"] == "s1"
        assert "feature_vector" not in row

    def test_unknown_id_returns_none(self):
        repo = _repo()
        repo.get_sighting.return_value = None
        svc = SightingService(repo, ai_manager=MagicMock())
        assert run(svc.get_sighting_by_id("ghost")) is None

    def test_db_error_is_reraised(self):
        repo = _repo()
        repo.get_sighting.side_effect = RuntimeError("DB connection lost")
        svc = SightingService(repo, ai_manager=MagicMock())
        with pytest.raises(RuntimeError):
            run(svc.get_sighting_by_id("s1"))


# --------------------------------------------------------------------------- #
# get_hunter_activity — three repo round-trips + the pure assemble step
# (sighting_logic.assemble_hunter_activity). Category-Partition:
#   * sightings list: empty (short-circuit) / non-empty
#   * per sighting: has matches / has none (assemble default branch)
#   * awards: attached by sighting_id / with a NULL sighting_id (excluded)
#   * any repo call raises -> propagates
# --------------------------------------------------------------------------- #
class TestGetHunterActivity:
    def test_empty_sightings_short_circuits(self):
        repo = _repo()
        repo.count_sightings_for_hunter.return_value = 0
        repo.list_sightings_for_hunter.return_value = []
        svc = SightingService(repo, ai_manager=None)

        out = run(svc.get_hunter_activity("hunter-1"))

        assert out == {"sightings": [], "total_count": 0}
        # no reason to fetch matches/awards when there are no sightings
        repo.get_matches_for_sightings.assert_not_called()
        repo.get_awards_for_hunter.assert_not_called()

    def test_attaches_matches_and_awards_per_sighting(self):
        repo = _repo()
        repo.count_sightings_for_hunter.return_value = 2
        repo.list_sightings_for_hunter.return_value = [{"id": "s1"}, {"id": "s2"}]
        repo.get_matches_for_sightings.return_value = [
            {"sighting_id": "s1", "missing_pet_id": "p1",
             "similarity_score": 0.9, "owner_status": None},
        ]
        repo.get_awards_for_hunter.return_value = [
            {"sighting_id": "s1", "points": 25},   # attaches to s1
            {"sighting_id": None, "points": 5},     # NULL sighting_id -> excluded
        ]
        svc = SightingService(repo, ai_manager=None)

        out = run(svc.get_hunter_activity("hunter-1"))

        assert out["total_count"] == 2
        s1, s2 = out["sightings"]
        assert s1["matches"] == [{
            "sighting_id": "s1", "missing_pet_id": "p1",
            "similarity_score": 0.9, "owner_status": None,
        }]
        assert s1["score_award"] == {"sighting_id": "s1", "points": 25}
        assert s2["matches"] == []          # no matches -> default []
        assert s2["score_award"] is None    # no award for s2

    def test_repo_error_is_reraised(self):
        repo = _repo()
        repo.count_sightings_for_hunter.side_effect = RuntimeError("db down")
        svc = SightingService(repo, ai_manager=None)
        with pytest.raises(RuntimeError):
            run(svc.get_hunter_activity("hunter-1"))


# --------------------------------------------------------------------------- #
# get_hunter_stats — total_score defaults to 0 when the user row is absent.
# --------------------------------------------------------------------------- #
class TestGetHunterStats:
    def test_returns_all_counts_with_user_score(self):
        repo = _repo()
        repo.get_user.return_value = {"total_score": 42}
        repo.count_sightings_for_hunter.return_value = 5
        repo.count_verified_sightings_for_hunter.return_value = 3
        repo.count_contributions_for_hunter.return_value = 2
        svc = SightingService(repo, ai_manager=None)

        out = run(svc.get_hunter_stats("hunter-1"))

        assert out == {
            "total_score": 42,
            "sightings_submitted": 5,
            "sightings_verified": 3,
            "resolutions_contributed_to": 2,
        }

    def test_defaults_score_to_zero_when_no_user_row(self):
        repo = _repo()
        repo.get_user.return_value = None   # no profile row
        repo.count_sightings_for_hunter.return_value = 0
        repo.count_verified_sightings_for_hunter.return_value = 0
        repo.count_contributions_for_hunter.return_value = 0
        svc = SightingService(repo, ai_manager=None)

        assert run(svc.get_hunter_stats("hunter-1"))["total_score"] == 0

    def test_repo_error_is_reraised(self):
        repo = _repo()
        repo.get_user.side_effect = RuntimeError("db down")
        svc = SightingService(repo, ai_manager=None)
        with pytest.raises(RuntimeError):
            run(svc.get_hunter_stats("hunter-1"))


# --------------------------------------------------------------------------- #
# decide_match — the owner's verdict on one AI match (2026-08-17).
#
# The reply half of the loop. Before it existed `sighting_matches.owner_status`
# had no writer at all: the owner could read their timeline but not answer it,
# so a wrong match kept counting as a real sighting of their pet forever.
#
# What each test protects:
#   * ownership — a stranger must not be able to rule on (or learn about)
#     someone else's pet, and must not get a different answer than "not found";
#   * the asymmetry — Confirmed advances the sighting, Rejected does not,
#     because one photo can match several pets;
#   * the verdict surviving a failure in the secondary write.
# --------------------------------------------------------------------------- #
class TestDecideMatch:
    @staticmethod
    def _svc(owner="owner-1", updated={"sighting_id": "s1", "owner_status": "X"}):
        repo = _repo()
        repo.get_pet_owners.return_value = {"p1": owner} if owner else {}
        repo.update_match_owner_status.return_value = updated
        return SightingService(repo, ai_manager=None), repo

    def test_confirm_writes_both_the_verdict_and_the_sighting_status(self):
        svc, repo = self._svc()

        out = run(svc.decide_match("p1", "s1", "owner-1", "Confirmed"))

        repo.update_match_owner_status.assert_called_once_with(
            "s1", "p1", "Confirmed"
        )
        repo.set_sighting_status.assert_called_once_with("s1", "Confirmed")
        assert out["sighting_status_updated"] is True

    def test_reject_writes_the_verdict_only(self):
        """A sighting can match several pets. This owner saying "not mine" is
        no statement about anybody else's pet, so the sighting is untouched."""
        svc, repo = self._svc()

        out = run(svc.decide_match("p1", "s1", "owner-1", "Rejected"))

        repo.update_match_owner_status.assert_called_once_with(
            "s1", "p1", "Rejected"
        )
        repo.set_sighting_status.assert_not_called()
        assert out["sighting_status_updated"] is False

    def test_pet_belonging_to_someone_else_is_not_found(self):
        """404, not 403 — a non-owner must not even learn the pet exists. And
        nothing may be written on the way to finding that out."""
        svc, repo = self._svc(owner="someone-else")

        with pytest.raises(LookupError):
            run(svc.decide_match("p1", "s1", "owner-1", "Confirmed"))

        repo.update_match_owner_status.assert_not_called()
        repo.set_sighting_status.assert_not_called()

    def test_unknown_pet_is_not_found(self):
        svc, repo = self._svc(owner=None)
        with pytest.raises(LookupError):
            run(svc.decide_match("ghost", "s1", "owner-1", "Confirmed"))
        repo.update_match_owner_status.assert_not_called()

    def test_sighting_that_is_not_a_match_for_this_pet_is_not_found(self):
        svc, repo = self._svc(updated=None)
        with pytest.raises(LookupError):
            run(svc.decide_match("p1", "s1", "owner-1", "Confirmed"))
        repo.set_sighting_status.assert_not_called()

    @pytest.mark.parametrize("bad", ["Pending", "Maybe", "", None])
    def test_rejects_a_decision_outside_the_set_before_any_io(self, bad):
        """'Pending' is the value a match is born with, not an answer — writing
        it back would silently un-make a decision."""
        svc, repo = self._svc()
        with pytest.raises(ValueError):
            run(svc.decide_match("p1", "s1", "owner-1", bad))
        repo.get_pet_owners.assert_not_called()
        repo.update_match_owner_status.assert_not_called()

    def test_verdict_survives_a_failed_sighting_status_write(self):
        """The owner's answer is recorded; the sighting's own lifecycle column
        is secondary and must not take the verdict down with it."""
        svc, repo = self._svc()
        repo.set_sighting_status.side_effect = RuntimeError("db down")

        out = run(svc.decide_match("p1", "s1", "owner-1", "Confirmed"))

        assert out["match"]["owner_status"] == "X"
        assert out["sighting_status_updated"] is False
