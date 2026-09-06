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
  * SRS-33 — the server refuses to save when no cat/dog/bird is detected
    (TestProcessAndSaveCacheMiss::test_miss_yolo_finds_nothing_raises_and_skips_insert).
  * SRS-34 — the sighting save persists the USER-confirmed species + the cached
    vector, returns the ranked matches (with their cosine similarity scores,
    SRS-38), and links candidate pets via sighting_matches
    (TestProcessAndSaveCacheHit::{test_persists_user_species_and_cached_vector_not_yolo,
    test_persists_matches_into_sighting_matches}, TestGetMatches, TestPersistMatches).
  * SRS-49 — the TARGETED flow: the hunter reports one known pet straight to its
    owner (pet-detail "Report Sighting"). It stores initial_target_pet_id, skips
    the CLIP vector + match RPC, and goes through the dedicated
    save_targeted_sighting method (its own endpoint POST /sightings/targeted),
    NOT a skip_matching flag (TestSaveTargetedSighting).
  * SRS-46 — the species-correction dropdown ("No" -> pick correct species) is a
    DISCOVERY re-submit: the client calls createSightingWithMatch, which goes
    through process_and_save_sighting (matching runs again with the corrected
    species). It has no target pet, so it is distinct from SRS-49
    (TestProcessAndSaveCacheHit::test_persists_user_species_and_cached_vector_not_yolo).
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.repositories.sighting_repository import (
    SearchAlreadyClosed,
    SightingActionLocked,
    SightingAlreadyDecided,
    SightingNotSaved,
    SightingOutOfOrder,
    SightingRepository,
)
from app.schemas.sightings import SightingCreate, TargetedSightingCreate
from app.services.ai_cache import AnalyzeCache
from app.services.ai_service import EmbedResult
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
        ai.embed_image.assert_not_called()

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
# save_targeted_sighting — the pet-detail TARGETED path (UTC-26, MD-36). The
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

    def test_puts_the_report_on_the_owners_decision_queue(self):
        """A targeted report skips the AI match, so nothing else would ever
        give it a sighting_matches row — and without one its owner cannot rule
        on it and its hunter can never be scored. similarity_score stays NULL
        because no vector was computed; sightings_for_pet keys off exactly that
        to keep calling the report 'targeted' rather than 'both'."""
        sighting = make_targeted_sighting(target_pet_id="target-pet-1")
        repo = _repo()
        repo.insert_sighting.return_value = {
            "id": "s1", "hunter_id": "hunter-1",
            "detected_species": "Dog", "sighted_location": POINT,
        }
        svc = SightingService(repo, ai_manager=MagicMock())

        run(svc.save_targeted_sighting(sighting))

        repo.upsert_sighting_matches.assert_called_once_with([{
            "sighting_id": "s1",
            "missing_pet_id": "target-pet-1",
            "similarity_score": None,
            "owner_status": "Pending",
        }])

    def test_a_failed_queue_row_does_not_fail_the_report(self):
        """The sighting is already saved by then. Turning a stored report into
        an error response would tell the hunter their sighting was lost when it
        was not — the same trade-off _persist_matches makes."""
        sighting = make_targeted_sighting()
        repo = _repo()
        repo.insert_sighting.return_value = {
            "id": "s1", "hunter_id": "hunter-1",
            "detected_species": "Dog", "sighted_location": POINT,
        }
        repo.upsert_sighting_matches.side_effect = RuntimeError("db down")
        svc = SightingService(repo, ai_manager=MagicMock())

        result = run(svc.save_targeted_sighting(sighting))

        assert result["sighting"]["id"] == "s1"

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
        ai.embed_image.assert_not_called()
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
# must transparently re-run the heavy pipeline via the shared `embed_image`
# (AsyncMock). The individual download → YOLO → isolate → CLIP wiring inside
# embed_image is asserted in test_ai_service.py.
# --------------------------------------------------------------------------- #
class TestProcessAndSaveCacheMiss:
    @staticmethod
    def _make_ai(*, embed_result):
        ai = MagicMock()
        ai.embed_image = AsyncMock(return_value=embed_result)
        return ai

    def test_miss_reruns_pipeline_and_inserts_recomputed_vector(self):
        recomputed = [0.7, 0.8, 0.9]
        sighting = make_sighting(detected_species="Dog")
        # No AnalyzeCache.set -> the autouse fixture leaves the cache empty -> MISS.
        repo = _repo()
        wire_save_path(repo, recomputed)
        ai = self._make_ai(embed_result=EmbedResult(
            # YOLO re-detects a Cat; the user said Dog. The vector follows YOLO,
            # the stored species must follow the user.
            feature_vector=recomputed, species="Cat", confidence=0.91,
            bbox=[0, 0, 10, 10], isolated_image="ISOLATED_IMG",
            primary_color_hex="#AABBCC",
        ))
        svc = SightingService(repo, ai_manager=ai)

        result = run(svc.process_and_save_sighting(sighting))

        # the re-run went through the shared pipeline on the real URL
        ai.embed_image.assert_awaited_once_with(
            "https://img.example/sighting.jpg", with_color=True
        )

        insert_payload = repo.insert_sighting.call_args.args[0]
        assert insert_payload["feature_vector"] == recomputed       # re-encoded, not cached
        assert insert_payload["detected_species"] == "Dog"          # user wins over YOLO's "Cat"
        assert insert_payload["primary_color_hex"] == "#AABBCC"     # colour from the re-run
        assert "feature_vector" not in result["sighting"]           # still stripped from response

    def test_miss_isolate_runs_without_species_constraint(self):
        # The "forgiving re-run" contract: embed_image is called with NO
        # expected_species, so the vector represents whatever animal pixels YOLO
        # finds regardless of the user's species choice.
        recomputed = [0.1, 0.2]
        sighting = make_sighting(detected_species="Bird")
        repo = _repo()
        wire_save_path(repo, recomputed)
        ai = self._make_ai(embed_result=EmbedResult(
            feature_vector=recomputed, species="Dog", confidence=0.8,
            bbox=[1, 2, 3, 4], isolated_image="ISOLATED_IMG",
        ))
        svc = SightingService(repo, ai_manager=ai)

        run(svc.process_and_save_sighting(sighting))

        _, kwargs = ai.embed_image.call_args
        assert "expected_species" not in kwargs

    def test_miss_yolo_finds_nothing_raises_and_skips_insert(self):
        # SRS-33: when no cat/dog/bird is detected the server refuses to save a
        # sighting (the UI surfaces "No target animal detected, please try again").
        # embed_image reports used_full_frame -> ValueError, no row written.
        sighting = make_sighting()
        repo = _repo()
        ai = self._make_ai(embed_result=EmbedResult(
            feature_vector=[0.0], used_full_frame=True,
        ))
        svc = SightingService(repo, ai_manager=ai)

        with pytest.raises(ValueError, match="No target animal detected"):
            run(svc.process_and_save_sighting(sighting))

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
# analyze_sighting_image (UTC-14, SRS-32/33) — the heavy first step. We test its
# COORDINATION logic with a mocked AI manager + the real AnalyzeCache: the
# success path returns the verify-screen payload and caches the full result; a
# YOLO miss returns not_found without caching or encoding; a pipeline error
# re-raises without caching. (The real YOLO/CLIP detection accuracy is manual.)
# --------------------------------------------------------------------------- #
class TestAnalyzeSightingImage:
    @staticmethod
    def _make_ai(*, embed_result=None, embed_exc=None):
        ai = MagicMock()
        ai.embed_image = (
            AsyncMock(side_effect=embed_exc) if embed_exc is not None
            else AsyncMock(return_value=embed_result)
        )
        return ai

    def test_success_caches_and_returns_payload(self):
        url = "https://img.example/analyze.jpg"
        bbox = [1.0, 2.0, 3.0, 4.0]
        ai = self._make_ai(embed_result=EmbedResult(
            feature_vector=[0.11, 0.22], species="Dog", confidence=0.8825,
            bbox=bbox, isolated_image="ISOLATED_IMG", primary_color_hex="#123456",
        ))
        svc = SightingService(_repo(), ai_manager=ai)

        out = run(svc.analyze_sighting_image(url))

        assert out["status"] == "success"
        assert out["data"]["species"] == "Dog"
        assert out["data"]["confidence"] == 88.25      # round(0.8825 * 100, 2)
        assert out["data"]["bbox"] == bbox
        ai.embed_image.assert_awaited_once_with(url, with_color=True)

        cached = AnalyzeCache.get(url)
        assert cached is not None
        assert cached["feature_vector"] == [0.11, 0.22]
        assert cached["confidence"] == 0.8825          # raw, not the percentage
        assert cached["species"] == "Dog"
        assert cached["isolated_image"] == "ISOLATED_IMG"
        assert cached["primary_color_hex"] == "#123456"
        # The full decoded source frame is deliberately NOT retained: nothing
        # reads it back, and at ~9.4 MB a photo it was the whole memory
        # footprint of this cache. See the ai_cache module docstring.
        assert "pil_image" not in cached

    def test_no_target_animal_returns_not_found_without_caching(self):
        url = "https://img.example/empty-scene.jpg"
        ai = self._make_ai(embed_result=EmbedResult(
            feature_vector=[0.0], used_full_frame=True,
        ))
        svc = SightingService(_repo(), ai_manager=ai)

        out = run(svc.analyze_sighting_image(url))

        assert out["status"] == "not_found"
        assert AnalyzeCache.get(url) is None

    def test_pipeline_error_reraises_without_caching(self):
        url = "https://img.example/broken.jpg"
        ai = self._make_ai(embed_exc=RuntimeError("download failed"))
        svc = SightingService(_repo(), ai_manager=ai)

        with pytest.raises(RuntimeError):
            run(svc.analyze_sighting_image(url))
        assert AnalyzeCache.get(url) is None


# --------------------------------------------------------------------------- #
# get_sighting_by_id (UTC-32) — single read with the internal feature vector
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
# get_hunter_activity — four repo round-trips + the pure assemble step
# (sighting_logic.assemble_hunter_activity). Category-Partition:
#   * sightings list: empty (short-circuit) / non-empty
#   * per sighting: has matches / has none (assemble default branch)
#   * awards: attached by sighting_id / with a NULL sighting_id (excluded)
#   * penalties (2026-08-20): attached by sighting_id / NULL sighting_id
#     (excluded — ON DELETE SET NULL lets one outlive its sighting) / a
#     sighting carrying both an award and a penalty
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
        # no reason to fetch matches/awards/penalties when there are no sightings
        repo.get_matches_for_sightings.assert_not_called()
        repo.get_awards_for_hunter.assert_not_called()
        repo.get_penalties_for_hunter.assert_not_called()

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
        repo.get_penalties_for_hunter.return_value = []
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

    def test_attaches_the_penalty_that_punished_each_sighting(self):
        """A hunter looking at why their score fell needs the deduction on the
        sighting that caused it, not just a smaller number on the stats card."""
        repo = _repo()
        repo.count_sightings_for_hunter.return_value = 2
        repo.list_sightings_for_hunter.return_value = [{"id": "s1"}, {"id": "s2"}]
        repo.get_matches_for_sightings.return_value = []
        repo.get_awards_for_hunter.return_value = []
        repo.get_penalties_for_hunter.return_value = [
            {"sighting_id": "s2", "points": 10, "reason": "Not_a_pet"},
            # sighting_id is ON DELETE SET NULL, so a deduction can outlive the
            # sighting it punished — it must not crash the assembly.
            {"sighting_id": None, "points": 5, "reason": "Spam"},
        ]
        svc = SightingService(repo, ai_manager=None)

        s1, s2 = run(svc.get_hunter_activity("hunter-1"))["sightings"]

        assert s1["score_penalty"] is None
        assert s2["score_penalty"] == {
            "sighting_id": "s2", "points": 10, "reason": "Not_a_pet",
        }

    def test_a_sighting_can_carry_both_an_award_and_a_penalty(self):
        """They are independent records: a sighting that earned points on one
        case can still have been flagged and upheld."""
        repo = _repo()
        repo.count_sightings_for_hunter.return_value = 1
        repo.list_sightings_for_hunter.return_value = [{"id": "s1"}]
        repo.get_matches_for_sightings.return_value = []
        repo.get_awards_for_hunter.return_value = [
            {"sighting_id": "s1", "points": 25},
        ]
        repo.get_penalties_for_hunter.return_value = [
            {"sighting_id": "s1", "points": 10},
        ]
        svc = SightingService(repo, ai_manager=None)

        s1 = run(svc.get_hunter_activity("hunter-1"))["sightings"][0]

        assert s1["score_award"]["points"] == 25
        assert s1["score_penalty"]["points"] == 10

    def test_repo_error_is_reraised(self):
        repo = _repo()
        repo.count_sightings_for_hunter.side_effect = RuntimeError("db down")
        svc = SightingService(repo, ai_manager=None)
        with pytest.raises(RuntimeError):
            run(svc.get_hunter_activity("hunter-1"))


# --------------------------------------------------------------------------- #
# get_hunter_stats — total_score defaults to 0 when the user row is absent, and
# (2026-08-20) the deduction summary that explains a fallen score.
# --------------------------------------------------------------------------- #
class TestGetHunterStats:
    def test_returns_all_counts_with_user_score(self):
        repo = _repo()
        repo.get_user.return_value = {"total_score": 42}
        repo.count_sightings_for_hunter.return_value = 5
        repo.count_owner_confirmed_sightings_for_hunter.return_value = 3
        repo.count_contributions_for_hunter.return_value = 2
        repo.get_penalties_for_hunter.return_value = []
        svc = SightingService(repo, ai_manager=None)

        out = run(svc.get_hunter_stats("hunter-1"))

        assert out == {
            "total_score": 42,
            "sightings_submitted": 5,
            "sightings_verified": 3,
            "resolutions_contributed_to": 2,
            "penalties_received": 0,
            "penalty_points_total": 0,
        }

    def test_summarises_the_deductions_behind_the_score(self):
        """total_score already has them subtracted; these two fields are what
        let the card explain the drop instead of just showing it."""
        repo = _repo()
        repo.get_user.return_value = {"total_score": 15}
        repo.count_sightings_for_hunter.return_value = 4
        repo.count_owner_confirmed_sightings_for_hunter.return_value = 2
        repo.count_contributions_for_hunter.return_value = 1
        repo.get_penalties_for_hunter.return_value = [
            {"points": 10, "reason": "Not_a_pet"},
            {"points": 5, "reason": "Spam"},
        ]
        svc = SightingService(repo, ai_manager=None)

        out = run(svc.get_hunter_stats("hunter-1"))

        assert out["penalties_received"] == 2
        assert out["penalty_points_total"] == 15
        assert out["total_score"] == 15   # the balance, not derived from above

    def test_a_ruling_larger_than_the_balance_is_summed_as_ruled(self):
        """The RPC floors total_score at 0 but records the full ruling, so a
        20-point penalty against a 3-point hunter still reads as 20 here."""
        repo = _repo()
        repo.get_user.return_value = {"total_score": 0}
        repo.count_sightings_for_hunter.return_value = 1
        repo.count_owner_confirmed_sightings_for_hunter.return_value = 0
        repo.count_contributions_for_hunter.return_value = 0
        repo.get_penalties_for_hunter.return_value = [{"points": 20}]
        svc = SightingService(repo, ai_manager=None)

        out = run(svc.get_hunter_stats("hunter-1"))

        assert out["penalty_points_total"] == 20
        assert out["total_score"] == 0

    def test_defaults_score_to_zero_when_no_user_row(self):
        repo = _repo()
        repo.get_user.return_value = None   # no profile row
        repo.count_sightings_for_hunter.return_value = 0
        repo.count_owner_confirmed_sightings_for_hunter.return_value = 0
        repo.count_contributions_for_hunter.return_value = 0
        repo.get_penalties_for_hunter.return_value = []
        svc = SightingService(repo, ai_manager=None)

        assert run(svc.get_hunter_stats("hunter-1"))["total_score"] == 0

    def test_repo_error_is_reraised(self):
        repo = _repo()
        repo.get_user.side_effect = RuntimeError("db down")
        svc = SightingService(repo, ai_manager=None)
        with pytest.raises(RuntimeError):
            run(svc.get_hunter_stats("hunter-1"))


# --------------------------------------------------------------------------- #
# decide_match — the owner's verdict on one card of their pet's queue.
#
# Rewritten 2026-08-21. The verdict used to be two writes issued from here
# (owner_status, then the sighting's own status) behind a Python-side ownership
# check. It is now a single `owner_decide_sighting` RPC, because confirming a
# 'Caught' card ALSO ends the search and distributes every clue score for the
# case — work that cannot be left half-done, and whose queue rules cannot be
# checked here without a window for a second request to slip through.
#
# So what is left to test at this layer is exactly the seam: the decision string
# is normalised BEFORE any I/O, the RPC is called with the caller's identity,
# its result is passed through untouched, and its refusals arrive as the typed
# errors the route maps to 404 / 409. The rules themselves — ordering, one seat
# per hunter, the 25/15/10/5/5 ladder — belong to the database and are pinned by
# tests/integration/test_owner_decide_sighting.py, against the real function.
# --------------------------------------------------------------------------- #
class TestDecideMatch:
    @staticmethod
    def _svc(result=None):
        repo = _repo()
        repo.owner_decide_sighting.return_value = result if result is not None else {
            "pet_id": "p1", "sighting_id": "s1", "owner_status": "Confirmed",
            "search_closed": False, "pet_status": "Searching", "awards": [],
        }
        return SightingService(repo, ai_manager=None), repo

    def test_passes_the_normalised_verdict_and_the_caller_identity(self):
        svc, repo = self._svc()

        out = run(svc.decide_match("p1", "s1", "owner-1", "confirm"))

        # The alias "confirm" reaches the database as the enum value.
        repo.owner_decide_sighting.assert_called_once_with(
            "p1", "s1", "owner-1", "Confirmed",
        )
        assert out["owner_status"] == "Confirmed"

    def test_returns_the_rpc_result_unchanged(self):
        """The awards list is the only record of who was paid what. The service
        must not summarise, filter or re-shape it on the way out."""
        payload = {
            "pet_id": "p1", "sighting_id": "s4", "owner_status": "Confirmed",
            "search_closed": True, "pet_status": "Found",
            "awards": [
                {"user_id": "h3", "sighting_id": "s4", "rank": 1, "points": 25},
                {"user_id": "h1", "sighting_id": "s2", "rank": 2, "points": 15},
            ],
        }
        svc, _ = self._svc(result=payload)

        assert run(svc.decide_match("p1", "s4", "owner-1", "Confirmed")) is payload

    def test_a_rejection_is_forwarded_like_any_other_verdict(self):
        """One photo can match several pets, so "not mine" is a verdict about
        this pet only. Nothing else is called on the way."""
        svc, repo = self._svc()

        run(svc.decide_match("p1", "s1", "owner-1", "Rejected"))

        repo.owner_decide_sighting.assert_called_once_with(
            "p1", "s1", "owner-1", "Rejected",
        )
        repo.set_sighting_status.assert_not_called()

    @pytest.mark.parametrize("bad", ["Pending", "Maybe", "", None])
    def test_rejects_a_decision_outside_the_set_before_any_io(self, bad):
        """'Pending' is the value a card is born with, not an answer — writing
        it back would silently un-make a decision."""
        svc, repo = self._svc()
        with pytest.raises(ValueError):
            run(svc.decide_match("p1", "s1", "owner-1", bad))
        repo.owner_decide_sighting.assert_not_called()

    @pytest.mark.parametrize("error", [
        LookupError("Missing pet p1 not found or not owned by you"),
        SightingAlreadyDecided("Sighting s1 has already been decided"),
        SightingOutOfOrder("Sighting s1 is out of order"),
        SearchAlreadyClosed("Search for pet p1 is already closed"),
    ])
    def test_repository_refusals_reach_the_caller_with_their_type_intact(
        self, error,
    ):
        """Each refusal is a different HTTP status at the route, so the service
        must not flatten them into one generic failure."""
        svc, repo = self._svc()
        repo.owner_decide_sighting.side_effect = error

        with pytest.raises(type(error)):
            run(svc.decide_match("p1", "s1", "owner-1", "Confirmed"))


# --------------------------------------------------------------------------- #
# confirm_sighting_action (2026-08-19) — the final-review screen, one step
# after "Confirm Match": JUST SPOTTED or RESCUE.
#
# The row already exists (POST /sightings/ wrote it with the 'Spotted'
# default), so this is a narrow, hunter-scoped update of `action_type`. What
# the tests pin: the 404-not-403 ownership rule, the freeze once someone has
# reviewed the sighting (409 — otherwise a Verified sighting could be
# retro-fitted into the bounty-eligible Caught+Verified shape), and the
# no-write short circuit when nothing actually changed.
# --------------------------------------------------------------------------- #
class TestConfirmSightingAction:
    @staticmethod
    def _svc(row={"id": "s1", "hunter_id": "hunter-1", "action_type": "Spotted",
                  "verification_status": "Pending"},
             updated={"id": "s1", "hunter_id": "hunter-1",
                      "action_type": "Caught", "feature_vector": "[0.1]"}):
        repo = _repo()
        repo.get_sighting_for_action.return_value = row
        repo.set_sighting_action_type.return_value = updated
        return SightingService(repo, ai_manager=None), repo

    def test_rescue_writes_caught_and_returns_the_row(self):
        svc, repo = self._svc()

        out = run(svc.confirm_sighting_action("s1", "hunter-1", "Rescue"))

        repo.set_sighting_action_type.assert_called_once_with(
            "s1", "hunter-1", "Caught"
        )
        assert out["action_type"] == "Caught"
        assert out["changed"] is True

    def test_strips_the_feature_vector_from_the_returned_row(self):
        """Same contract as every other sighting response — the 512-D vector
        is bloat the client never reads."""
        svc, _ = self._svc()

        out = run(svc.confirm_sighting_action("s1", "hunter-1", "Caught"))

        assert "feature_vector" not in out["sighting"]

    def test_reconfirming_the_stored_value_writes_nothing(self):
        """'Spotted' is both the UI default and the column default, so the
        common submit is a no-op. It must still report success."""
        svc, repo = self._svc()

        out = run(svc.confirm_sighting_action("s1", "hunter-1", "Spotted"))

        repo.set_sighting_action_type.assert_not_called()
        assert out["changed"] is False
        assert out["action_type"] == "Spotted"

    def test_sighting_reported_by_someone_else_is_not_found(self):
        """404, not 403 — a caller must not learn that a sighting id exists.
        And nothing may be written on the way to finding that out."""
        svc, repo = self._svc()

        with pytest.raises(LookupError):
            run(svc.confirm_sighting_action("s1", "another-hunter", "Caught"))

        repo.set_sighting_action_type.assert_not_called()

    def test_unknown_sighting_is_not_found(self):
        svc, repo = self._svc(row=None)
        with pytest.raises(LookupError):
            run(svc.confirm_sighting_action("ghost", "hunter-1", "Caught"))
        repo.set_sighting_action_type.assert_not_called()

    def test_row_vanishing_between_read_and_write_is_not_found(self):
        """The read said it was theirs, the scoped write matched nothing — the
        row went away or changed hands. Same 404, never a silent success."""
        svc, _ = self._svc(updated=None)
        with pytest.raises(LookupError):
            run(svc.confirm_sighting_action("s1", "hunter-1", "Caught"))

    @pytest.mark.parametrize("reviewed", ["Verified", "Dismissed"])
    def test_already_reviewed_sighting_is_locked(self, reviewed):
        """Once an owner/admin has judged the report, flipping it to 'Caught'
        would retro-fit it into the shape the resolve RPC pays out on."""
        svc, repo = self._svc(row={
            "id": "s1", "hunter_id": "hunter-1", "action_type": "Spotted",
            "verification_status": reviewed,
        })

        with pytest.raises(SightingActionLocked):
            run(svc.confirm_sighting_action("s1", "hunter-1", "Caught"))

        repo.set_sighting_action_type.assert_not_called()

    def test_missing_verification_status_is_treated_as_pending(self):
        """The column is NOT NULL DEFAULT 'Pending'; absence means 'not yet
        judged', so a fresh sighting must not be locked by a partial read."""
        svc, repo = self._svc(row={
            "id": "s1", "hunter_id": "hunter-1", "action_type": "Spotted",
        })

        out = run(svc.confirm_sighting_action("s1", "hunter-1", "Caught"))

        repo.set_sighting_action_type.assert_called_once()
        assert out["changed"] is True

    @pytest.mark.parametrize("bad", ["Found", "Verified", "Maybe", "", None])
    def test_rejects_an_action_outside_the_enum_before_any_io(self, bad):
        svc, repo = self._svc()
        with pytest.raises(ValueError):
            run(svc.confirm_sighting_action("s1", "hunter-1", bad))
        repo.get_sighting_for_action.assert_not_called()
        repo.set_sighting_action_type.assert_not_called()
