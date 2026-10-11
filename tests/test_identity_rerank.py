"""
Identity re-rank: the CLIP candidate pool re-ordered by an identity model.

What these pin:
  * rerank_by_identity orders by identity, lets colour only nudge, and never
    drops a candidate. The live failures it exists for are pinned by name:
    Oreo (a white cat measured cream under warm light, cut at CLIP 0.9996) and
    Silver (ranked 3rd because CLIP scored two other silver tabbies higher).
  * IdentityIndex never scores a pet with a vector built from another photo.
  * ensure_vectors / warm_from_repo fill only what is missing and survive one
    bad photo.
  * SightingService uses the identity path when it has the sighting's vector,
    and falls back to CLIP + colour when it does not or when that path fails.
  * register_missing_pet stores the new pet's vector straight away.

Models are never loaded: AIManager is patched at its class boundary and the
repository is a MagicMock(spec=port), as in the rest of the suite.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import settings
from app.repositories.missing_pet_repository import MissingPetRepository
from app.repositories.pagination import Page
from app.repositories.sighting_repository import SightingRepository
from app.services import identity_index
from app.services.ai_cache import AnalyzeCache
from app.services.ai_service import AIManager, EmbedResult
from app.services.identity_index import (
    IdentityIndex,
    ensure_vectors,
    identity_similarities,
    warm_from_repo,
)
from app.services.sighting_logic import rerank_by_identity
from app.services.sighting_service import SightingService

ORANGE, CREAM, WHITE, GREY = "#D9A064", "#B6A48F", "#F2F2F0", "#9E9992"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean_state():
    IdentityIndex.clear()
    AnalyzeCache.clear()
    yield
    IdentityIndex.clear()
    AnalyzeCache.clear()


def _rerank(matches, sighting_hex, sims, *, color_weight=0.1, limit=5):
    return rerank_by_identity(
        matches, sighting_hex, sims,
        color_weight=color_weight,
        exclude_distance=settings.COLOR_EXCLUDE_DISTANCE,
        lightness_weight=settings.COLOR_LIGHTNESS_WEIGHT,
        neutral_chroma=settings.NEUTRAL_CHROMA_THRESHOLD,
        neutral_lightness_exclude=settings.NEUTRAL_LIGHTNESS_EXCLUDE,
        limit=limit,
    )


# --------------------------------------------------------------------------- #
# rerank_by_identity (pure)
# --------------------------------------------------------------------------- #
class TestRerankByIdentity:
    def test_the_true_cat_ranks_first_even_below_on_clip(self):
        # The Silver screenshot: CLIP put two other silver tabbies above it.
        matches = [
            {"id": "smokey", "similarity": 0.938, "primary_color_hex": GREY},
            {"id": "pepper", "similarity": 0.927, "primary_color_hex": GREY},
            {"id": "silver", "similarity": 0.926, "primary_color_hex": GREY},
        ]
        sims = {"smokey": 0.57, "pepper": 0.68, "silver": 0.83}
        assert [m["id"] for m in _rerank(matches, GREY, sims)] == [
            "silver", "pepper", "smokey"]

    def test_a_clashing_colour_never_removes_a_candidate(self):
        # The Oreo screenshot: the sighting measured cream under warm light,
        # the pet is stored white, and the old re-rank cut it at CLIP 0.9996.
        matches = [
            {"id": "oreo", "similarity": 0.9996, "primary_color_hex": WHITE},
            {"id": "mango", "similarity": 0.89, "primary_color_hex": ORANGE},
        ]
        sims = {"oreo": 0.98, "mango": 0.40}
        out = _rerank(matches, CREAM, sims)
        assert [m["id"] for m in out] == ["oreo", "mango"]

    def test_colour_breaks_a_near_tie(self):
        matches = [
            {"id": "grey", "similarity": 0.9, "primary_color_hex": GREY},
            {"id": "orange", "similarity": 0.9, "primary_color_hex": ORANGE},
        ]
        sims = {"grey": 0.70, "orange": 0.69}
        assert [m["id"] for m in _rerank(matches, ORANGE, sims)] == [
            "orange", "grey"]

    def test_colour_cannot_overturn_a_clear_identity_lead(self):
        matches = [
            {"id": "same_cat", "similarity": 0.9, "primary_color_hex": GREY},
            {"id": "same_colour", "similarity": 0.9, "primary_color_hex": ORANGE},
        ]
        sims = {"same_cat": 0.80, "same_colour": 0.60}
        assert _rerank(matches, ORANGE, sims)[0]["id"] == "same_cat"

    def test_unknown_colour_sits_between_a_match_and_a_clash(self):
        matches = [
            {"id": "clash", "similarity": 0.9, "primary_color_hex": GREY},
            {"id": "unknown", "similarity": 0.9, "primary_color_hex": None},
            {"id": "match", "similarity": 0.9, "primary_color_hex": ORANGE},
        ]
        sims = {"clash": 0.7, "unknown": 0.7, "match": 0.7}
        assert [m["id"] for m in _rerank(matches, ORANGE, sims)] == [
            "match", "unknown", "clash"]

    def test_candidates_without_a_vector_follow_in_clip_order(self):
        matches = [
            {"id": "a", "similarity": 0.95, "primary_color_hex": None},
            {"id": "b", "similarity": 0.94, "primary_color_hex": None},
            {"id": "scored", "similarity": 0.80, "primary_color_hex": None},
        ]
        out = _rerank(matches, None, {"scored": 0.3})
        assert [m["id"] for m in out] == ["scored", "a", "b"]

    def test_limit_trims_and_dicts_are_not_mutated(self):
        matches = [{"id": f"p{i}", "similarity": 0.9, "primary_color_hex": None}
                   for i in range(4)]
        before = [dict(m) for m in matches]
        out = _rerank(matches, None, {f"p{i}": i / 10 for i in range(4)}, limit=2)
        assert [m["id"] for m in out] == ["p3", "p2"]
        assert matches == before


# --------------------------------------------------------------------------- #
# IdentityIndex + filling
# --------------------------------------------------------------------------- #
class TestIdentityIndex:
    def test_a_vector_built_from_another_photo_is_not_returned(self):
        IdentityIndex.put("p1", "https://img/old.jpg", [1.0, 0.0])
        assert IdentityIndex.get("p1", "https://img/old.jpg") == [1.0, 0.0]
        assert IdentityIndex.get("p1", "https://img/new.jpg") is None

    def test_similarities_skip_rows_without_a_vector(self):
        IdentityIndex.put("p1", "u1", [1.0, 0.0])
        IdentityIndex.put("p2", "u2", [0.0, 1.0])
        pool = [{"id": "p1", "image_url": "u1"}, {"id": "p2", "image_url": "u2"},
                {"id": "p3", "image_url": "u3"}]
        sims = identity_similarities([1.0, 0.0], pool)
        assert sims == {"p1": pytest.approx(1.0), "p2": pytest.approx(0.0)}


class TestEnsureVectors:
    def test_only_missing_vectors_are_built_and_one_failure_is_isolated(
        self, monkeypatch,
    ):
        IdentityIndex.put("have", "u-have", [1.0])

        async def fake(url, *, expected_species=None):
            if url == "u-bad":
                raise RuntimeError("download failed")
            return [0.5]

        spy = AsyncMock(side_effect=fake)
        monkeypatch.setattr(AIManager, "identity_vector_for", spy)
        run(ensure_vectors([
            {"id": "have", "image_url": "u-have", "species": "Cat"},
            {"id": "new", "image_url": "u-new", "species": "Cat"},
            {"id": "bad", "image_url": "u-bad", "species": "Cat"},
        ]))

        assert IdentityIndex.get("new", "u-new") == [0.5]
        assert IdentityIndex.get("bad", "u-bad") is None
        called = sorted(c.args[0] for c in spy.await_args_list)
        assert called == ["u-bad", "u-new"]
        assert spy.await_args_list[0].kwargs == {"expected_species": "Cat"}

    def test_warm_reads_every_searching_page(self, monkeypatch):
        monkeypatch.setattr(
            AIManager, "identity_vector_for", AsyncMock(return_value=[0.1]))
        repo = MagicMock(spec=MissingPetRepository)
        repo.list_all.side_effect = [
            Page([{"id": "a", "image_url": "ua"}, {"id": "b", "image_url": "ub"}], 3),
            Page([{"id": "c", "image_url": "uc"}], 3),
        ]

        ready = run(warm_from_repo(repo, page_size=2))

        assert ready == 3
        assert [c.args for c in repo.list_all.call_args_list] == [
            ("Searching", None, 2, 0), ("Searching", None, 2, 2)]


# --------------------------------------------------------------------------- #
# SightingService wiring
# --------------------------------------------------------------------------- #
def _sighting_repo(pool):
    repo = MagicMock(spec=SightingRepository)
    repo.get_sighting_for_match.return_value = {
        "id": "s1", "feature_vector": [0.1], "detected_species": "Cat",
        "sighted_location": "POINT(98.95 18.79)", "primary_color_hex": GREY,
    }
    repo.match_missing_pets.return_value = pool
    return repo


POOL = [
    {"id": "smokey", "similarity": 0.938, "primary_color_hex": GREY,
     "image_url": "u-smokey", "species": "Cat"},
    {"id": "silver", "similarity": 0.926, "primary_color_hex": GREY,
     "image_url": "u-silver", "species": "Cat"},
]


class TestGetMatchesIdentityPath:
    def test_with_the_sighting_vector_the_pool_is_ordered_by_identity(
        self, monkeypatch,
    ):
        monkeypatch.setattr(settings, "IDENTITY_RERANK_ENABLED", True)
        IdentityIndex.put("smokey", "u-smokey", [0.6, 0.8])
        IdentityIndex.put("silver", "u-silver", [1.0, 0.0])
        svc = SightingService(_sighting_repo(POOL), ai_manager=None)

        out = run(svc.get_matches("s1", identity_vector=[1.0, 0.0]))

        assert [m["id"] for m in out] == ["silver", "smokey"]
        assert out[0]["similarity"] == 0.926  # CLIP score kept for the stars

    def test_without_the_sighting_vector_clip_and_colour_decide(self, monkeypatch):
        monkeypatch.setattr(settings, "IDENTITY_RERANK_ENABLED", True)
        IdentityIndex.put("silver", "u-silver", [1.0, 0.0])
        svc = SightingService(_sighting_repo(POOL), ai_manager=None)

        out = run(svc.get_matches("s1"))

        assert [m["id"] for m in out] == ["smokey", "silver"]

    def test_switched_off_the_identity_vector_is_ignored(self, monkeypatch):
        monkeypatch.setattr(settings, "IDENTITY_RERANK_ENABLED", False)
        IdentityIndex.put("smokey", "u-smokey", [0.6, 0.8])
        IdentityIndex.put("silver", "u-silver", [1.0, 0.0])
        svc = SightingService(_sighting_repo(POOL), ai_manager=None)

        out = run(svc.get_matches("s1", identity_vector=[1.0, 0.0]))

        assert [m["id"] for m in out] == ["smokey", "silver"]

    def test_a_failing_identity_step_falls_back_instead_of_failing(
        self, monkeypatch,
    ):
        monkeypatch.setattr(settings, "IDENTITY_RERANK_ENABLED", True)
        monkeypatch.setattr(
            "app.services.sighting_service.ensure_vectors",
            AsyncMock(side_effect=RuntimeError("model crashed")),
        )
        svc = SightingService(_sighting_repo(POOL), ai_manager=None)

        out = run(svc.get_matches("s1", identity_vector=[1.0, 0.0]))

        assert [m["id"] for m in out] == ["smokey", "silver"]

    def test_an_unknown_sighting_still_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "IDENTITY_RERANK_ENABLED", True)
        repo = _sighting_repo(POOL)
        repo.get_sighting_for_match.return_value = None
        svc = SightingService(repo, ai_manager=None)

        with pytest.raises(ValueError):
            run(svc.get_matches("ghost", identity_vector=[1.0, 0.0]))

    def test_missing_pet_vectors_are_built_for_the_pool(self, monkeypatch):
        monkeypatch.setattr(settings, "IDENTITY_RERANK_ENABLED", True)
        vectors = {"u-smokey": [0.0, 1.0], "u-silver": [1.0, 0.0]}
        monkeypatch.setattr(
            AIManager, "identity_vector_for",
            AsyncMock(side_effect=lambda url, **_: vectors[url]),
        )
        svc = SightingService(_sighting_repo(POOL), ai_manager=None)

        out = run(svc.get_matches("s1", identity_vector=[1.0, 0.0]))

        assert out[0]["id"] == "silver"


class TestIdentityVectorTravelsThroughTheSave:
    def test_analyze_caches_the_identity_vector(self, monkeypatch):
        monkeypatch.setattr(settings, "IDENTITY_RERANK_ENABLED", True)
        ai = MagicMock()
        ai.embed_image = AsyncMock(return_value=EmbedResult(
            feature_vector=[0.1], species="Cat", confidence=0.9, bbox=[0, 0, 1, 1],
            isolated_image="IMG", primary_color_hex=GREY,
            identity_vector=[1.0, 0.0],
        ))
        svc = SightingService(MagicMock(spec=SightingRepository), ai_manager=ai)

        run(svc.analyze_sighting_image("https://img/s.jpg"))

        assert AnalyzeCache.get("https://img/s.jpg")["identity_vector"] == [1.0, 0.0]
        assert ai.embed_image.await_args.kwargs["with_identity"] is True

    def test_the_save_hands_the_cached_vector_to_matching(self, monkeypatch):
        AnalyzeCache.set("https://img.example/sighting.jpg", {
            "isolated_image": "IMG", "species": "Cat", "bbox": [0, 0, 1, 1],
            "confidence": 0.9, "feature_vector": [0.1],
            "primary_color_hex": GREY, "identity_vector": [1.0, 0.0],
        })
        repo = MagicMock(spec=SightingRepository)
        repo.insert_sighting.return_value = {"id": "s1"}
        svc = SightingService(repo, ai_manager=MagicMock())
        svc.get_matches = AsyncMock(return_value=[])

        from app.schemas.sightings import SightingCreate
        run(svc.process_and_save_sighting(SightingCreate(
            hunter_id="h1", image_url="https://img.example/sighting.jpg",
            latitude=18.79, longitude=98.95, detected_species="Cat",
        )))

        assert svc.get_matches.await_args.kwargs["identity_vector"] == [1.0, 0.0]


class TestRegisterStoresTheVector:
    def test_a_new_pet_is_ready_for_the_next_sighting(self, monkeypatch):
        from datetime import datetime

        from app.schemas.missing_pets import MissingPetCreate
        from app.services.pet_service import PetService

        monkeypatch.setattr(settings, "IDENTITY_RERANK_ENABLED", True)
        monkeypatch.setattr(AIManager, "embed_image", AsyncMock(return_value=EmbedResult(
            feature_vector=[0.1], species="Cat", confidence=0.9, bbox=[0, 0, 1, 1],
            isolated_image="IMG", primary_color_hex=GREY, identity_vector=[0.3, 0.4],
        )))
        repo = MagicMock(spec=MissingPetRepository)
        repo.insert_missing_pet.return_value = {"id": "pet-9"}
        pet = MissingPetCreate(
            owner_id="o1", pet_name="Silver", species="Cat",
            characteristics={"traits": "silver tabby"}, bounty_amount=0,
            latitude=18.79, longitude=98.95,
            last_seen_time=datetime(2026, 9, 25, 12, 0),
            image_url="https://img.example/silver.jpg",
        )

        run(PetService(repo).register_missing_pet(pet))

        assert IdentityIndex.get("pet-9", "https://img.example/silver.jpg") == [0.3, 0.4]
        assert AIManager.embed_image.await_args.kwargs["with_identity"] is True
