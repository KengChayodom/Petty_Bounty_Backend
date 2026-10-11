"""
The stars on a match card.

The defect: the cards are ordered by the identity model (DINOv2) but the app
drew the stars from `similarity`, which is the CLIP cosine. The two are on
different scales and disagree, so a card could sit above another and show fewer
stars ("2 stars ahead of 3"). Measured on the 23 labelled photos, 13 of 23
queries had such an inversion in their top five. CLIP also scored the true cat
and the wrong ones alike, so half of all top-five cards got the top rating.

Now the backend decides the rating (`match_stars`, 1 to 3), from the same score
that decided the order, so the two cannot disagree. `similarity` stays the CLIP
cosine, unchanged, and is still what is stored.

Thresholds come from that same eval: the true cat's final score was 0.85 to
0.90, other cats of the same colour reached 0.82, cats of another colour 0.62.
There are only four same-cat pairs behind them, so they are settings, not
constants.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.repositories.sighting_repository import SightingRepository
from app.services.identity_index import IdentityIndex
from app.services.sighting_logic import (
    rerank_by_color_rated,
    rerank_by_identity_rated,
    stars_for_score,
)
from app.services.sighting_service import SightingService

GREY = "#9E9992"
THREE, TWO = 0.80, 0.60


def run(coro):
    return asyncio.run(coro)


def _identity_rated(matches, sims, *, sighting_hex=None, limit=5):
    return rerank_by_identity_rated(
        matches, sighting_hex, sims,
        color_weight=0.1,
        exclude_distance=settings.COLOR_EXCLUDE_DISTANCE,
        lightness_weight=settings.COLOR_LIGHTNESS_WEIGHT,
        neutral_chroma=settings.NEUTRAL_CHROMA_THRESHOLD,
        neutral_lightness_exclude=settings.NEUTRAL_LIGHTNESS_EXCLUDE,
        limit=limit, three_star=THREE, two_star=TWO,
    )


def _color_rated(matches, sighting_hex, *, limit=5):
    return rerank_by_color_rated(
        matches, sighting_hex,
        clip_weight=settings.CLIP_MATCH_WEIGHT,
        color_weight=settings.COLOR_MATCH_WEIGHT,
        exclude_distance=settings.COLOR_EXCLUDE_DISTANCE,
        lightness_weight=settings.COLOR_LIGHTNESS_WEIGHT,
        neutral_chroma=settings.NEUTRAL_CHROMA_THRESHOLD,
        neutral_lightness_exclude=settings.NEUTRAL_LIGHTNESS_EXCLUDE,
        limit=limit, three_star=0.9, two_star=0.7,
    )


def _stars(matches):
    return [m["match_stars"] for m in matches]


class TestStarsForScore:
    @pytest.mark.parametrize("score, expected", [
        (0.95, 3), (0.80, 3),            # exactly at the top threshold counts
        (0.7999, 2), (0.60, 2),          # just below it, and exactly at the middle one
        (0.5999, 1), (0.0, 1), (-0.05, 1),
        (None, 1),                       # no score is the lowest rating
    ])
    def test_the_boundaries(self, score, expected):
        assert stars_for_score(score, three=THREE, two=TWO) == expected


class TestIdentityPathStars:
    def test_the_stars_follow_the_order_the_cards_are_shown_in(self):
        """The reported case. CLIP prefers the second card (0.95 against 0.85),
        identity prefers the first. The first must not show fewer stars."""
        matches = [
            {"id": "clip_fav", "similarity": 0.95, "primary_color_hex": None},
            {"id": "same_cat", "similarity": 0.85, "primary_color_hex": None},
        ]
        sims = {"clip_fav": 0.72, "same_cat": 0.88}

        out = _identity_rated(matches, sims)

        assert [m["id"] for m in out] == ["same_cat", "clip_fav"]
        assert _stars(out) == [3, 2]

    def test_stars_never_increase_down_the_list(self):
        matches = [{"id": f"p{i}", "similarity": 0.9, "primary_color_hex": GREY}
                   for i in range(8)]
        sims = {f"p{i}": v for i, v in enumerate(
            [0.9, 0.2, 0.85, 0.5, 0.61, 0.79, 0.3, 0.6])}
        stars = _stars(_identity_rated(matches, sims, sighting_hex=GREY))
        assert stars == sorted(stars, reverse=True)

    def test_a_clear_match_and_a_clear_miss_get_different_ratings(self):
        matches = [
            {"id": "same", "similarity": 0.92, "primary_color_hex": GREY},
            {"id": "other", "similarity": 0.92, "primary_color_hex": GREY},
        ]
        out = _identity_rated(matches, {"same": 0.83, "other": 0.30},
                              sighting_hex=GREY)
        assert _stars(out) == [3, 1]

    def test_the_clip_score_is_left_alone_for_display_and_storage(self):
        matches = [{"id": "a", "similarity": 0.926, "primary_color_hex": None}]
        before = [dict(m) for m in matches]
        out = _identity_rated(matches, {"a": 0.9})
        assert out[0]["similarity"] == 0.926
        assert matches == before          # the caller's dicts are not touched
        assert "match_stars" not in matches[0]

    def test_a_candidate_with_no_identity_vector_is_rated_lowest(self):
        matches = [
            {"id": "unscored", "similarity": 0.99, "primary_color_hex": None},
            {"id": "scored", "similarity": 0.80, "primary_color_hex": None},
        ]
        out = _identity_rated(matches, {"scored": 0.85})
        assert [m["id"] for m in out] == ["scored", "unscored"]
        assert _stars(out) == [3, 1]

    def test_limit_trims_after_rating(self):
        matches = [{"id": f"p{i}", "similarity": 0.9, "primary_color_hex": None}
                   for i in range(4)]
        out = _identity_rated(matches, {f"p{i}": 0.5 + i / 10 for i in range(4)},
                              limit=2)
        assert [m["id"] for m in out] == ["p3", "p2"]
        assert all("match_stars" in m for m in out)


class TestClipPathStars:
    def test_a_card_never_shows_more_stars_than_the_one_above_it(self):
        """The CLIP-only ranking orders by a blend with colour, so the raw CLIP
        score the stars come from can disagree with the order. The rating is
        capped by the card above."""
        matches = [
            {"id": "a", "similarity": 0.89, "primary_color_hex": GREY},   # 2 stars alone
            {"id": "b", "similarity": 0.905, "primary_color_hex": "#F2F2F0"},  # 3 stars alone
        ]
        out = _color_rated(matches, GREY)
        assert [m["id"] for m in out] == ["a", "b"]
        assert _stars(out) == [2, 2]

    def test_a_high_clip_score_still_earns_three_stars_at_the_top(self):
        out = _color_rated(
            [{"id": "a", "similarity": 0.93, "primary_color_hex": GREY}], GREY)
        assert _stars(out) == [3]

    def test_a_low_clip_score_earns_one_star(self):
        out = _color_rated(
            [{"id": "a", "similarity": 0.65, "primary_color_hex": GREY}], GREY)
        assert _stars(out) == [1]


def _svc(pool):
    repo = MagicMock(spec=SightingRepository)
    repo.get_sighting_for_match.return_value = {
        "id": "s1", "feature_vector": [0.1], "detected_species": "Cat",
        "sighted_location": "POINT(98.95 18.79)", "primary_color_hex": GREY,
    }
    repo.match_missing_pets.return_value = pool
    return SightingService(repo, ai_manager=None)


class TestTheServiceReturnsTheRating:
    POOL = [
        {"id": "smokey", "similarity": 0.938, "primary_color_hex": GREY,
         "image_url": "u-smokey", "species": "Cat"},
        {"id": "silver", "similarity": 0.926, "primary_color_hex": GREY,
         "image_url": "u-silver", "species": "Cat"},
    ]

    @pytest.fixture(autouse=True)
    def _clean(self):
        IdentityIndex.clear()
        yield
        IdentityIndex.clear()

    def test_identity_path_rates_every_card_in_display_order(self, monkeypatch):
        monkeypatch.setattr(settings, "IDENTITY_RERANK_ENABLED", True)
        IdentityIndex.put("smokey", "u-smokey", [0.6, 0.8])
        IdentityIndex.put("silver", "u-silver", [1.0, 0.0])

        out = run(_svc(self.POOL).get_matches("s1", identity_vector=[1.0, 0.0]))

        assert [m["id"] for m in out] == ["silver", "smokey"]
        stars = _stars(out)
        assert stars == sorted(stars, reverse=True)
        assert stars[0] == 3          # identity 1.0 against a matching photo
        assert out[0]["similarity"] == 0.926   # CLIP cosine kept

    def test_identity_path_uses_the_identity_thresholds_not_the_clip_ones(
        self, monkeypatch
    ):
        """Pins WHICH thresholds the service applies. Both cards are the same
        colour as the sighting, so each final score is identity + 0.1. Card A
        ends at 0.82, which is 3 stars against the identity thresholds (0.80)
        but only 2 against the CLIP ones (0.90). Card B ends at 0.60, which is
        2 stars against 0.60 but 1 against 0.70."""
        monkeypatch.setattr(settings, "IDENTITY_RERANK_ENABLED", True)
        pool = [
            {"id": "a", "similarity": 0.90, "primary_color_hex": GREY,
             "image_url": "u-a", "species": "Cat"},
            {"id": "b", "similarity": 0.90, "primary_color_hex": GREY,
             "image_url": "u-b", "species": "Cat"},
        ]
        IdentityIndex.put("a", "u-a", [0.72, 0.6940])   # dot with [1, 0] = 0.72
        IdentityIndex.put("b", "u-b", [0.50, 0.8660])   # dot with [1, 0] = 0.50

        out = run(_svc(pool).get_matches("s1", identity_vector=[1.0, 0.0]))

        assert [m["id"] for m in out] == ["a", "b"]
        assert _stars(out) == [3, 2]

    def test_clip_path_also_carries_the_rating(self, monkeypatch):
        monkeypatch.setattr(settings, "IDENTITY_RERANK_ENABLED", False)
        out = run(_svc(self.POOL).get_matches("s1"))
        assert all(m["match_stars"] in (1, 2, 3) for m in out)
        stars = _stars(out)
        assert stars == sorted(stars, reverse=True)
