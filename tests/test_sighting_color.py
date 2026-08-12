"""Colour-aware matching — pure logic + coat-colour extraction.

Covers app/services/sighting_logic.py (hex→Lab, lightness-weighted distance,
rerank_by_color, payload colour field) and AIManager.extract_coat_color_hex.
All pure/sync — no DB, no network, no model load.

The user's acceptance example is pinned directly: a grey pet must NOT surface
for an orange sighting, but other greys still may. See
Petty_Bounty_Brain/log.md and the colour section of api_flow.md.
"""
import numpy as np
import pytest
from PIL import Image

from app.services.ai_service import AIManager
from app.services.sighting_logic import (
    build_sighting_payload,
    color_distance,
    hex_to_lab,
    rerank_by_color,
)

# Explicit knobs so these tests don't drift with config tuning.
W_CLIP, W_COLOR = 0.7, 0.3
EXCLUDE, W_L = 45.0, 0.25


def _rerank(matches, sighting_hex, *, limit=5):
    return rerank_by_color(
        matches, sighting_hex,
        clip_weight=W_CLIP, color_weight=W_COLOR,
        exclude_distance=EXCLUDE, lightness_weight=W_L, limit=limit,
    )


# --------------------------------------------------------------------------- #
# hex_to_lab — parsing + malformed handling
# --------------------------------------------------------------------------- #
class TestHexToLab:
    @pytest.mark.parametrize("value", [None, "", "not-a-color", "#FFF",
                                       "#GGGGGG", "12345", "#1234567"])
    def test_malformed_returns_none(self, value):
        assert hex_to_lab(value) is None

    @pytest.mark.parametrize("value", ["#FFA500", "FFA500", "  #ffa500  "])
    def test_accepts_hash_optional_and_whitespace_and_case(self, value):
        assert hex_to_lab(value) is not None

    def test_black_and_white_landmarks(self):
        l_black, _, _ = hex_to_lab("#000000")
        l_white, _, _ = hex_to_lab("#FFFFFF")
        assert l_black == pytest.approx(0.0, abs=0.5)
        assert l_white == pytest.approx(100.0, abs=0.5)


# --------------------------------------------------------------------------- #
# color_distance — the metric behind the acceptance example
# --------------------------------------------------------------------------- #
class TestColorDistance:
    def test_missing_either_side_is_none(self):
        assert color_distance(None, "#808080") is None
        assert color_distance("#808080", None) is None
        assert color_distance("bad", "#808080") is None

    def test_identical_is_zero(self):
        assert color_distance("#E8820E", "#E8820E", lightness_weight=W_L) == 0.0

    def test_orange_vs_grey_is_far(self):
        # THE user example: orange sighting, grey pet — clearly different.
        d = color_distance("#E8820E", "#808080", lightness_weight=W_L)
        assert d > EXCLUDE

    def test_grey_vs_grey_is_near(self):
        # Different greys are the same colour family — must stay well inside.
        d = color_distance("#808080", "#505050", lightness_weight=W_L)
        assert d < EXCLUDE

    def test_lightness_downweight_forgives_brightness(self):
        # Same colour family, very different brightness (bright swatch vs a coat
        # in shade): down-weighting L must pull the distance DOWN vs full L.
        near = color_distance("#E8820E", "#B5651D", lightness_weight=W_L)
        full = color_distance("#E8820E", "#B5651D", lightness_weight=1.0)
        assert near < full


# --------------------------------------------------------------------------- #
# rerank_by_color — exclusion, re-ranking, fallback, trim, no mutation
# --------------------------------------------------------------------------- #
class TestRerankByColor:
    def test_excludes_clearly_wrong_colour(self):
        # Orange sighting: the grey candidate is dropped, the orange stays.
        matches = [
            {"id": "grey", "similarity": 0.95, "primary_color_hex": "#808080"},
            {"id": "orange", "similarity": 0.60, "primary_color_hex": "#E8820E"},
        ]
        out = _rerank(matches, "#E8820E")
        assert [m["id"] for m in out] == ["orange"]

    def test_colour_reranks_above_higher_clip(self):
        # pB has higher CLIP but an off colour; pA's exact-colour blend wins.
        matches = [
            {"id": "pB", "similarity": 0.68, "primary_color_hex": "#A0651E"},
            {"id": "pA", "similarity": 0.60, "primary_color_hex": "#E8820E"},
        ]
        out = _rerank(matches, "#E8820E")
        assert [m["id"] for m in out] == ["pA", "pB"]

    def test_no_sighting_colour_falls_back_to_clip_order(self):
        matches = [
            {"id": "p1", "similarity": 0.9, "primary_color_hex": "#808080"},
            {"id": "p2", "similarity": 0.3, "primary_color_hex": "#E8820E"},
        ]
        # sighting colour unavailable → colour can't help or exclude.
        out = _rerank(matches, None)
        assert [m["id"] for m in out] == ["p1", "p2"]

    def test_candidate_without_colour_is_kept_on_clip(self):
        # A candidate missing its own colour is never excluded — ranked on CLIP.
        matches = [
            {"id": "nocolor", "similarity": 0.9, "primary_color_hex": None},
            {"id": "orange", "similarity": 0.5, "primary_color_hex": "#E8820E"},
        ]
        out = _rerank(matches, "#E8820E")
        assert {m["id"] for m in out} == {"nocolor", "orange"}
        assert out[0]["id"] == "nocolor"  # raw CLIP 0.9 outranks blended 0.5+…

    def test_limit_trims_after_rerank(self):
        matches = [
            {"id": f"p{i}", "similarity": 0.9 - i * 0.1,
             "primary_color_hex": "#E8820E"} for i in range(5)
        ]
        assert len(_rerank(matches, "#E8820E", limit=2)) == 2

    def test_does_not_mutate_input_dicts(self):
        # Response contract: match dicts stay {id, similarity, …} — the blend
        # only decides order/membership, it must not inject score keys.
        m = {"id": "p1", "similarity": 0.8, "primary_color_hex": "#E8820E"}
        _rerank([m], "#E8820E")
        assert set(m.keys()) == {"id", "similarity", "primary_color_hex"}

    def test_empty_pool_returns_empty(self):
        assert _rerank([], "#E8820E") == []


# --------------------------------------------------------------------------- #
# build_sighting_payload — colour column
# --------------------------------------------------------------------------- #
class _S:
    hunter_id = "h1"
    latitude = 13.75
    longitude = 100.5
    image_url = "https://img/x.jpg"
    detected_species = "Cat"
    action_type = "Spotted"


class TestPayloadColour:
    def test_colour_written_when_present(self):
        p = build_sighting_payload(
            _S(), vector=[0.1], target_pet_id=None, primary_color_hex="#E8820E"
        )
        assert p["primary_color_hex"] == "#E8820E"

    @pytest.mark.parametrize("value", [None, ""])
    def test_colour_omitted_when_absent(self, value):
        p = build_sighting_payload(
            _S(), vector=[0.1], target_pet_id=None, primary_color_hex=value
        )
        assert "primary_color_hex" not in p


# --------------------------------------------------------------------------- #
# AIManager.extract_coat_color_hex — median over the non-black foreground
# --------------------------------------------------------------------------- #
def _img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), "RGB")


class TestExtractCoatColor:
    def test_solid_colour_returns_that_colour(self):
        arr = np.full((20, 20, 3), (232, 130, 14), dtype=np.uint8)  # #E8820E
        assert AIManager.extract_coat_color_hex(_img(arr)) == "#E8820E"

    def test_ignores_black_background(self):
        arr = np.zeros((20, 20, 3), dtype=np.uint8)      # black bg
        arr[5:15, 5:15] = (232, 130, 14)                  # 100 orange px foreground
        assert AIManager.extract_coat_color_hex(_img(arr)) == "#E8820E"

    def test_near_black_subject_returns_none(self):
        # Whole crop below the background threshold → nothing to read.
        arr = np.full((20, 20, 3), (5, 5, 5), dtype=np.uint8)
        assert AIManager.extract_coat_color_hex(_img(arr)) is None

    def test_too_few_foreground_pixels_returns_none(self):
        arr = np.zeros((20, 20, 3), dtype=np.uint8)
        arr[0, 0:10] = (232, 130, 14)   # only 10 fg px (< COLOR_MIN_PIXELS)
        assert AIManager.extract_coat_color_hex(_img(arr)) is None
