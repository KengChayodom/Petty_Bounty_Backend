"""
Unit tests for app/services/ai_service.AIManager — the NON-ML branches only.

The real YOLO/CLIP inference (model load + predict + encode) is the L4/slow
concern; here every model boundary is stubbed, so what these tests actually
exercise is the plumbing and the pure logic:
  * lazy-load + class-level caching of the two models,
  * run_yolo_seg / clip_encode delegating to the (stubbed) model,
  * warmup_models running both and swallowing a load failure,
  * isolate_subject — the subject-selection loop + numpy mask handling, which
    is pure array work and needs no model at all.

`isolate_subject` is fed hand-built stand-ins for the Ultralytics result
(boxes / masks / names), so the branch behaviour is exercised deterministically.
"""
import asyncio
import io
from unittest.mock import AsyncMock, MagicMock

import numpy as np
from PIL import Image

from app.services import ai_service
from app.services.ai_service import AIManager, EmbedResult


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Stand-ins for the Ultralytics result shape isolate_subject reads.
# --------------------------------------------------------------------------- #
class _Tensor:
    """Mimics a torch tensor's `.cpu().numpy()` used to pull a mask out."""
    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _Box:
    def __init__(self, cls_idx, conf):
        self.cls = [cls_idx]     # box.cls[0] -> int(...)
        self.conf = [conf]       # box.conf[0] -> float(...)


class _Masks:
    def __init__(self, arrays):
        self.data = [_Tensor(a) for a in arrays]


class _Result:
    def __init__(self, boxes, masks, names):
        self.boxes = boxes
        self.masks = masks
        self.names = names


def _img(w=10, h=10, fill=200):
    return Image.fromarray(np.full((h, w, 3), fill, dtype=np.uint8))


def _mask(h=10, w=10, region=None):
    m = np.zeros((h, w), dtype=float)
    if region:
        r0, r1, c0, c1 = region
        m[r0:r1, c0:c1] = 1.0
    return m


# --------------------------------------------------------------------------- #
# isolate_subject — the pure subject-selection + mask logic (no model)
# --------------------------------------------------------------------------- #
class TestIsolateSubject:
    def test_none_when_no_results(self):
        assert AIManager.isolate_subject(_img(), []) is None

    def test_none_when_masks_or_boxes_missing(self):
        box = _Box(0, 0.9)
        no_masks = _Result(boxes=[box], masks=None, names={0: "dog"})
        no_boxes = _Result(boxes=None, masks=_Masks([_mask()]), names={0: "dog"})
        assert AIManager.isolate_subject(_img(), [no_masks]) is None
        assert AIManager.isolate_subject(_img(), [no_boxes]) is None

    def test_none_when_no_target_animal(self):
        # 'person' is not in TARGET_ANIMALS -> skipped -> no candidate.
        res = _Result(boxes=[_Box(0, 0.99)], masks=_Masks([_mask()]),
                      names={0: "person"})
        assert AIManager.isolate_subject(_img(), [res]) is None

    def test_expected_species_filters_out_other_animals(self):
        # A higher-confidence cat is present, but expected_species='dog' pins it.
        masks = _Masks([_mask(region=(2, 5, 3, 6)), _mask(region=(2, 5, 3, 6))])
        res = _Result(
            boxes=[_Box(0, 0.90), _Box(1, 0.95)],
            masks=masks,
            names={0: "dog", 1: "cat"},
        )
        out = AIManager.isolate_subject(_img(), [res], expected_species="dog")
        assert out is not None
        _iso, species, conf, _bbox = out
        assert species == "Dog"       # the cat was filtered despite higher conf
        assert conf == 0.90

    def test_picks_highest_confidence_target(self):
        # Order high-then-low so BOTH arcs of `if c > best_conf` are taken:
        # the first box updates the best, the second (lower) does not.
        masks = _Masks([_mask(region=(2, 5, 3, 6)), _mask(region=(2, 5, 3, 6))])
        res = _Result(
            boxes=[_Box(0, 0.90), _Box(0, 0.70)],
            masks=masks,
            names={0: "dog"},
        )
        out = AIManager.isolate_subject(_img(), [res])
        assert out is not None
        assert out[2] == 0.90         # the 0.90 detection won

    def test_happy_path_no_resize_returns_crop_species_conf_bbox(self):
        img = _img(w=50, h=50)
        mask = _mask(h=50, w=50, region=(20, 25, 30, 35))  # rows 20-24, cols 30-34
        res = _Result(boxes=[_Box(0, 0.88)], masks=_Masks([mask]),
                      names={0: "dog"})

        out = AIManager.isolate_subject(img, [res])

        assert out is not None
        iso, species, conf, bbox = out
        assert isinstance(iso, Image.Image)
        assert species == "Dog"
        assert conf == 0.88
        # bbox = mask bbox padded by MASK_PADDING (10), clamped to the image.
        assert bbox == [20.0, 10.0, 45.0, 35.0]

    def test_resize_branch_when_mask_dims_differ(self):
        # Mask comes at 640x640 (network resolution) but the image is 50x50, so
        # the resize-to-image branch runs.
        img = _img(w=50, h=50)
        big = _mask(h=640, w=640, region=(100, 540, 100, 540))
        res = _Result(boxes=[_Box(0, 0.8)], masks=_Masks([big]),
                      names={0: "dog"})

        out = AIManager.isolate_subject(img, [res])

        assert out is not None          # resized mask is non-empty
        assert out[1] == "Dog"

    def test_none_when_mask_is_all_background(self):
        # A qualifying box, but its mask selects zero pixels -> no crop.
        res = _Result(boxes=[_Box(0, 0.9)], masks=_Masks([_mask()]),  # all zeros
                      names={0: "dog"})
        assert AIManager.isolate_subject(_img(), [res]) is None


# --------------------------------------------------------------------------- #
# download_image — httpx fetch + PIL decode (network boundary stubbed)
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None


class _FakeClient:
    def __init__(self, content):
        self._content = content

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        return _FakeResp(self._content)


class TestDownloadImage:
    def test_fetches_and_decodes_to_rgb(self, monkeypatch):
        buf = io.BytesIO()
        Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, format="PNG")
        png = buf.getvalue()
        monkeypatch.setattr(
            ai_service.httpx, "AsyncClient", lambda *a, **k: _FakeClient(png)
        )

        img = run(AIManager.download_image("http://img/x.png"))

        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (4, 4)


# --------------------------------------------------------------------------- #
# Model lazy-load + caching (the model constructor is stubbed)
# --------------------------------------------------------------------------- #
class TestModelLoading:
    def test_get_yolo_loads_once_then_caches(self, monkeypatch):
        monkeypatch.setattr(AIManager, "_yolo", None)
        calls = []
        monkeypatch.setattr(ai_service, "YOLO",
                            lambda weights: calls.append(weights) or "YOLO_MODEL")

        first = AIManager.get_yolo()
        second = AIManager.get_yolo()

        assert first == "YOLO_MODEL"
        assert second is first                     # cached, not rebuilt
        assert calls == [AIManager.YOLO_WEIGHTS]   # constructed exactly once

    def test_get_clip_loads_once_then_caches(self, monkeypatch):
        monkeypatch.setattr(AIManager, "_clip", None)
        calls = []
        monkeypatch.setattr(ai_service, "SentenceTransformer",
                            lambda name: calls.append(name) or "CLIP_MODEL")

        first = AIManager.get_clip()
        second = AIManager.get_clip()

        assert first == "CLIP_MODEL"
        assert second is first
        assert calls == [AIManager.CLIP_MODEL]


# --------------------------------------------------------------------------- #
# run_yolo_seg / clip_encode delegate to the (stubbed) model
# --------------------------------------------------------------------------- #
class TestPipelineDelegation:
    def test_run_yolo_seg_delegates_to_predict(self, monkeypatch):
        model = MagicMock()
        model.predict.return_value = "YOLO_RESULTS"
        monkeypatch.setattr(AIManager, "get_yolo", lambda: model)

        out = run(AIManager.run_yolo_seg("PIL_IMAGE", conf=0.3))

        assert out == "YOLO_RESULTS"
        model.predict.assert_called_once_with(
            source="PIL_IMAGE", conf=0.3, verbose=False
        )

    def test_clip_encode_returns_plain_list(self, monkeypatch):
        model = MagicMock()
        model.encode.return_value = np.array([1.0, 2.0, 3.0])
        monkeypatch.setattr(AIManager, "get_clip", lambda: model)

        out = run(AIManager.clip_encode("ISOLATED_IMG"))

        assert out == [1.0, 2.0, 3.0]              # ndarray -> list
        model.encode.assert_called_once_with("ISOLATED_IMG")


# --------------------------------------------------------------------------- #
# embed_image — the ONE shared pipeline (download → YOLO → isolate → CLIP →
# colour). This is where the wiring assertions live now that the four callers
# (register_missing_pet, analyze, cache-miss, seed) delegate here.
# --------------------------------------------------------------------------- #
class TestEmbedImage:
    @staticmethod
    def _patch(monkeypatch, *, iso, vector, color="#ABCDEF"):
        monkeypatch.setattr(AIManager, "download_image", AsyncMock(return_value="SRC_IMG"))
        monkeypatch.setattr(AIManager, "run_yolo_seg", AsyncMock(return_value="YOLO_RESULTS"))
        monkeypatch.setattr(AIManager, "isolate_subject", MagicMock(return_value=iso))
        monkeypatch.setattr(AIManager, "clip_encode", AsyncMock(return_value=vector))
        monkeypatch.setattr(AIManager, "extract_coat_color_hex", MagicMock(return_value=color))

    def test_hit_encodes_the_isolated_crop_and_fills_every_field(self, monkeypatch):
        self._patch(
            monkeypatch,
            iso=("CROP_IMG", "Dog", 0.9, [1.0, 2.0, 3.0, 4.0]),
            vector=[0.1, 0.2],
        )

        r = run(AIManager.embed_image("http://img/x.jpg",
                                      expected_species="Dog", with_color=True))

        assert r == EmbedResult(
            feature_vector=[0.1, 0.2], species="Dog", confidence=0.9,
            bbox=[1.0, 2.0, 3.0, 4.0], isolated_image="CROP_IMG",
            primary_color_hex="#ABCDEF", used_full_frame=False,
        )
        AIManager.download_image.assert_awaited_once_with("http://img/x.jpg")
        AIManager.run_yolo_seg.assert_awaited_once_with("SRC_IMG")
        AIManager.isolate_subject.assert_called_once_with(
            "SRC_IMG", "YOLO_RESULTS", expected_species="Dog"
        )
        AIManager.clip_encode.assert_awaited_once_with("CROP_IMG")  # the crop, not SRC_IMG

    def test_miss_encodes_the_full_frame_and_flags_it(self, monkeypatch):
        self._patch(monkeypatch, iso=None, vector=[0.3, 0.4])

        r = run(AIManager.embed_image("http://img/x.jpg", with_color=True))

        assert r.feature_vector == [0.3, 0.4]
        assert r.used_full_frame is True
        assert r.species is None and r.bbox is None and r.isolated_image is None
        assert r.primary_color_hex is None                 # skipped on the fallback
        AIManager.clip_encode.assert_awaited_once_with("SRC_IMG")  # full frame
        AIManager.extract_coat_color_hex.assert_not_called()

    def test_without_color_skips_the_colour_pass(self, monkeypatch):
        self._patch(monkeypatch, iso=("CROP_IMG", "Cat", 0.7, [0, 0, 1, 1]),
                    vector=[0.5])

        r = run(AIManager.embed_image("http://img/x.jpg"))  # with_color defaults False

        assert r.primary_color_hex is None
        AIManager.extract_coat_color_hex.assert_not_called()

    def test_no_species_constraint_when_expected_is_none(self, monkeypatch):
        self._patch(monkeypatch, iso=("CROP_IMG", "Cat", 0.7, [0, 0, 1, 1]),
                    vector=[0.5])

        run(AIManager.embed_image("http://img/x.jpg"))

        AIManager.isolate_subject.assert_called_once_with(
            "SRC_IMG", "YOLO_RESULTS", expected_species=None
        )


# --------------------------------------------------------------------------- #
# warmup_models — runs both, and swallows a load failure
# --------------------------------------------------------------------------- #
class TestWarmup:
    def test_warms_both_models(self, monkeypatch):
        yolo, clip = MagicMock(), MagicMock()
        monkeypatch.setattr(AIManager, "get_yolo", lambda: yolo)
        monkeypatch.setattr(AIManager, "get_clip", lambda: clip)

        AIManager.warmup_models()                  # must not raise

        yolo.predict.assert_called_once()
        clip.encode.assert_called_once()

    def test_swallows_load_failure(self, monkeypatch):
        def boom():
            raise RuntimeError("weights not found")
        monkeypatch.setattr(AIManager, "get_yolo", boom)

        AIManager.warmup_models()                  # except branch, no raise
