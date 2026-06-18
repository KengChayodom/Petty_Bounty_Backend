"""
Unit tests for the pure / boundary-mockable parts of app/services/ai_service.py:

  * UTC-20  AIManager.isolate_subject  (MD-20) — pure selection + mask + crop
    logic, exercised with a hand-built YOLO results double (real numpy masks +
    a real PIL image). No model is loaded.
  * UTC-21  AIManager.download_image   (MD-18) — httpx is replaced at the
    boundary; HTTP-status and transport errors must propagate.

run_yolo_seg / clip_encode are thin asyncio.to_thread wrappers around the models
(no branch logic) and stay covered by the parity smoke + integration, per plan.
"""
import asyncio
import io

import httpx
import numpy as np
import pytest
from PIL import Image

from app.services import ai_service
from app.services.ai_service import AIManager


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# YOLO results double. Mirrors only the surface isolate_subject reads:
#   results[0].masks.data[i].cpu().numpy()  and
#   results[0].boxes -> iterable of boxes with .cls[0]/.conf[0]
#   results[0].names -> {class_id: label}
# --------------------------------------------------------------------------- #
class _Mask:
    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _Masks:
    def __init__(self, arrays):
        self.data = [_Mask(a) for a in arrays]


class _Box:
    def __init__(self, cls_id, conf):
        self.cls = [cls_id]
        self.conf = [conf]


class _Result:
    def __init__(self, masks, boxes, names):
        self.masks = masks
        self.boxes = boxes
        self.names = names


def _block_mask(size, x0, x1, y0, y1):
    """A (size, size) float mask with a solid 1.0 block in [y0:y1, x0:x1]."""
    m = np.zeros((size, size), dtype=np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


class TestIsolateSubject:
    def test_empty_or_none_results_returns_none(self):
        assert AIManager.isolate_subject(Image.new("RGB", (4, 4)), []) is None
        assert AIManager.isolate_subject(Image.new("RGB", (4, 4)), None) is None

    def test_missing_masks_returns_none(self):
        result = _Result(masks=None, boxes=[_Box(1, 0.9)], names={1: "dog"})
        assert AIManager.isolate_subject(Image.new("RGB", (4, 4)), [result]) is None

    def test_only_non_target_classes_returns_none(self):
        img = Image.new("RGB", (10, 10))
        result = _Result(
            masks=_Masks([_block_mask(10, 1, 5, 1, 5)]),
            boxes=[_Box(0, 0.99)],          # car — not a target animal
            names={0: "car"},
        )
        assert AIManager.isolate_subject(img, [result]) is None

    def test_picks_best_target_and_crops(self):
        size = 30
        # Original image: paint one known pixel inside the dog mask region.
        img = Image.new("RGB", (size, size), (0, 0, 0))
        img.putpixel((16, 16), (200, 100, 50))

        # dog mask occupies cols/rows 15..18 -> with padding 10 and clamping at
        # the image edge this yields bbox [5, 5, 29, 29] and a 24x24 crop.
        masks = _Masks([
            _block_mask(size, 0, 4, 0, 4),     # car  (ignored)
            _block_mask(size, 15, 19, 15, 19), # dog  (best target)
            _block_mask(size, 20, 24, 20, 24), # cat  (lower conf)
        ])
        boxes = [_Box(0, 0.99), _Box(1, 0.80), _Box(2, 0.60)]
        result = _Result(masks, boxes, {0: "car", 1: "dog", 2: "cat"})

        out = AIManager.isolate_subject(img, [result])
        assert out is not None
        isolated, species, confidence, bbox = out
        assert species == "Dog"
        assert confidence == pytest.approx(0.80)
        assert bbox == [5.0, 5.0, 29.0, 29.0]
        assert isolated.size == (24, 24)
        # background pixel (top-left of the crop) blacked out
        assert isolated.getpixel((0, 0)) == (0, 0, 0)
        # the masked pixel at original (16,16) -> crop (11,11) keeps its colour
        assert isolated.getpixel((11, 11)) == (200, 100, 50)

    def test_species_filter_overrides_confidence(self):
        size = 30
        img = Image.new("RGB", (size, size), (0, 0, 0))
        masks = _Masks([
            _block_mask(size, 15, 19, 15, 19),  # dog 0.9
            _block_mask(size, 5, 9, 5, 9),      # cat 0.6
        ])
        boxes = [_Box(0, 0.90), _Box(1, 0.60)]
        result = _Result(masks, boxes, {0: "dog", 1: "cat"})

        out = AIManager.isolate_subject(img, [result], expected_species="Cat")
        assert out is not None
        _isolated, species, confidence, _bbox = out
        assert species == "Cat"
        assert confidence == pytest.approx(0.60)

    def test_filtered_species_absent_returns_none(self):
        size = 30
        img = Image.new("RGB", (size, size), (0, 0, 0))
        masks = _Masks([
            _block_mask(size, 15, 19, 15, 19),
            _block_mask(size, 5, 9, 5, 9),
        ])
        boxes = [_Box(0, 0.90), _Box(1, 0.60)]
        result = _Result(masks, boxes, {0: "dog", 1: "cat"})

        assert AIManager.isolate_subject(img, [result], expected_species="bird") is None


# --------------------------------------------------------------------------- #
# download_image — httpx replaced at the module boundary.
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, content=b"", status_error=None):
        self.content = content
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error:
            raise self._status_error


class _FakeClient:
    def __init__(self, resp=None, get_exc=None):
        self._resp = resp
        self._get_exc = get_exc
        self.requested = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def get(self, url):
        self.requested.append(url)
        if self._get_exc:
            raise self._get_exc
        return self._resp


def _patch_client(monkeypatch, client):
    monkeypatch.setattr(ai_service.httpx, "AsyncClient", lambda *a, **k: client)
    return client


class TestDownloadImage:
    def test_downloads_and_decodes_to_rgb(self, monkeypatch):
        buf = io.BytesIO()
        Image.new("RGB", (7, 11), (10, 20, 30)).save(buf, format="PNG")
        client = _patch_client(
            monkeypatch, _FakeClient(resp=_FakeResp(content=buf.getvalue()))
        )

        url = "https://img.example/x.png"
        img = run(AIManager.download_image(url))

        assert img.mode == "RGB"
        assert img.size == (7, 11)
        assert client.requested == [url]

    def test_http_status_error_propagates(self, monkeypatch):
        err = httpx.HTTPStatusError(
            "404",
            request=httpx.Request("GET", "https://img.example/missing.png"),
            response=httpx.Response(404),
        )
        _patch_client(monkeypatch, _FakeClient(resp=_FakeResp(status_error=err)))

        with pytest.raises(httpx.HTTPError):
            run(AIManager.download_image("https://img.example/missing.png"))

    def test_network_error_propagates(self, monkeypatch):
        _patch_client(
            monkeypatch,
            _FakeClient(get_exc=httpx.ConnectError("connection refused")),
        )

        with pytest.raises(httpx.HTTPError):
            run(AIManager.download_image("https://img.example/x.png"))
