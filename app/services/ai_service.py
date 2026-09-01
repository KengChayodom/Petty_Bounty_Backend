"""
AI service for managing YOLO and CLIP models.

Pipeline: download → YOLO-seg → mask-isolate subject → CLIP-encode.

**Backend**: pure PyTorch. The ONNX path was removed because the local
`./clip-onnx` export only covers the text encoder, which makes the
image-tower `get_image_features` call raise and 500 the `/analyze`
endpoint. PyTorch is the stable, supported path for sentence-transformers
image encoding today. The `yolo11s-seg.onnx` and `./clip-onnx` artefacts
on disk are now unused — keep or delete at your discretion.
"""
import asyncio
import io
import logging
from dataclasses import dataclass

import httpx
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class EmbedResult:
    """The output of the one shared embed pipeline (`AIManager.embed_image`).

    On a YOLO hit every field is populated. On a YOLO miss the full frame is
    encoded instead: `feature_vector` still holds a vector, `used_full_frame`
    is True, and `species` / `confidence` / `bbox` / `isolated_image` /
    `primary_color_hex` are all None. A caller that must refuse a miss checks
    `used_full_frame`.
    """
    feature_vector: list[float]
    species: str | None = None
    confidence: float | None = None
    bbox: list[float] | None = None
    isolated_image: Image.Image | None = None
    primary_color_hex: str | None = None
    used_full_frame: bool = False


class AIManager:
    """
    Manager for YOLO + CLIP with lazy loading and class-level caching.
    `app/main.py` warm-loads both via the FastAPI lifespan hook so the
    first user request doesn't pay the model-load cost.
    """

    _yolo = None
    _clip = None

    # PyTorch weights — single source of truth.
    YOLO_WEIGHTS = "yolo11s-seg.pt"
    CLIP_MODEL = "clip-ViT-B-32"  # resolved via sentence-transformers hub

    # Same set as SightingService.TARGET_ANIMALS — keep in sync.
    TARGET_ANIMALS = {"cat", "dog", "bird"}
    MASK_PADDING = 10  # px — applied around the mask bbox before CLIP encode

    @classmethod
    def get_yolo(cls):
        """Get or load the YOLO model for segmentation."""
        if cls._yolo is None:
            logger.info("Loading YOLO model (%s)…", cls.YOLO_WEIGHTS)
            cls._yolo = YOLO(cls.YOLO_WEIGHTS)
        return cls._yolo

    @classmethod
    def get_clip(cls):
        """Get or load the CLIP model for feature extraction."""
        if cls._clip is None:
            logger.info("Loading CLIP model (%s, backend=pytorch)…", cls.CLIP_MODEL)
            cls._clip = SentenceTransformer(cls.CLIP_MODEL)
        return cls._clip

    # --- Pipeline helpers (in-memory, single-responsibility) -----------------

    @classmethod
    async def download_image(cls, image_url: str) -> Image.Image:
        """Download once via httpx and decode to PIL RGB."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(image_url)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")

    @classmethod
    async def run_yolo_seg(cls, image: Image.Image, conf: float = 0.25):
        """
        Run YOLO-seg directly on an in-memory PIL image — no second HTTP
        fetch. Wrapped in `asyncio.to_thread` so the synchronous Ultralytics
        call doesn't block the FastAPI event loop.
        """
        model = cls.get_yolo()
        return await asyncio.to_thread(
            model.predict, source=image, conf=conf, verbose=False
        )

    @classmethod
    def isolate_subject(
        cls,
        image: Image.Image,
        results,
        expected_species: str | None = None,
    ) -> tuple[Image.Image, str, float, list[float]] | None:
        """
        Use YOLO-seg masks to isolate the highest-confidence target animal.

        Steps:
            1. Pick the best-confidence detection whose class is in
               TARGET_ANIMALS (optionally constrained to `expected_species`).
            2. Resize its mask to the image's original resolution.
            3. Black out every non-mask pixel.
            4. Tight-crop to the mask bbox + MASK_PADDING.

        Returns (isolated_image, species_capitalised, confidence, bbox) or
        None if no usable detection exists. The returned image is what
        CLIP should encode.
        """
        if not results:
            return None
        result = results[0]
        if result.masks is None or result.boxes is None:
            return None

        target = expected_species.lower() if expected_species else None

        best_idx: int | None = None
        best_conf = 0.0
        best_cls: str | None = None
        for i, box in enumerate(result.boxes):
            cls_name = result.names[int(box.cls[0])].lower()
            if cls_name not in cls.TARGET_ANIMALS:
                continue
            if target and cls_name != target:
                continue
            c = float(box.conf[0])
            if c > best_conf:
                best_idx, best_conf, best_cls = i, c, cls_name

        if best_idx is None or best_cls is None:
            return None

        # Resize mask to original image dims if needed (YOLO masks come at the
        # network's input resolution, typically 640×640).
        mask = result.masks.data[best_idx].cpu().numpy()
        img_w, img_h = image.size
        if mask.shape != (img_h, img_w):
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(
                (img_w, img_h), Image.NEAREST
            )
            mask_bool = np.asarray(mask_pil) > 127
        else:
            mask_bool = mask > 0.5

        # Black out background, then tight-crop to the mask bbox + padding.
        arr = np.array(image)
        arr[~mask_bool] = 0
        ys, xs = np.where(mask_bool)
        if len(xs) == 0 or len(ys) == 0:
            return None

        p = cls.MASK_PADDING
        x1 = max(0, int(xs.min()) - p)
        y1 = max(0, int(ys.min()) - p)
        x2 = min(img_w, int(xs.max()) + 1 + p)
        y2 = min(img_h, int(ys.max()) + 1 + p)

        isolated = Image.fromarray(arr[y1:y2, x1:x2])
        bbox = [float(x1), float(y1), float(x2), float(y2)]
        return isolated, best_cls.capitalize(), best_conf, bbox

    @classmethod
    async def clip_encode(cls, image: Image.Image) -> list[float]:
        """CLIP-encode a (typically mask-isolated) PIL image to 512-D."""
        model = cls.get_clip()
        vector = await asyncio.to_thread(model.encode, image)
        return vector.tolist()

    @classmethod
    async def embed_image(
        cls,
        image_url: str,
        *,
        expected_species: str | None = None,
        with_color: bool = False,
    ) -> EmbedResult:
        """Download → YOLO-seg → mask-isolate → CLIP-encode — the ONE pipeline
        every caller shares (`register_missing_pet`, `analyze_sighting_image`,
        the `process_and_save_sighting` cache-miss re-run, and
        `seed_embeddings.py`), so seed and live vectors stay comparable by
        construction rather than by a parity test.

        `expected_species` constrains subject selection to that one species; pass
        None for the "forgiving" behaviour (highest-confidence target animal).
        `with_color` also runs the coat-colour extraction, which is only
        meaningful on a genuine isolated crop and is skipped on a full-frame
        fallback.

        On a YOLO miss the full frame is CLIP-encoded so a vector always exists
        (`used_full_frame=True`); a caller that must refuse a miss inspects that
        flag itself — this method never raises for "no animal found".
        """
        image = await cls.download_image(image_url)
        results = await cls.run_yolo_seg(image)
        iso = await asyncio.to_thread(
            cls.isolate_subject, image, results,
            expected_species=expected_species,
        )
        if iso is None:
            return EmbedResult(
                feature_vector=await cls.clip_encode(image),
                used_full_frame=True,
            )
        isolated, species, confidence, bbox = iso
        return EmbedResult(
            feature_vector=await cls.clip_encode(isolated),
            species=species,
            confidence=confidence,
            bbox=bbox,
            isolated_image=isolated,
            primary_color_hex=(
                await asyncio.to_thread(cls.extract_coat_color_hex, isolated)
                if with_color else None
            ),
            used_full_frame=False,
        )

    # Coat-colour extraction tunables. A mask-isolated crop has a BLACK
    # background, so foreground = pixels bright enough to not be that black
    # fill. A near-black subject (or a full-frame fallback with no mask) has
    # too few foreground pixels to trust → return None and let matching fall
    # back to CLIP-only rather than reporting the background colour.
    COLOR_BLACK_THRESHOLD = 15   # max-channel value below which a px is "background"
    COLOR_MIN_PIXELS = 50        # need at least this many fg px to call a colour

    @classmethod
    def extract_coat_color_hex(cls, isolated_image: Image.Image) -> str | None:
        """Median coat colour of a mask-isolated crop as '#RRGGBB', or None.

        Operates on the black-background crop returned by `isolate_subject`:
        drops near-black background pixels, then takes the per-channel MEDIAN of
        what's left (median, not mean, so eyes/collar/white patches don't drag
        the estimate). None when too few foreground pixels survive — which is
        exactly the near-black-subject / full-frame-fallback case where any
        colour reading would be the background, not the animal.

        MUST be called only on a genuine isolated crop (iso is not None). On the
        full-frame fallback the "background" is the real scene, so its pixels
        are not black and this would happily return the wall colour.
        """
        arr = np.asarray(isolated_image.convert("RGB")).reshape(-1, 3)
        foreground = arr[arr.max(axis=1) > cls.COLOR_BLACK_THRESHOLD]
        if foreground.shape[0] < cls.COLOR_MIN_PIXELS:
            return None
        r, g, b = (int(round(float(c))) for c in np.median(foreground, axis=0))
        return f"#{r:02X}{g:02X}{b:02X}"

    @classmethod
    def warmup_models(cls):
        """
        Warm up YOLO and CLIP models by forcing an initial inference pass.
        This triggers PyTorch internal lazy loading during application startup.
        """
        logger.info("Starting AI models warm-up pass...")
        try:
            # 1. สร้างรูปภาพสีดำเปล่าๆ ขนาด 224x224 (ขนาดมาตรฐานที่โมเดลใช้ประมวลผล)
            blank_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

            # 2. หลอกรัน YOLO 1 ครั้ง
            yolo_model = cls.get_yolo()
            yolo_model.predict(source=blank_image, conf=0.25, verbose=False)
            logger.info("  YOLO model warm-up complete.")

            # 3. หลอกรัน CLIP 1 ครั้ง (จุดนี้แหละที่จะทำลายกำแพง 4 วินาทีทิ้งไป)
            clip_model = cls.get_clip()
            clip_model.encode(blank_image)
            logger.info("  CLIP model warm-up complete.")

            logger.info("All AI models warmed up successfully!")
        except Exception as e:
            logger.error("Failed to warm up AI models: %s", e)
