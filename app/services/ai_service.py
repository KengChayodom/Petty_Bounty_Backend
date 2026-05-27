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

import httpx
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO

logger = logging.getLogger(__name__)


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
