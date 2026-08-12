"""
Short-lived in-memory cache populated by POST /sightings/analyze.

The analyze step runs the heavy AI work (download + YOLO-seg + mask +
CLIP encode) UP FRONT and stores the result here, keyed by `image_url`.
The follow-up POST /sightings/ then only has to: pull the pre-computed
`feature_vector` out of the cache, INSERT the row with the user-confirmed
species, and call the match RPC. Hot path becomes a single DB write.

If the entry expires or the server bounces between calls, POST /sightings/
falls back to re-running the full pipeline — slower, but correct.

Why cachetools.TTLCache, not functools.lru_cache: lru_cache has no TTL,
no eviction on memory pressure, and can't safely hold large numpy/PIL
values across processes. TTLCache evicts on time AND on size, so memory
stays bounded under a flood of unique URLs.
"""
from threading import Lock
from typing import Optional, TypedDict

from cachetools import TTLCache
from PIL import Image


class AnalyzePayload(TypedDict):
    pil_image: Image.Image       # full decoded RGB image (kept for debugging)
    isolated_image: Image.Image  # YOLO-masked + tight-cropped subject
    species: str                 # YOLO's guess ('Cat' | 'Dog' | 'Bird')
    bbox: list[float]            # [x1, y1, x2, y2] in original image coords
    confidence: float            # 0.0–1.0, YOLO confidence on the picked detection
    feature_vector: list[float]  # 512-D CLIP embedding (the heavy artefact)
    primary_color_hex: Optional[str]  # '#RRGGBB' coat colour, or None if unreadable


class AnalyzeCache:
    """Thread-safe TTL cache. Class-level so it survives across requests."""

    _store: TTLCache = TTLCache(maxsize=256, ttl=600)  # 10 minutes
    _lock = Lock()

    @classmethod
    def get(cls, image_url: str) -> Optional[AnalyzePayload]:
        with cls._lock:
            return cls._store.get(image_url)

    @classmethod
    def set(cls, image_url: str, payload: AnalyzePayload) -> None:
        with cls._lock:
            cls._store[image_url] = payload

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._store.clear()
