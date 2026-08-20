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

WHAT IS DELIBERATELY *NOT* CACHED
---------------------------------
The full decoded source image. It used to be kept here as `pil_image` "for
debugging", but nothing on the save path ever read it, and a decoded RGB
frame costs width x height x 3 bytes -- ~9.4 MB for the 2048x1536 photos the
app uploads. At the old maxsize of 256 that was a ceiling of roughly 2.8 GB
of live objects, none of it load-bearing. Once the host starts swapping,
EVERY endpoint slows down, not just this one, so the cheapest fix is to not
hold the bytes at all.

What remains is the small stuff: the 512-D vector (the artefact that is
genuinely expensive to recompute), the mask-isolated crop, and three scalars.
"""
from threading import Lock
from typing import Optional, TypedDict

from cachetools import TTLCache
from PIL import Image


class AnalyzePayload(TypedDict):
    isolated_image: Image.Image  # YOLO-masked + tight-cropped subject
    species: str                 # YOLO's guess ('Cat' | 'Dog' | 'Bird')
    bbox: list[float]            # [x1, y1, x2, y2] in original image coords
    confidence: float            # 0.0–1.0, YOLO confidence on the picked detection
    feature_vector: list[float]  # 512-D CLIP embedding (the heavy artefact)
    primary_color_hex: Optional[str]  # '#RRGGBB' coat colour, or None if unreadable


class AnalyzeCache:
    """Thread-safe TTL cache. Class-level so it survives across requests."""

    # 64, not 256: an entry only has to survive the seconds between /analyze
    # and the user confirming on the verify screen, so the size cap is really
    # "how many people are mid-report at once", not a throughput figure.
    _store: TTLCache = TTLCache(maxsize=64, ttl=600)  # 10 minutes
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
