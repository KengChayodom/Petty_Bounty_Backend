"""
In-process identity vectors for missing pets, used to re-rank the CLIP pool.

Why this exists: CLIP (the pgvector column) is trained to match a photo to a
caption, so it scores every cat within about 0.85 to 0.95 and cannot tell one
silver tabby from another. An identity model (DINOv2) is trained on "same
object in two photos", which is the question a sighting asks. The eval in
eval/eval_identity_matching.py put the true cat first in 4 of 4 identity
queries with DINOv2 against 2 of 4 with CLIP.

Why in memory: it adds the second model without a schema change. The vectors
are cheap to rebuild from each pet's image_url, so losing them on restart only
costs a warm-up. Keyed by pet id AND image_url, so a changed photo is never
scored with a stale vector.

Filled three ways:
  * at startup, for every Searching pet (warm_from_repo)
  * at registration, for the new pet (register_missing_pet)
  * at match time, for any pool row still missing one (ensure_vectors)
"""
import asyncio
import logging
from threading import Lock

import numpy as np

from app.services.ai_service import AIManager

logger = logging.getLogger(__name__)

# Enough to overlap downloads without running many YOLO + DINOv2 passes at once.
_FILL_CONCURRENCY = 4


class IdentityIndex:
    _store: dict[str, tuple[str, list[float]]] = {}
    _lock = Lock()

    @classmethod
    def get(cls, pet_id: str, image_url: str | None) -> list[float] | None:
        with cls._lock:
            entry = cls._store.get(str(pet_id))
        if entry is None or entry[0] != str(image_url):
            return None
        return entry[1]

    @classmethod
    def put(cls, pet_id: str, image_url: str, vector: list[float]) -> None:
        with cls._lock:
            cls._store[str(pet_id)] = (str(image_url), vector)

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._store.clear()


async def _fill_one(pet: dict, sem: asyncio.Semaphore) -> None:
    pet_id, url = pet.get("id"), pet.get("image_url")
    if not pet_id or not url or IdentityIndex.get(pet_id, url) is not None:
        return
    async with sem:
        try:
            vec = await AIManager.identity_vector_for(
                str(url), expected_species=pet.get("species"),
            )
        except Exception as e:  # one bad photo must not block the others
            logger.warning("Identity vector failed for pet %s: %s", pet_id, e)
            return
    IdentityIndex.put(pet_id, url, vec)


async def ensure_vectors(pets: list[dict]) -> None:
    """Compute and store identity vectors for any pet in `pets` lacking one.
    Each dict needs `id`, `image_url` and optionally `species`."""
    sem = asyncio.Semaphore(_FILL_CONCURRENCY)
    await asyncio.gather(*(_fill_one(p, sem) for p in pets))


async def warm_from_repo(repo, page_size: int = 100) -> int:
    """Fill vectors for every Searching pet. Returns how many are stored."""
    offset, pets = 0, []
    while True:
        page = await asyncio.to_thread(
            repo.list_all, "Searching", None, page_size, offset,
        )
        pets.extend(page.items)
        offset += page_size
        if len(page.items) < page_size:
            break
    await ensure_vectors(pets)
    ready = sum(IdentityIndex.get(p.get("id"), p.get("image_url")) is not None
                for p in pets)
    # WARNING so it shows in the uvicorn console, like the AnalyzeCache logs.
    logger.warning("Identity index warm: %d of %d Searching pets", ready, len(pets))
    return ready


def identity_similarities(
    sighting_vector: list[float], pool: list[dict],
) -> dict[str, float]:
    """Cosine similarity (vectors are unit length) between the sighting and
    each pool row that has a stored vector. Rows without one are left out."""
    q = np.asarray(sighting_vector, dtype=np.float32)
    sims = {}
    for row in pool:
        vec = IdentityIndex.get(row.get("id"), row.get("image_url"))
        if vec is not None:
            sims[str(row["id"])] = float(q @ np.asarray(vec, dtype=np.float32))
    return sims
