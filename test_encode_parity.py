"""
Encode-parity test.

Purpose: isolate where CLIP embedding drift comes from when comparing a
Flutter-uploaded sighting against a `missing_pets` seed vector.

CLIP (`clip-ViT-B-32`) is deterministic — encoding identical RGB bytes twice
must yield cosine ~1.0. So if cosine drifts away from 1.0 between the seed
vector and a Flutter-uploaded sighting of the SAME source image, the drift
comes from one of three places:

    1. A preprocessing step in the backend (e.g. cropping to a YOLO bbox,
       resizing, color-space change) — every path now runs the single
       `AIManager.embed_image` pipeline, so a mismatch here means that method
       itself changed.
    2. The Flutter client re-encoding / down-sampling the image before
       upload — cosine vs the seed image bytes will land in 0.95-0.99.
    3. An entirely different photo of the same cat (different lighting,
       angle) — cosine typically 0.75-0.95.

Usage:
    # Self-parity: encode the same URL twice. Must print ~1.0.
    python test_encode_parity.py

    # Compare two URLs: e.g. seed image vs the URL the Flutter app uploaded.
    python test_encode_parity.py <seed_url> <flutter_upload_url>

If (a) the backend runs the shared `AIManager.embed_image` pipeline (seed and
live both do) and (b) the Flutter app uploads byte-identical bytes, end-to-end
similarity for the same source image should be ~1.0. A value like 0.87
indicates either (1) or (2).
"""
import io
import sys

import numpy as np
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

# Mochi's seed image (from missing_pets.image_url).
DEFAULT_URL = (
    "https://gleqzpqdoadmtckuegax.supabase.co/storage/v1/object/"
    "public/pet-images/12a0c465-792b-4326-bf3e-2a3de170381a.jpg"
)


def encode(model: SentenceTransformer, url: str) -> np.ndarray:
    """Raw CLIP on the full frame — isolates CLIP determinism from the
    YOLO-isolate step that AIManager.embed_image adds on top."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return np.asarray(model.encode(img), dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    print("Loading CLIP (clip-ViT-B-32)…")
    model = SentenceTransformer("clip-ViT-B-32")

    url_a = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    print(f"URL A: {url_a}")

    v1 = encode(model, url_a)
    v2 = encode(model, url_a)
    self_sim = cosine(v1, v2)
    print(f"  self-cosine (same URL, two passes): {self_sim:.6f}")
    print("  expected ~1.000000 — CLIP is deterministic for identical bytes.")

    if len(sys.argv) > 2:
        url_b = sys.argv[2]
        print(f"URL B: {url_b}")
        v3 = encode(model, url_b)
        sim = cosine(v1, v3)
        print(f"  cosine(A, B): {sim:.6f}")
        if sim >= 0.999:
            print("  → byte-identical encode; Flutter is NOT recompressing.")
        elif sim >= 0.95:
            print("  → tiny drift; consistent with JPEG recompression on upload.")
        else:
            print("  → significant drift; either preprocessing mismatch or"
                  " a different photo of the same subject.")


if __name__ == "__main__":
    main()
