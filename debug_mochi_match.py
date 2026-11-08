"""
debug_mochi_match.py — isolate why a seeded missing_pet (e.g. Mochi) does
not appear in `match_missing_pets` results for a sighting of the same animal.

What this script answers, in order:

    1. Did Mochi's row get a feature_vector at all? (NULL = seed failed.)
    2. Is the seeded vector 512-D and a real float array?
    3. Re-encode Mochi's seed image through the LIVE pipeline (YOLO →
       isolate_subject → CLIP). Cosine vs the stored vector should be
       ~1.0 if the pipeline is deterministic and unchanged since seeding.
       Cosine ~0.85-0.95 = preprocessing drift (the seed almost certainly
       used the full-frame fallback because YOLO missed the cat on seed
       day — re-run seed_embeddings.py --all to fix).
    4. If you pass a sighting image URL as argv[1], also cosine that
       against Mochi's seed. <0.7 = different photo / different cat /
       major drift; >0.85 = strong match the RPC should be returning.
    5. Print the species + status + distance-to-sighting (if argv[2,3]
       are lat lon) so you can confirm the SQL filter (`LOWER(species)`,
       `status='Searching'`, `ST_DWithin(..., 10000)`) is not what's
       excluding her.

Run from project root with .env loaded:
    python debug_mochi_match.py
    python debug_mochi_match.py <sighting_image_url>
    python debug_mochi_match.py <sighting_image_url> <lat> <lon>
"""
import io
import os
import sys

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image
from supabase import create_client

from app.services.ai_service import AIManager

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    sys.exit("❌ Missing SUPABASE_URL / SUPABASE_SERVICE_KEY in .env")

PET_NAME_QUERY = os.getenv("DEBUG_PET_NAME", "Mochi")
RADIUS_METERS = 10_000  # mirrors ST_DWithin(..., 10000) in match_missing_pets


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def download_pil(url: str) -> Image.Image:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def encode_live_pipeline(image_url: str, species_hint: str | None = None) -> tuple[np.ndarray, bool]:
    """Re-run exactly what the live /sightings/analyze does. Returns
    (vector, used_full_frame). used_full_frame=True means YOLO missed the
    subject and we fell back to encoding the whole frame."""
    img = download_pil(image_url)
    yolo = AIManager.get_yolo()
    results = yolo.predict(source=img, conf=0.25, verbose=False)
    iso = AIManager.isolate_subject(img, results, expected_species=species_hint)
    if iso is None:
        target = img
        used_full_frame = True
    else:
        target = iso[0]
        used_full_frame = False
    clip = AIManager.get_clip()
    vec = np.asarray(clip.encode(target), dtype=np.float32)
    return vec, used_full_frame


def parse_postgis_point(s: str) -> tuple[float, float] | None:
    """Parse 'POINT(lon lat)' or 'SRID=4326;POINT(lon lat)' to (lat, lon)."""
    if not s:
        return None
    try:
        coords = (s.replace("SRID=4326;", "")
                   .replace("POINT(", "")
                   .replace(")", "")
                   .split())
        return float(coords[1]), float(coords[0])
    except Exception:
        return None


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    import math
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def main() -> None:
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"🔎 Looking up pet named ~{PET_NAME_QUERY!r}…")
    rows = (supa.table("missing_pets")
                .select("id, pet_name, species, status, image_url, "
                        "feature_vector, last_seen_location")
                .ilike("pet_name", f"%{PET_NAME_QUERY}%")
                .execute().data)
    if not rows:
        sys.exit(f"❌ No pet matches name {PET_NAME_QUERY!r}.")
    if len(rows) > 1:
        print(f"⚠️  {len(rows)} matches — using the first.")
    pet = rows[0]
    print(f"✓ {pet['pet_name']} ({pet['species']}) — id={pet['id']}")
    print(f"  status={pet['status']!r}  image_url={pet['image_url']}")

    # ---- 1. seed-vector sanity --------------------------------------------
    vec = pet.get("feature_vector")
    if vec is None:
        sys.exit("❌ feature_vector is NULL — seed never ran or failed. "
                 "Run `python seed_embeddings.py --all` and retry.")
    seed = np.asarray(vec, dtype=np.float32)
    print(f"\n📦 Seed vector: dim={seed.shape[0]} norm={np.linalg.norm(seed):.4f}")
    if seed.shape[0] != 512:
        sys.exit(f"❌ Seed vector has unexpected dim {seed.shape[0]} "
                 f"(expected 512). The DB has a mixed-pipeline row.")

    # ---- 2. SQL-filter pre-flight -----------------------------------------
    if pet["status"] != "Searching":
        print(f"⚠️  status is {pet['status']!r} — match_missing_pets "
              f"only returns rows with status='Searching'. This pet would "
              f"NOT appear in matches regardless of similarity.")

    # ---- 3. Re-encode the SEED image through the live pipeline -----------
    print("\n🔁 Re-encoding seed image via LIVE pipeline (YOLO + CLIP)…")
    AIManager.get_yolo(); AIManager.get_clip()  # warm-load
    live_seed_vec, used_full_frame = encode_live_pipeline(
        pet["image_url"], species_hint=pet["species"]
    )
    sim_seed = cosine(seed, live_seed_vec)
    print(f"  cosine(stored, live-re-encode of same image) = {sim_seed:.6f}")
    if used_full_frame:
        print("  ⚠️  YOLO did NOT find the animal in the seed image — live "
              "path fell back to full frame. If the original seed was "
              "mask-isolated, that's the drift.")
    if sim_seed >= 0.999:
        print("  ✅ identical embeddings — pipeline is stable and "
              "deterministic for this pet.")
    elif sim_seed >= 0.95:
        print("  🟡 minor drift — most likely the seed used the full-frame "
              "fallback while the live path mask-isolates (or vice-versa). "
              "Re-run `python seed_embeddings.py --all` to re-vector Mochi.")
    else:
        print("  🔴 large drift — preprocessing OR model change between "
              "seed time and now. Re-seed and investigate.")

    # ---- 4. Optional: compare against a sighting image URL ----------------
    if len(sys.argv) > 1:
        sighting_url = sys.argv[1]
        print(f"\n🆚 Encoding sighting image: {sighting_url}")
        sighting_vec, sighting_full_frame = encode_live_pipeline(
            sighting_url, species_hint=pet["species"]
        )
        sim_sighting = cosine(seed, sighting_vec)
        print(f"  cosine(stored seed, sighting) = {sim_sighting:.6f}")
        if sighting_full_frame:
            print("  ⚠️  YOLO did not detect the animal in the SIGHTING image "
                  "either — sighting path also fell back to full frame.")
        if sim_sighting >= 0.85:
            print("  ✅ strong match — RPC should be returning this. If it "
                  "isn't, the SQL filter is excluding it (species/status/"
                  "radius). See section 5.")
        elif sim_sighting >= 0.7:
            print("  🟡 borderline — your threshold may be too high.")
        else:
            print("  🔴 weak — likely different photo / different animal "
                  "OR preprocessing drift.")

    # ---- 5. Optional: distance gate check ---------------------------------
    if len(sys.argv) > 3:
        try:
            lat, lon = float(sys.argv[2]), float(sys.argv[3])
        except ValueError:
            print("(skip distance check — couldn't parse lat/lon args)")
            return
        loc = parse_postgis_point(pet["last_seen_location"] or "")
        if loc is None:
            print("⚠️  Could not parse pet last_seen_location — skipping "
                  "distance check.")
            return
        dm = haversine_m(loc, (lat, lon))
        print(f"\n📍 Distance from sighting ({lat:.5f}, {lon:.5f}) to "
              f"{pet['pet_name']} ({loc[0]:.5f}, {loc[1]:.5f}) = {dm:.0f} m")
        if dm > RADIUS_METERS:
            print(f"  🔴 OUTSIDE {RADIUS_METERS} m — match_missing_pets' "
                  f"hardcoded ST_DWithin will exclude this pet regardless "
                  f"of similarity. Re-seed Mochi closer, or relax the SQL.")
        else:
            print(f"  ✅ within {RADIUS_METERS} m — the spatial filter is "
                  f"not what's hiding her.")


if __name__ == "__main__":
    main()
