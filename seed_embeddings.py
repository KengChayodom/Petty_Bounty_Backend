"""
seed_embeddings.py — backfill missing_pets.feature_vector using the SAME
mask-isolated CLIP pipeline as the live sighting path
(app/services/ai_service.py::AIManager.isolate_subject).

Why it must match:
    pgvector similarity is only meaningful if seed vectors and sighting
    vectors come from comparable inputs. The live path runs YOLO-seg, blacks
    out the background using the segmentation mask, and tight-crops to the
    mask bbox before CLIP encoding. The seed must do the same — otherwise
    seed embeddings carry background that sighting embeddings don't, and
    similarity scores degrade.

Pipeline per pet: `AIManager.embed_image(image_url, expected_species=...)` — the
exact same download → YOLO-seg → mask-isolate → CLIP-encode path the live
`/sightings/analyze` and `register_missing_pet` run, so seed vectors and
sighting vectors are comparable by construction.

Usage:
    # Default: only pets whose feature_vector is currently NULL.
    python seed_embeddings.py

    # Re-vector every pet (use after changing the encoding pipeline).
    python seed_embeddings.py --all

    # Give pets with no real coat colour the colour measured from their photo:
    # primary_color_hex NULL, or the app's old placeholder #D4AF37, which the
    # form sent whenever the owner never opened the picker. Colours an owner
    # chose are left alone. Vectors are left untouched.
    python seed_embeddings.py --colors
"""
import asyncio
import os
import sys

from dotenv import load_dotenv
from supabase import create_client, Client

from app.services.ai_service import AIManager, EmbedResult

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("⏳ Warm-loading YOLO + CLIP via AIManager…")
AIManager.get_yolo()
AIManager.get_clip()
print("✅ Models loaded.")


def embed_pet(image_url: str, species: str) -> EmbedResult | None:
    """
    The shared pipeline's result, or None if it failed.

    `used_full_frame` on the result is True when YOLO found no matching subject
    and the full image was encoded instead, useful for the run summary. The
    coat colour is extracted with the same call register_missing_pet makes.

    Runs the async `AIManager.embed_image` via `asyncio.run` (one loop per pet,
    fine for a CLI backfill) so this script and the live path share one pipeline.
    """
    try:
        return asyncio.run(AIManager.embed_image(
            image_url, expected_species=species, with_color=True,
        ))
    except Exception as e:
        print(f"   ❌ embed failed for {image_url}: {e}")
        return None


# The create form's default before 2026-09-25. It means "never picked", not gold.
OLD_PLACEHOLDER_COLOR = "#D4AF37"


def backfill(force_all: bool = False, colors_only: bool = False) -> None:
    query = (supabase.table("missing_pets")
                     .select("id, pet_name, species, image_url"))
    if colors_only:
        query = query.or_(f"primary_color_hex.is.null,"
                          f"primary_color_hex.eq.{OLD_PLACEHOLDER_COLOR}")
    elif not force_all:
        query = query.is_("feature_vector", "null")

    pets = query.execute().data
    if not pets:
        scope = ("pets without a chosen coat colour" if colors_only
                 else "any pets" if force_all
                 else "pets with NULL feature_vector")
        print(f"✨ Nothing to do — found no {scope}.")
        return

    mode = ("fill coat colours" if colors_only
            else "RE-VECTOR ALL" if force_all else "fill missing only")
    print(f"\n📦 {len(pets)} pets to process ({mode}).")

    ok = fail = no_detection = 0
    for pet in pets:
        pid, name, species, url = (
            pet["id"], pet["pet_name"], pet["species"], pet["image_url"])
        print(f"\n • {name} ({species}) — {pid}")
        if not url:
            print("   ⚠️  no image_url, skipping")
            fail += 1
            continue

        result = embed_pet(url, species)
        if result is None:
            fail += 1
            continue
        if result.used_full_frame:
            print(f"   ⚠️  YOLO found no {species.lower()} → encoded full"
                  f" frame (recall will degrade for this pet)")
            no_detection += 1
        else:
            print("   ✓ mask-isolated subject encoded")

        if colors_only:
            if not result.primary_color_hex:
                print("   ⚠️  no readable coat colour, left NULL (CLIP only)")
                continue
            changes = {"primary_color_hex": result.primary_color_hex}
        else:
            changes = {"feature_vector": result.feature_vector}
        upd = (supabase.table("missing_pets")
                       .update(changes)
                       .eq("id", pid)
                       .execute())
        if upd.data:
            print(f"   🎯 saved {', '.join(changes)} for {name}")
            ok += 1
        else:
            print(f"   ❌ DB update failed for {name}")
            fail += 1

    print(f"\nSummary — saved: {ok} | failed: {fail} | "
          f"no-YOLO-match (full frame fallback): {no_detection}")


if __name__ == "__main__":
    backfill(force_all="--all" in sys.argv, colors_only="--colors" in sys.argv)
    print("\n🎉 Done.")
