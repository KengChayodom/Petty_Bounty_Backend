"""
Seed demo missing-pet reports from the cat photos in a local folder.

Every photo in DEMO_CATS becomes one missing_pets row with status 'Searching',
created through the SAME path the app uses (`PetService.register_missing_pet`):
upload to the `pet-images` bucket, then YOLO-seg + CLIP embedding, then
INSERT. Seed vectors and live sighting vectors are therefore comparable
by construction (see "seed vs live pipeline parity" in CLAUDE.md).

Photos whose file name contains "Test" are never seeded. They are kept back
as sighting photos for testing the matching flow:
    เทาTest.jpg  → should match "Silver" (เทา2.jpg), same cat
    ขาวTest.jpg  → white Persian, compare against "Pearl" (ขาว.jpg)
    ส้มTest.jpg  → orange tabby, compare against the orange cats

Leo1.jpg and Leo2.jpg are the same cat, so only Leo1 is posted. Leo2 is
left over and works as another test sighting for "Leo".

No push notifications are sent (the route's background task is not called),
so seeding does not alert every hunter within 10 km of each pet.

Usage:
    python seed_demo_cats.py --owner <email-or-user-uuid> [--dir <folder>] [--dry-run]

    --owner    account that owns the reports. Use a different account from
               the one you test sightings with.
    --dir      folder with the photos (default: ~/Downloads/แมว)
    --dry-run  print what would be seeded without uploading or inserting.

Re-running is safe: a pet whose name this owner has already posted is skipped.

Requires SUPABASE_URL and SUPABASE_SERVICE_KEY in the environment / .env.
"""
import argparse
import asyncio
import mimetypes
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from supabase import Client, create_client

from app.core.config import settings
from app.repositories.supabase_missing_pet_repository import (
    SupabaseMissingPetRepository,
)
from app.schemas.missing_pets import MissingPetCreate
from app.services.ai_service import AIManager
from app.services.pet_service import PetService

BUCKET = "pet-images"
DEFAULT_DIR = Path.home() / "Downloads" / "แมว"

# Centre of the demo area: CAMT, Chiang Mai University. Every pet sits within
# roughly 4 km of it, well inside the 10 km search radius, so all of them show
# on the map and are candidates for matching from the same test location.
CENTER_LAT, CENTER_LON = 18.7963, 98.9530


@dataclass(frozen=True)
class DemoCat:
    file: str
    pet_name: str
    traits: str
    bounty: float                # THB; 0 means no bounty
    d_lat: float                 # offset from CENTER, degrees
    d_lon: float
    hours_ago: int               # last seen this many hours before seeding
    secondary_hex: str | None = None


# No coat colour is sent, like an owner who keeps the app's default: the app's
# default is the colour measured from the photo, and register_missing_pet
# stores that same measurement when none is sent. A hand-typed hex here is how
# Kaprao (a grey tabby) once got a brownish #8A7560 and surfaced for orange
# searches. secondary_hex is only a descriptive trait and does not feed matching.
DEMO_CATS: list[DemoCat] = [
    DemoCat("เทา2.jpg", "Silver",
            "Silver classic tabby (American Shorthair type), yellow eyes, "
            "red collar with red bell and red leash",
            2000, 0.0042, 0.0061, 20, secondary_hex="#3C3C3C"),
    DemoCat("เทา1.jpg", "Marble",
            "Silver classic tabby with bullseye pattern on the sides, "
            "white paws, yellow eyes",
            1500, -0.0105, 0.0120, 44, secondary_hex="#2F2F2F"),
    DemoCat("เทา3.jpg", "Kaprao",
            "Brown mackerel tabby, large ears, yellow-green eyes, "
            "pastel beaded collar",
            500, 0.0150, -0.0080, 30),
    DemoCat("เทา4.jpg", "Pepper",
            "Silver tabby kitten with dark stripes, bushy tail, black collar",
            800, -0.0060, -0.0140, 12, secondary_hex="#3A3A3A"),
    DemoCat("เทา5.jpg", "Moon",
            "Pale cream-grey shorthair, stocky build, pink paw pads",
            0, 0.0210, 0.0035, 60),
    DemoCat("ลายเทา.jpg", "Smokey",
            "Light grey mackerel tabby with beige undertone, yellow eyes, "
            "fluffy dark-tipped tail",
            1000, -0.0180, -0.0025, 26),
    DemoCat("ขาว.jpg", "Pearl",
            "Long-haired white Persian, flat face, amber eyes",
            3000, 0.0090, 0.0190, 8),
    DemoCat("ขาว1.jpg", "Cotton",
            "White Scottish Fold kitten, folded ears, dark blue collar "
            "with silver bell",
            2500, -0.0030, 0.0240, 36),
    DemoCat("ขาว2.jpg", "Snow",
            "White British Shorthair, round face, copper eyes",
            2000, 0.0260, -0.0170, 52),
    DemoCat("ขาว3.jpg", "Salapao",
            "White cat with grey tabby patches on the head, grey ringed tail, "
            "teal collar with blue bell",
            700, -0.0230, 0.0160, 18, secondary_hex="#6E6A66"),
    DemoCat("ขาว4.jpg", "Oreo",
            "Long-haired white cat with black head cap, black saddle patch "
            "and black tail, copper eyes",
            1500, 0.0120, -0.0250, 40, secondary_hex="#1E1E1E"),
    DemoCat("ดำ1.jpg", "Domino",
            "Black and white tuxedo, white chest bib and paws, small black "
            "spot on the chin, yellow eyes",
            1000, -0.0290, -0.0110, 70, secondary_hex="#F5F5F5"),
    DemoCat("ส้ม.jpg", "Pumpkin",
            "Orange classic tabby, amber eyes, ringed stripes on the chest",
            1200, 0.0055, -0.0045, 16),
    DemoCat("ส้ม1.jpg", "Peach",
            "Orange kitten, amber eyes, pink collar with pink bell",
            1800, -0.0140, 0.0045, 10),
    DemoCat("ส้ม2.jpg", "Tiger",
            "Adult orange tabby, lean build, red bell on the collar",
            600, 0.0310, 0.0090, 90),
    DemoCat("ส้ม3.jpg", "Mango",
            "Orange and white tabby, white chest and legs, silver bell collar",
            900, -0.0075, 0.0300, 28, secondary_hex="#FFFFFF"),
    DemoCat("ส้ม4.jpg", "Khanom",
            "Very small orange tabby kitten, blue-grey eyes",
            500, 0.0180, 0.0260, 6),
    DemoCat("ส้มถ.jpg", "Tangmo",
            "Young orange tabby kitten, white chest and white paws",
            0, -0.0330, 0.0020, 48, secondary_hex="#FFFFFF"),
    DemoCat("Leo1.jpg", "Leo",
            "Cream long-haired Persian, flat face, copper eyes, "
            "fluffy plume tail",
            3000, 0.0015, -0.0300, 22),
]


def resolve_owner_id(client: Client, owner: str) -> str:
    """Accept a user UUID as-is, or look an email up in Supabase Auth."""
    try:
        return str(uuid.UUID(owner))
    except ValueError:
        pass
    page = 1
    while True:
        users = client.auth.admin.list_users(page=page, per_page=1000)
        if not users:
            break
        for user in users:
            if (user.email or "").lower() == owner.lower():
                return user.id
        page += 1
    raise SystemExit(f"❌ No auth user with email {owner}")


def existing_pet_names(client: Client, owner_id: str) -> set[str]:
    rows = (client.table("missing_pets")
                  .select("pet_name")
                  .eq("owner_id", owner_id)
                  .execute()).data or []
    return {r["pet_name"] for r in rows}


def upload_photo(client: Client, path: Path) -> str:
    """Upload like POST /upload/pet-image does and return the public URL."""
    content_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    object_name = f"{uuid.uuid4()}{path.suffix.lower()}"
    client.storage.from_(BUCKET).upload(
        path=object_name,
        file=path.read_bytes(),
        file_options={"content-type": content_type},
    )
    return client.storage.from_(BUCKET).get_public_url(object_name)


def build_request(cat: DemoCat, owner_id: str, image_url: str,
                  now: datetime) -> MissingPetCreate:
    # The secondary colour key is the one the edit screen writes for two-tone
    # coats. There is no primary colour: registration measures it.
    characteristics = {"traits": cat.traits}
    if cat.secondary_hex:
        characteristics["secondary_color"] = cat.secondary_hex
    return MissingPetCreate(
        owner_id=owner_id,
        pet_name=cat.pet_name,
        species="Cat",
        characteristics=characteristics,
        bounty_amount=cat.bounty,
        latitude=CENTER_LAT + cat.d_lat,
        longitude=CENTER_LON + cat.d_lon,
        last_seen_time=now - timedelta(hours=cat.hours_ago),
        image_url=image_url,
    )


async def seed(owner: str, folder: Path, dry_run: bool) -> None:
    missing = [c.file for c in DEMO_CATS if not (folder / c.file).is_file()]
    if missing:
        raise SystemExit(f"❌ Not found in {folder}: {', '.join(missing)}")
    if any("test" in c.file.lower() for c in DEMO_CATS):
        raise SystemExit("❌ A *Test* photo is in DEMO_CATS; those are "
                         "reserved for sighting tests")

    client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
    owner_id = resolve_owner_id(client, owner)
    already = existing_pet_names(client, owner_id)
    now = datetime.now(timezone.utc)
    print(f"👤 Owner {owner} -> {owner_id}")

    todo = [c for c in DEMO_CATS if c.pet_name not in already]
    for c in DEMO_CATS:
        if c.pet_name in already:
            print(f" ⏭  {c.pet_name}: already posted by this owner, skipped")

    if dry_run:
        for c in todo:
            req = build_request(c, owner_id, "<uploaded url>", now)
            print(f" • {c.pet_name:8} {c.file:12} "
                  f"฿{c.bounty:>6.0f}  ({req.latitude:.4f}, {req.longitude:.4f})")
        print(f"\n🧪 Dry run: {len(todo)} reports would be created.")
        return

    print("⏳ Warm-loading YOLO + CLIP…")
    AIManager.get_yolo()
    AIManager.get_clip()

    repo = SupabaseMissingPetRepository(client)
    ok = fail = 0
    for c in todo:
        print(f"\n • {c.pet_name} ({c.file})")
        try:
            url = await asyncio.to_thread(upload_photo, client, folder / c.file)
            created = await PetService.register_missing_pet(
                repo, build_request(c, owner_id, url, now)
            )
            print(f"   🎯 created {created['id']}")
            ok += 1
        except Exception as e:
            print(f"   ❌ {e}")
            fail += 1

    print(f"\nSummary: created {ok} | failed {fail} | "
          f"skipped {len(DEMO_CATS) - len(todo)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--owner", required=True,
                        help="owner's email or user UUID")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                        help=f"photo folder (default: {DEFAULT_DIR})")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(seed(args.owner, args.dir, args.dry_run))
    except KeyboardInterrupt:
        sys.exit(130)
