#!/usr/bin/env python3
"""
Seed demo cats across random owners from the database.
Picks 3-5 random users and distributes the demo cats evenly.

Usage:
    python seed_demo_cats_random.py [--dir <folder>] [--dry-run] [--num-owners <N>]
"""
import argparse
import asyncio
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from supabase import create_client

from app.core.config import settings
from seed_demo_cats import (
    DEMO_CATS, DEFAULT_DIR, upload_photo, build_request,
    CENTER_LAT, CENTER_LON
)


async def get_random_owners(num_owners: int = None) -> list[str]:
    """Fetch random users from Supabase."""
    client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

    # Get all users
    users = []
    page = 1
    while True:
        batch = client.auth.admin.list_users(page=page, per_page=1000)
        if not batch:
            break
        users.extend(batch)
        page += 1

    if not users:
        raise SystemExit("❌ No users found in database")

    if num_owners is None:
        num_owners = min(random.randint(3, 5), len(users))
    else:
        num_owners = min(num_owners, len(users))

    selected = random.sample(users, num_owners)
    print(f"📋 Selected {len(selected)} random owners:")
    for user in selected:
        print(f"   • {user.email} ({user.id})")

    return [user.email for user in selected]


async def seed_subset(owner: str, folder: Path, dry_run: bool, cat_subset: list) -> None:
    """Seed only a subset of cats for one owner."""
    missing = [c.file for c in cat_subset if not (folder / c.file).is_file()]
    if missing:
        raise SystemExit(f"❌ Not found in {folder}: {', '.join(missing)}")

    client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

    # Resolve owner ID (same as original seed_demo_cats.py)
    from seed_demo_cats import resolve_owner_id
    owner_id = resolve_owner_id(client, owner)

    # Check which cats already exist
    rows = (client.table("missing_pets")
                  .select("pet_name")
                  .eq("owner_id", owner_id)
                  .execute()).data or []
    already = {r["pet_name"] for r in rows}

    todo = [c for c in cat_subset if c.pet_name not in already]
    for c in cat_subset:
        if c.pet_name in already:
            print(f"   ⏭  {c.pet_name}: already posted, skipped")

    if dry_run:
        now = datetime.now(timezone.utc)
        for c in todo:
            req = build_request(c, owner_id, "<uploaded url>", now)
            print(f"   • {c.pet_name:8} {c.file:12} ฿{c.bounty:>6.0f}")
        print(f"   🧪 Would create: {len(todo)} | Already have: {len(cat_subset) - len(todo)}\n")
        return

    # Real seeding
    from app.services.ai_service import AIManager
    from app.repositories.supabase_missing_pet_repository import (
        SupabaseMissingPetRepository,
    )
    from app.services.pet_service import PetService

    print("   ⏳ Warm-loading YOLO + CLIP…")
    AIManager.get_yolo()
    AIManager.get_clip()

    repo = SupabaseMissingPetRepository(client)
    ok = fail = 0
    now = datetime.now(timezone.utc)
    for c in todo:
        print(f"     {c.pet_name} ({c.file})", end=" ")
        try:
            url = await asyncio.to_thread(upload_photo, client, folder / c.file)
            created = await PetService.register_missing_pet(
                repo, build_request(c, owner_id, url, now)
            )
            print(f"✓ {created['id']}")
            ok += 1
        except Exception as e:
            print(f"✗ {e}")
            fail += 1

    print(f"   Summary: {ok} created | {fail} failed | {len(cat_subset) - len(todo)} skipped\n")


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                        help=f"photo folder (default: {DEFAULT_DIR})")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-owners", type=int, default=None,
                        help="number of owners (default: random 3-5)")
    args = parser.parse_args()

    try:
        owners = await get_random_owners(args.num_owners)

        # Split DEMO_CATS across owners
        cats_per_owner = len(DEMO_CATS) // len(owners)
        remainder = len(DEMO_CATS) % len(owners)

        print(f"\n🐱 Distributing {len(DEMO_CATS)} cats across {len(owners)} owners")
        print(f"   (~{cats_per_owner} cats per owner)\n")

        for i, owner in enumerate(owners):
            start = i * cats_per_owner + min(i, remainder)
            end = start + cats_per_owner + (1 if i < remainder else 0)
            owner_cats = DEMO_CATS[start:end]

            print(f"{'='*60}")
            print(f"Owner {i+1}: {owner}")
            print(f"Cats: {', '.join(c.pet_name for c in owner_cats)}")
            print(f"{'='*60}")

            await seed_subset(owner, args.dir, args.dry_run, owner_cats)

        print(f"✅ Done!")

    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    asyncio.run(main())
