"""Pure, I/O-free logic extracted from PetService."""
from app.utils.postgis import create_postgis_point

# The three states a report is shown in (decided 2026-08-17). They are DERIVED,
# never stored: `missing_pets.status` says only whether the search is still open,
# and the rest comes from counting the sightings recorded against the pet.
#
# Deriving rather than storing is what keeps the badge honest when a sighting
# goes away — an admin dismissing a bogus sighting drops the count, so a report
# whose only sighting was bogus falls back to PENDING by itself. A stored badge
# would have to be recomputed at every place a sighting can disappear, and would
# quietly claim "someone has seen your pet" the first time one was missed.
POST_STATUS_PENDING = "Pending"    # search open, nobody has reported a sighting
POST_STATUS_SPOTTED = "Spotted"    # search open, at least one sighting exists
POST_STATUS_RESCUED = "Rescued"    # the owner closed the search

# `pet_status` values that mean the search is over. 'Resolved' is written by the
# resolve RPC; 'Found' is what the owner's End Search button writes.
_CLOSED_PET_STATUSES = frozenset({"found", "resolved"})

# `sighting_matches.owner_status` — the owner's verdict on one AI match. A
# rejected match is not a sighting of this pet.
OWNER_REJECTED = "Rejected"


def build_missing_pet_payload(pet, *, feature_vector) -> dict:
    """The missing_pets INSERT payload — location as a PostGIS POINT, status
    seeded to 'Searching', and the CLIP feature vector attached."""
    location_point = create_postgis_point(pet.latitude, pet.longitude)
    return {
        "owner_id": pet.owner_id,
        "pet_name": pet.pet_name,
        "species": pet.species,
        "characteristics": pet.characteristics,
        "bounty_amount": pet.bounty_amount,
        "last_seen_location": location_point,
        "last_seen_time": pet.last_seen_time.isoformat(),
        "image_url": str(pet.image_url),
        "feature_vector": feature_vector,
        "status": "Searching",
        "primary_color_hex": pet.primary_color_hex,
        "pattern_id": pet.pattern_id,
    }


def derive_post_status(pet_status: str | None, sighting_count: int) -> str:
    """The badge shown on an owner's report card.

    A closed search wins over everything: a recovered pet reads RESCUED even
    though sightings were reported along the way — those sightings are how it
    got home, not an argument that it is still missing.
    """
    if (pet_status or "").strip().lower() in _CLOSED_PET_STATUSES:
        return POST_STATUS_RESCUED
    return POST_STATUS_SPOTTED if sighting_count > 0 else POST_STATUS_PENDING


def count_sightings(links: list[dict]) -> dict[str, int]:
    """How many sightings count towards each pet, from the raw link rows.

    Two rules, both product decisions rather than storage details, which is why
    they live here instead of in the query:

    * **One sighting counts once.** A hunter can report a pet from its detail
      page AND have the photo match it, producing a row from each source; that
      is one person having seen the pet once.
    * **A match the owner Rejected does not count.** They told us it is not
      their pet — continuing to count it would leave the report reading
      "someone has seen your pet" on the strength of a sighting they have
      already dismissed.
    """
    seen: dict[str, set[str]] = {}
    for link in links:
        pet_id = link.get("pet_id")
        sighting_id = link.get("sighting_id")
        if not pet_id or not sighting_id:
            continue
        if link.get("owner_status") == OWNER_REJECTED:
            continue
        seen.setdefault(pet_id, set()).add(sighting_id)
    return {pet_id: len(ids) for pet_id, ids in seen.items()}


def attach_sighting_counts(
    pets: list[dict], counts: dict[str, int]
) -> list[dict]:
    """Add `sighting_count` and `post_status` to each report.

    A pet absent from `counts` has had no sightings — it must read 0, not raise
    and not go missing from the list, because "nobody has reported anything" is
    the ordinary state of a fresh report rather than an error.

    Returns new dicts; the input rows are not mutated.
    """
    enriched = []
    for pet in pets:
        count = counts.get(pet.get("id"), 0)
        enriched.append({
            **pet,
            "sighting_count": count,
            "post_status": derive_post_status(pet.get("status"), count),
        })
    return enriched
