"""Pure, I/O-free logic extracted from PetService."""
from datetime import datetime, timezone

from app.utils.postgis import create_postgis_point

# The four states a report is shown in (three decided 2026-08-17, EXPIRED added
# 2026-08-21). They are DERIVED, never stored: `missing_pets.status` says only
# whether the search is still open, and the rest comes from the report's own
# `expires_at` and from counting the sightings recorded against the pet.
#
# Deriving rather than storing is what keeps the badge honest when a sighting
# goes away — an admin dismissing a bogus sighting drops the count, so a report
# whose only sighting was bogus falls back to PENDING by itself. A stored badge
# would have to be recomputed at every place a sighting can disappear, and would
# quietly claim "someone has seen your pet" the first time one was missed.
#
# This is the ONE vocabulary for the badge. `pet_status` is a storage column
# with a different word list (Searching / Spotted / Found / Resolved) and a
# different job — it is the matching filter, not the label — so clients must
# read `post_status` and never re-derive from `status` themselves.
POST_STATUS_PENDING = "Pending"    # search open, nobody has reported a sighting
POST_STATUS_SPOTTED = "Spotted"    # search open, at least one sighting exists
POST_STATUS_EXPIRED = "Expired"    # aged out with no sighting to show for it
POST_STATUS_RESCUED = "Rescued"    # the owner closed the search

# `pet_status` values that mean the search is over. 'Resolved' is written by the
# resolve RPC; 'Found' is what the owner's End Search button writes.
_CLOSED_PET_STATUSES = frozenset({"found", "resolved"})

# SRS-85: a post stops reaching new hunters once it passes its `expires_at`.
# The read paths (`match_missing_pets`, `get_nearby_missing_pets`) filter
# `mp.expires_at > NOW()`, and the badge below reads the same column — one
# source of truth, no duplicated interval. The seven-day grant lives only in
# the column DEFAULT (`NOW() + INTERVAL '7 days'`); extending one post is a
# plain UPDATE of `expires_at` and needs no change here.

# `sighting_matches.owner_status` — the owner's verdict on one AI match. A
# rejected match is not a sighting of this pet.
OWNER_REJECTED = "Rejected"


def build_missing_pet_payload(pet, *, feature_vector) -> dict:
    """The missing_pets INSERT payload — location as a PostGIS POINT, status
    seeded to 'Searching', and the CLIP feature vector attached.

    `expires_at` is deliberately absent: the column DEFAULT (`NOW() + INTERVAL
    '7 days'`) is the single source of the seven-day grant (SRS-85).
    """
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


def _parse_timestamp(value) -> datetime | None:
    """A timestamp as an aware datetime, or None if it cannot be read.

    PostgREST hands timestamps back as ISO strings, but a caller that already
    holds a row from a driver may have a real datetime — accept both. A value
    that parses to a naive datetime is read as UTC, which is what the column
    stores (`timestamp with time zone`, written by NOW()).

    Returning None rather than raising is deliberate: see `is_post_expired`.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        # Postgres/PostgREST emit 'Z' for UTC, which fromisoformat rejects
        # before Python 3.11.
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def is_post_expired(expires_at, *, now: datetime | None = None) -> bool:
    """Whether a report has passed its `expires_at` (SRS-85).

    The comparison mirrors the SQL predicate exactly. The read paths keep a post
    while `expires_at > NOW()`, so expiry is the negation of that and the
    boundary instant itself counts as expired — a post the database has already
    stopped matching must never still read ACTIVE SEARCH.

    An unreadable or absent `expires_at` is NOT expired. The alternative is to
    grey out a live search because one timestamp arrived in a shape we did not
    expect, and hiding a findable pet is the worse failure.
    """
    expires = _parse_timestamp(expires_at)
    if expires is None:
        return False
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    return expires <= reference


def derive_post_status(
    pet_status: str | None,
    sighting_count: int,
    expires_at=None,
    *,
    now: datetime | None = None,
) -> str:
    """The badge shown on an owner's report card.

    Precedence, highest first:

    * **RESCUED** — a closed search wins over everything: a recovered pet reads
      RESCUED even though sightings were reported along the way, and even if it
      was recovered after the post expired. Those sightings are how it got home,
      not an argument that it is still missing.
    * **SPOTTED** — at least one sighting has come in.
    * **EXPIRED** — nothing has come in AND the post has aged out (SRS-85): it
      no longer matches new sightings and is off the map, so the owner is
      waiting on something that can no longer happen.
    * **PENDING** — nothing has come in yet, but the post is still live.

    EXPIRED ranks BELOW spotted on purpose. An expired post that collected
    sightings still has a queue its owner must work through — the row is
    untouched by expiry and the case can still be closed and paid — so badging
    it EXPIRED would grey out the one report that needs action. Expiry is the
    whole story only when there is no story: the post aged out with nothing to
    show for it.

    `expires_at` is optional: a caller that does not have it gets the pre-expiry
    behaviour rather than a wrong answer.
    """
    if (pet_status or "").strip().lower() in _CLOSED_PET_STATUSES:
        return POST_STATUS_RESCUED
    if sighting_count > 0:
        return POST_STATUS_SPOTTED
    return (
        POST_STATUS_EXPIRED if is_post_expired(expires_at, now=now)
        else POST_STATUS_PENDING
    )


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
    pets: list[dict],
    counts: dict[str, int],
    *,
    now: datetime | None = None,
) -> list[dict]:
    """Add `sighting_count` and `post_status` to each report.

    A pet absent from `counts` has had no sightings — it must read 0, not raise
    and not go missing from the list, because "nobody has reported anything" is
    the ordinary state of a fresh report rather than an error.

    `now` is sampled once for the whole list rather than per row, so a list
    being rendered across an `expires_at` boundary cannot show two reports
    expiring in the same second on opposite sides of it.

    Returns new dicts; the input rows are not mutated.
    """
    reference = now or datetime.now(timezone.utc)
    enriched = []
    for pet in pets:
        count = counts.get(pet.get("id"), 0)
        enriched.append({
            **pet,
            "sighting_count": count,
            "post_status": derive_post_status(
                pet.get("status"), count, pet.get("expires_at"), now=reference,
            ),
        })
    return enriched
