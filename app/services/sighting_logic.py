"""Pure, I/O-free logic extracted from SightingService.

No DB, no AI, no awaiting — just payload construction, filtering, row mapping,
and the in-Python assembly of hunter activity. Because these are pure, they get
cheap exhaustive boundary tests (empty/single/at-threshold/missing-field).
"""
from app.utils.postgis import create_postgis_point


def build_sighting_payload(sighting, *, vector, target_pet_id: str | None) -> dict:
    """The sightings INSERT payload. detected_species is the CLIENT-supplied
    (user-confirmed) value, NOT YOLO's guess. `vector` is the CLIP embedding for
    discovery or None for targeted; `target_pet_id` is set only for targeted.
    """
    location = create_postgis_point(sighting.latitude, sighting.longitude)
    payload = {
        "hunter_id": sighting.hunter_id,
        "sighted_location": location,
        "image_url": str(sighting.image_url),
        "detected_species": sighting.detected_species,  # user-confirmed
        "action_type": sighting.action_type,
        "sighting_status": "Pending_Analysis",
    }
    if vector is not None:
        payload["feature_vector"] = vector
    if target_pet_id:
        payload["initial_target_pet_id"] = target_pet_id
    return payload


def strip_feature_vector(row: dict) -> dict:
    """Drop the 512-D feature_vector before a row goes to the client (bloat +
    clients never use it). Mutates and returns the same row."""
    row.pop("feature_vector", None)
    return row


def filter_by_threshold(matches: list[dict], threshold: float) -> list[dict]:
    """Keep matches whose similarity >= threshold. A falsy threshold (0.0)
    returns every row; a NULL similarity is treated as 0.0."""
    if not threshold:
        return matches
    return [m for m in matches if (m.get("similarity") or 0.0) >= threshold]


def build_match_rows(sighting_id: str, matches: list[dict]) -> list[dict]:
    """Map match RPC rows (keyed `id` + `similarity`) to sighting_matches rows.
    Matches without an `id` are dropped."""
    return [
        {
            "sighting_id": sighting_id,
            "missing_pet_id": m["id"],
            "similarity_score": m.get("similarity"),
        }
        for m in matches
        if m.get("id")
    ]


def assemble_hunter_activity(
    sightings: list[dict], matches: list[dict], awards: list[dict]
) -> list[dict]:
    """Attach each sighting's AI match candidates and score award in place."""
    matches_by_sighting: dict[str, list] = {}
    for m in matches:
        matches_by_sighting.setdefault(m["sighting_id"], []).append(m)

    awards_by_sighting: dict[str, dict] = {
        a["sighting_id"]: a for a in awards if a.get("sighting_id")
    }

    for s in sightings:
        s["matches"] = matches_by_sighting.get(s["id"], [])
        s["score_award"] = awards_by_sighting.get(s["id"])
    return sightings
