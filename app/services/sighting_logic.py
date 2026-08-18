"""Pure, I/O-free logic extracted from SightingService.

No DB, no AI, no awaiting — just payload construction, filtering, row mapping,
colour maths, and the in-Python assembly of hunter activity. Because these are
pure, they get cheap exhaustive boundary tests (empty/single/at-threshold/
missing-field).
"""
import math

from app.utils.postgis import create_postgis_point


def build_sighting_payload(
    sighting, *, vector, target_pet_id: str | None,
    primary_color_hex: str | None = None,
) -> dict:
    """The sightings INSERT payload. detected_species is the CLIENT-supplied
    (user-confirmed) value, NOT YOLO's guess. `vector` is the CLIP embedding for
    discovery or None for targeted; `target_pet_id` is set only for targeted;
    `primary_color_hex` is the coat colour auto-extracted from the isolated
    subject (None when extraction was skipped — full-frame fallback or a
    near-black subject — so the column is simply left NULL).
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
    if primary_color_hex:
        payload["primary_color_hex"] = primary_color_hex
    return payload


# --- colour maths ------------------------------------------------------- #
# The owner picks a missing pet's colour by hand; a sighting's colour is
# auto-extracted from the photo. Those two sources disagree most on
# BRIGHTNESS (a swatch is idealised; a real coat is shaded), so we compare in
# CIELab with lightness down-weighted: what separates grey (a*,b*≈0) from
# orange (a*,b* large) is chroma, not L*. See config.COLOR_* for the tunables.

# sRGB→XYZ (D65) matrix rows and the D65 white point.
_D65 = (0.95047, 1.0, 1.08883)


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int] | None:
    """'#RRGGBB' (or 'RRGGBB') → (r, g, b) in 0–255, or None if malformed."""
    if not hex_str:
        return None
    s = hex_str.strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return None


def hex_to_lab(hex_str: str | None) -> tuple[float, float, float] | None:
    """Convert a hex colour to CIELab (L*, a*, b*). None on missing/malformed
    input so callers can treat 'no colour' as 'colour can't decide'."""
    rgb = _hex_to_rgb(hex_str) if hex_str else None
    if rgb is None:
        return None

    def _linear(c: int) -> float:
        cs = c / 255.0
        return cs / 12.92 if cs <= 0.04045 else ((cs + 0.055) / 1.055) ** 2.4

    r, g, b = (_linear(c) for c in rgb)
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    def _f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = _f(x / _D65[0]), _f(y / _D65[1]), _f(z / _D65[2])
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def color_distance(
    hex_a: str | None, hex_b: str | None, *, lightness_weight: float = 1.0
) -> float | None:
    """Lightness-weighted CIELab distance between two hex colours, or None if
    either colour is missing/malformed. `lightness_weight` < 1.0 makes the
    metric care mostly about chroma (colour family) and forgive brightness."""
    lab_a, lab_b = hex_to_lab(hex_a), hex_to_lab(hex_b)
    if lab_a is None or lab_b is None:
        return None
    dl = (lab_a[0] - lab_b[0]) * math.sqrt(max(lightness_weight, 0.0))
    da = lab_a[1] - lab_b[1]
    db = lab_a[2] - lab_b[2]
    return math.sqrt(dl * dl + da * da + db * db)


def _chroma(lab: tuple[float, float, float]) -> float:
    """CIELab chroma C* = √(a*² + b*²) — how colourful (vs neutral grey) a
    colour is, independent of brightness."""
    return math.hypot(lab[1], lab[2])


# EXCLUDE is a sentinel score meaning "drop this candidate", kept distinct from a
# real 0.0 similarity so a same-family-but-distant colour can still score 0.
EXCLUDE = -1.0


def color_similarity(
    hex_a: str | None,
    hex_b: str | None,
    *,
    neutral_chroma: float,
    neutral_lightness_exclude: float,
    chromatic_exclude: float,
    lightness_weight: float,
) -> float | None:
    """Coat-colour agreement between two hex colours, split by chroma regime.

    Returns:
      • None    — colour unavailable/malformed on a side (caller falls back to
                  CLIP-only; a missing colour never excludes);
      • EXCLUDE — clearly different colour, drop the candidate. Two cases:
                  a NEUTRAL colour (C* < `neutral_chroma`, e.g. grey) paired
                  with a CHROMATIC one (e.g. orange) — different family; or two
                  NEUTRALs whose |ΔL*| exceeds `neutral_lightness_exclude`
                  (black vs grey);
      • 0..1    — similarity (1 = identical). For two CHROMATICs it falls off
                  linearly to 0 at `chromatic_exclude` on the lightness-weighted
                  CIELab distance; for two NEUTRALs, linearly to 0 at
                  `neutral_lightness_exclude` on |ΔL*|.
    """
    lab_a, lab_b = hex_to_lab(hex_a), hex_to_lab(hex_b)
    if lab_a is None or lab_b is None:
        return None
    a_neutral = _chroma(lab_a) < neutral_chroma
    b_neutral = _chroma(lab_b) < neutral_chroma

    if a_neutral != b_neutral:
        return EXCLUDE  # neutral vs chromatic — different family (grey vs orange)

    if a_neutral and b_neutral:
        dl = abs(lab_a[0] - lab_b[0])
        if dl > neutral_lightness_exclude:
            return EXCLUDE  # black vs grey — same hue, far apart in brightness
        return (max(0.0, 1.0 - dl / neutral_lightness_exclude)
                if neutral_lightness_exclude > 0 else 0.0)

    # both chromatic — chroma-forgiving CIELab distance decides the family.
    dist = color_distance(hex_a, hex_b, lightness_weight=lightness_weight)
    if dist > chromatic_exclude:
        return EXCLUDE
    return (max(0.0, 1.0 - dist / chromatic_exclude)
            if chromatic_exclude > 0 else 0.0)


def _combined_score(
    match: dict,
    sighting_hex: str | None,
    *,
    clip_weight: float,
    color_weight: float,
    neutral_chroma: float,
    neutral_lightness_exclude: float,
    exclude_distance: float,
    lightness_weight: float,
) -> float | None:
    """Blended CLIP+colour score for one candidate, or None if the candidate
    must be EXCLUDED (colour clearly too different). When colour is unavailable
    on either side, the raw CLIP similarity is returned (old behaviour — colour
    can neither help nor exclude)."""
    clip = match.get("similarity") or 0.0
    sim = color_similarity(
        sighting_hex, match.get("primary_color_hex"),
        neutral_chroma=neutral_chroma,
        neutral_lightness_exclude=neutral_lightness_exclude,
        chromatic_exclude=exclude_distance,
        lightness_weight=lightness_weight,
    )
    if sim is None:
        return clip
    if sim == EXCLUDE:
        return None  # exclude
    return clip_weight * clip + color_weight * sim


def rerank_by_color(
    matches: list[dict],
    sighting_hex: str | None,
    *,
    clip_weight: float,
    color_weight: float,
    exclude_distance: float,
    lightness_weight: float,
    limit: int,
    neutral_chroma: float = 10.0,
    neutral_lightness_exclude: float = 45.0,
) -> list[dict]:
    """Fold coat-colour similarity into CLIP ranking, then trim to `limit`.

    Per candidate (each carries CLIP `similarity` + its own `primary_color_hex`),
    colour is judged by regime (see `color_similarity`):
      • colour missing on either side → keep, ranked by raw CLIP (a NULL colour
        never excludes or penalises);
      • neutral coat vs chromatic coat (grey vs orange) → DROP;
      • two neutrals too far apart in brightness (black vs grey) → DROP;
      • two chromatics past `exclude_distance` (orange vs blue) → DROP;
      • otherwise → ranked by clip_weight*clip + color_weight*color_sim, where
        color_sim ∈ [0,1] falls off to 0 at the relevant exclusion boundary.

    The returned match dicts are the SAME objects, UNMUTATED (the response
    contract stays `{id, pet_name, similarity, …}` — `similarity` remains the
    raw CLIP cosine; the colour blend only decides order and membership).
    Sorted by combined score desc. `limit` <= 0 means 'no trim'.
    """
    scored: list[tuple[float, dict]] = []
    for m in matches:
        score = _combined_score(
            m, sighting_hex,
            clip_weight=clip_weight, color_weight=color_weight,
            neutral_chroma=neutral_chroma,
            neutral_lightness_exclude=neutral_lightness_exclude,
            exclude_distance=exclude_distance, lightness_weight=lightness_weight,
        )
        if score is None:
            continue  # excluded — clearly-different colour
        scored.append((score, m))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    ranked = [m for _, m in scored]
    return ranked[:limit] if limit and limit > 0 else ranked


# The owner's verdict on one AI match (`sighting_matches.owner_status`, enum
# `owner_decision`). 'Pending' is the value a match is born with and is NOT a
# decision, so it cannot be written back through this path — an owner either
# says yes or says no.
OWNER_DECISION_CONFIRMED = "Confirmed"
OWNER_DECISION_REJECTED = "Rejected"
OWNER_DECISIONS = (OWNER_DECISION_CONFIRMED, OWNER_DECISION_REJECTED)

_OWNER_DECISION_ALIASES = {
    "confirmed": OWNER_DECISION_CONFIRMED,
    "confirm": OWNER_DECISION_CONFIRMED,
    "yes": OWNER_DECISION_CONFIRMED,
    "rejected": OWNER_DECISION_REJECTED,
    "reject": OWNER_DECISION_REJECTED,
    "no": OWNER_DECISION_REJECTED,
}


def normalize_owner_decision(decision: str | None) -> str:
    """Map the owner's answer onto the `owner_decision` enum.

    Raises ValueError for anything else — including 'Pending', which is a
    starting state rather than an answer, so accepting it would let a decision
    be silently un-made.
    """
    key = (decision or "").strip().lower()
    normalized = _OWNER_DECISION_ALIASES.get(key)
    if normalized is None:
        raise ValueError(
            f"decision must be one of {', '.join(OWNER_DECISIONS)}; "
            f"got {decision!r}"
        )
    return normalized


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
