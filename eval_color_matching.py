"""
Accuracy evaluator for colour-aware matching
(sighting_logic.color_similarity and rerank_by_color).

Ground truth is a human label per photo: the coat family, and which photos
show the same cat. By default the family comes from the file name
(ส้ม orange, เทา / ลายเทา grey, ขาว white, ดำ black, Leo cream). A --labels
CSV with columns file,cat_id,family replaces that.

Both colour sources are reproduced the way the app produces them:

  sighting side  median coat colour of the YOLO mask, the same code
                 AIManager.embed_image(with_color=True) runs
  owner side     --tap-model extract (default): the owner keeps the form's
                 default colour, which is measured from their photo exactly
                 like the sighting side (production since 2026-09-25).
                 --tap-model coat / random: the owner replaces it with a
                 one-pixel eyedropper tap (collar_marker_widget.dart), which
                 was the only option before, simulated as many taps inside
                 the cat mask.

Report sections:
  1. extraction   does each photo's sighting colour land in the regime its
                  label expects (neutral for grey/white/black, chromatic for
                  orange/cream/brown), and how many owner taps land wrong
  2. decisions    keep/exclude for every sighting photo x owner tap, scored
                  against the labels. Same family must keep, different family
                  must exclude, the same cat must never be excluded.
  3. sweep        the same score over a grid of the four colour thresholds,
                  to see where the current config.py values rank
  4. ranking      the full CLIP + colour re-rank with every other photo as a
                  candidate, counting wrong-family pets in the top 5
  5. weight       the same ranking over several colour weights (CLIP scores
                  every cat in a narrow band, so this weight decides how much
                  colour reorders the list)

Limits: the numbers only cover the colours in the photo set. Add photos
(brown, calico, tortie, dark grey, cats in shade) before trusting a tuned
threshold. The deployed match_missing_pets RPC also filters by species,
radius and a CLIP threshold, which this offline run does not model.

Usage:
    python eval_color_matching.py [--dir <folder>] [--labels labels.csv]
                                  [--taps 200] [--trials 50] [--seed 0]
                                  [--no-sweep] [--query <file>]
"""
import argparse
import csv
import itertools
import random
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.ai_service import AIManager
from app.services.sighting_logic import EXCLUDE, color_similarity, rerank_by_color

DEFAULT_DIR = Path.home() / "Downloads" / "แมว"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

FAMILY_BY_PREFIX = [
    ("ลายเทา", "grey"), ("เทา", "grey"), ("ขาว", "white"),
    ("ดำ", "black"), ("ส้ม", "orange"), ("Leo", "cream"),
]
SAME_CAT = {"Leo1.jpg": "Leo", "Leo2.jpg": "Leo",
            "เทา2.jpg": "Silver", "เทาTest.jpg": "Silver"}
NEUTRAL_FAMILIES = {"grey", "white", "black"}
# Pairs a person could reasonably call either way. Reported, never scored.
AMBIGUOUS = {frozenset({"orange", "cream"})}

SWEEP_NEUTRAL_CHROMA = np.arange(4, 25, 2)
SWEEP_CHROMATIC_EXCLUDE = np.arange(16, 61, 4)
SWEEP_NEUTRAL_LIGHTNESS = np.arange(15, 61, 5)
SWEEP_LIGHTNESS_WEIGHT = (0.2, 0.4, 0.6, 1.0)
SAME_CAT_KEEP_FLOOR = 0.95


def current_config() -> dict:
    return {
        "neutral_chroma": settings.NEUTRAL_CHROMA_THRESHOLD,
        "chromatic_exclude": settings.COLOR_EXCLUDE_DISTANCE,
        "neutral_lightness_exclude": settings.NEUTRAL_LIGHTNESS_EXCLUDE,
        "lightness_weight": settings.COLOR_LIGHTNESS_WEIGHT,
    }


@dataclass
class Photo:
    file: str
    cat_id: str
    family: str
    yolo_hit: bool
    sighting_hex: str | None
    taps: np.ndarray            # (T, 3) uint8 owner eyedropper samples
    sighting_vec: np.ndarray    # unit CLIP vector, sighting-side crop
    owner_vec: np.ndarray       # unit CLIP vector, owner-side crop


# --- labels ------------------------------------------------------------- #

def default_labels(folder: Path) -> dict[str, tuple[str, str]]:
    labels = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        name = unicodedata.normalize("NFC", path.name)
        family = next((f for p, f in FAMILY_BY_PREFIX if name.startswith(p)), None)
        if family is None:
            print(f" ⚠️  {name}: no family prefix, skipped (use --labels)")
            continue
        labels[name] = (SAME_CAT.get(name, name), family)
    return labels


def csv_labels(path: Path) -> dict[str, tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {unicodedata.normalize("NFC", r["file"]): (r["cat_id"], r["family"].lower())
                for r in csv.DictReader(fh)}


# --- photo loading (production AI code, no download) ------------------- #

def coat_pixels(pixels: np.ndarray) -> np.ndarray:
    """Pixels in the middle half of the mask's brightness: what an owner who
    deliberately taps the coat would hit, without eyes, highlights, dark
    stripes or deep shadow."""
    lum = pixels @ np.array([0.2126, 0.7152, 0.0722])
    lo, hi = np.percentile(lum, [25, 75])
    return pixels[(lum >= lo) & (lum <= hi)]


def load_photo(path: Path, cat_id: str, family: str, taps: int,
               tap_model: str, rng: np.random.Generator) -> Photo:
    image = Image.open(path).convert("RGB")
    results = AIManager.get_yolo().predict(source=image, conf=0.25, verbose=False)
    # Sighting analyze passes no species, register_missing_pet passes it.
    s_iso = AIManager.isolate_subject(image, results)
    o_iso = AIManager.isolate_subject(image, results, expected_species="cat")
    s_img = s_iso[0] if s_iso else image
    o_img = o_iso[0] if o_iso else image

    pixels = np.asarray(o_img).reshape(-1, 3)
    if o_iso:
        pixels = pixels[pixels.any(axis=1)]  # outside the mask is exact 0
    if tap_model == "coat":
        pixels = coat_pixels(pixels)
    elif tap_model == "extract" and o_iso:
        owner_hex = AIManager.extract_coat_color_hex(o_img)
        if owner_hex:
            pixels = np.array([hex_to_rgb(owner_hex)], dtype=np.uint8)
    clip = AIManager.get_clip()

    def unit(v):
        v = np.asarray(v, dtype=np.float32)
        return v / np.linalg.norm(v)

    return Photo(
        file=path.name, cat_id=cat_id, family=family,
        yolo_hit=s_iso is not None and o_iso is not None,
        sighting_hex=AIManager.extract_coat_color_hex(s_img) if s_iso else None,
        taps=pixels[rng.integers(0, len(pixels), size=taps)],
        sighting_vec=unit(clip.encode(s_img)),
        owner_vec=unit(clip.encode(o_img)),
    )


# --- vectorised CIELab, must agree with sighting_logic.hex_to_lab ------- #

_M = np.array([[0.4124, 0.3576, 0.1805],
               [0.2126, 0.7152, 0.0722],
               [0.0193, 0.1192, 0.9505]])
_D65 = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    t = (lin @ _M.T) / _D65
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(x) for x in rgb))


# --- pair decisions ----------------------------------------------------- #

@dataclass
class PairSet:
    sighting: list[int]      # index into photos, one per pair
    owner: list[int]
    kind: np.ndarray         # 'same_cat' | 'same_family' | 'diff_family' | 'ambiguous'
    has_colour: np.ndarray   # False when the sighting has no colour (always kept)
    s_lab: np.ndarray        # (P, 1, 3)
    o_lab: np.ndarray        # (P, T, 3)


def pair_kind(a: Photo, b: Photo) -> str:
    if a.cat_id == b.cat_id:
        return "same_cat"
    if a.family == b.family:
        return "same_family"
    if frozenset({a.family, b.family}) in AMBIGUOUS:
        return "ambiguous"
    return "diff_family"


def build_pairs(photos: list[Photo]) -> PairSet:
    idx = [(i, j) for i, j in itertools.permutations(range(len(photos)), 2)]
    s_hex = [photos[i].sighting_hex for i, _ in idx]
    return PairSet(
        sighting=[i for i, _ in idx], owner=[j for _, j in idx],
        kind=np.array([pair_kind(photos[i], photos[j]) for i, j in idx]),
        has_colour=np.array([h is not None for h in s_hex]),
        s_lab=rgb_to_lab(np.array([hex_to_rgb(h or "#000000") for h in s_hex]))[:, None, :],
        o_lab=rgb_to_lab(np.stack([photos[j].taps for _, j in idx])),
    )


def keep_matrix(ps: PairSet, neutral_chroma, chromatic_exclude,
                neutral_lightness_exclude, lightness_weight) -> np.ndarray:
    s, o = ps.s_lab, ps.o_lab
    s_neutral = np.hypot(s[..., 1], s[..., 2]) < neutral_chroma
    o_neutral = np.hypot(o[..., 1], o[..., 2]) < neutral_chroma
    dl = s[..., 0] - o[..., 0]
    neutral_keep = np.abs(dl) <= neutral_lightness_exclude
    dist = np.sqrt(dl * dl * max(lightness_weight, 0.0)
                   + (s[..., 1] - o[..., 1]) ** 2 + (s[..., 2] - o[..., 2]) ** 2)
    keep = np.where(s_neutral != o_neutral, False,
                    np.where(s_neutral, neutral_keep, dist <= chromatic_exclude))
    return keep | ~ps.has_colour[:, None]


def cross_check(ps: PairSet, photos: list[Photo], keep: np.ndarray,
                cfg: dict, rng: random.Random, samples: int = 5000) -> None:
    """The sweep uses the numpy copy above. Prove it matches production."""
    for _ in range(samples):
        p = rng.randrange(len(ps.sighting))
        t = rng.randrange(keep.shape[1])
        tap_hex = rgb_to_hex(photos[ps.owner[p]].taps[t])
        sim = color_similarity(
            photos[ps.sighting[p]].sighting_hex, tap_hex,
            neutral_chroma=cfg["neutral_chroma"],
            neutral_lightness_exclude=cfg["neutral_lightness_exclude"],
            chromatic_exclude=cfg["chromatic_exclude"],
            lightness_weight=cfg["lightness_weight"],
        )
        if (sim != EXCLUDE) != bool(keep[p, t]):
            raise SystemExit(
                f"❌ evaluator disagrees with color_similarity on "
                f"{photos[ps.sighting[p]].sighting_hex} vs {tap_hex}. "
                f"Fix keep_matrix before trusting any number below.")


def score(ps: PairSet, keep: np.ndarray) -> dict:
    rate = keep.mean(axis=1)

    def mean_of(*kinds):
        sel = np.isin(ps.kind, kinds)
        return float(rate[sel].mean()) if sel.any() else float("nan")

    same = mean_of("same_cat", "same_family")
    diff_excluded = 1.0 - mean_of("diff_family")
    return {
        "same_cat_keep": mean_of("same_cat"),
        "same_family_keep": same,
        "diff_family_exclude": diff_excluded,
        "balanced": (same + diff_excluded) / 2,
    }


# --- report sections ---------------------------------------------------- #

def pct(x: float) -> str:
    return "  n/a" if np.isnan(x) else f"{100 * x:5.1f}%"


def report_extraction(photos: list[Photo], cfg: dict) -> None:
    print("\n=== 1. Colour extraction per photo ===")
    print("regime OK = the sighting colour falls in the regime the label expects")
    print("bad taps  = owner eyedropper taps that land in the wrong regime\n")
    print(f"{'file':16} {'family':7} {'yolo':4} {'sighting':8} {'C*':>5} {'regime':9} {'bad taps':>8}")
    for p in photos:
        want_neutral = p.family in NEUTRAL_FAMILIES
        tap_neutral = np.hypot(*rgb_to_lab(p.taps)[:, 1:].T) < cfg["neutral_chroma"]
        bad_taps = float((tap_neutral != want_neutral).mean())
        if p.sighting_hex:
            lab = rgb_to_lab(np.array(hex_to_rgb(p.sighting_hex)))
            c = float(np.hypot(lab[1], lab[2]))
            ok = (c < cfg["neutral_chroma"]) == want_neutral
            regime = "OK" if ok else "WRONG"
            print(f"{p.file:16} {p.family:7} {'hit' if p.yolo_hit else 'MISS':4} "
                  f"{p.sighting_hex:8} {c:5.1f} {regime:9} {pct(bad_taps):>8}")
        else:
            print(f"{p.file:16} {p.family:7} {'hit' if p.yolo_hit else 'MISS':4} "
                  f"{'none':8} {'':>5} {'no colour':9} {pct(bad_taps):>8}")


def report_decisions(photos: list[Photo], ps: PairSet, keep: np.ndarray) -> dict:
    s = score(ps, keep)
    print("\n=== 2. Keep/exclude decisions, current config ===")
    print(f"same cat kept             {pct(s['same_cat_keep'])}   (lost pet must not be hidden, want ~100%)")
    print(f"same family kept          {pct(s['same_family_keep'])}   (want high)")
    print(f"different family excluded {pct(s['diff_family_exclude'])}   (want high, the screenshot bug lives here)")
    print(f"balanced accuracy         {pct(s['balanced'])}")

    families = sorted({p.family for p in photos})
    rate = keep.mean(axis=1)
    print("\nKeep rate by family. Rows = sighting, columns = owner's tapped colour.")
    print("Diagonal should be high, everything else low ('~' = ambiguous, not scored).\n")
    print(f"{'':8}" + "".join(f"{f:>9}" for f in families))
    for fs in families:
        cells = []
        for fo in families:
            sel = np.array([photos[i].family == fs and photos[j].family == fo
                            for i, j in zip(ps.sighting, ps.owner)])
            mark = "~" if frozenset({fs, fo}) in AMBIGUOUS else " "
            cells.append(f"{pct(float(rate[sel].mean())) if sel.any() else '    -'}{mark}".rjust(9))
        print(f"{fs:8}" + "".join(cells))

    print("\nWorst errors (pair keep rate):")
    order = np.argsort(rate)
    wrong_keep = [p for p in np.argsort(-rate) if ps.kind[p] == "diff_family"][:8]
    wrong_drop = [p for p in order if ps.kind[p] in ("same_cat", "same_family")][:8]
    for p in wrong_keep:
        a, b = photos[ps.sighting[p]], photos[ps.owner[p]]
        print(f"  wrongly kept   {a.file} ({a.family}) -> {b.file} ({b.family})  kept {pct(rate[p])}")
    for p in wrong_drop:
        a, b = photos[ps.sighting[p]], photos[ps.owner[p]]
        print(f"  wrongly dropped {a.file} ({a.family}) -> {b.file} ({b.family})  kept {pct(rate[p])}  [{ps.kind[p]}]")
    return s


def report_sweep(ps: PairSet, cfg: dict) -> None:
    print("\n=== 3. Threshold sweep ===")
    print(f"ranked by balanced accuracy among configs that keep the same cat >= {SAME_CAT_KEEP_FLOOR:.0%}\n")
    rows = []
    for nc, ce, nle, lw in itertools.product(SWEEP_NEUTRAL_CHROMA, SWEEP_CHROMATIC_EXCLUDE,
                                             SWEEP_NEUTRAL_LIGHTNESS, SWEEP_LIGHTNESS_WEIGHT):
        s = score(ps, keep_matrix(ps, nc, ce, nle, lw))
        rows.append(((float(nc), float(ce), float(nle), float(lw)), s))
    cur_key = (cfg["neutral_chroma"], cfg["chromatic_exclude"],
               cfg["neutral_lightness_exclude"], cfg["lightness_weight"])
    cur = score(ps, keep_matrix(ps, *cur_key))

    def eligible(s):
        return np.isnan(s["same_cat_keep"]) or s["same_cat_keep"] >= SAME_CAT_KEEP_FLOOR

    ranked = sorted((r for r in rows if eligible(r[1])),
                    key=lambda r: r[1]["balanced"], reverse=True)
    better = sum(1 for _, s in ranked if s["balanced"] > cur["balanced"])
    header = f"{'neutral_C':>9} {'chrom_ex':>8} {'neutral_L':>9} {'L_wt':>5} {'same cat':>8} {'same fam':>8} {'diff ex':>8} {'balanced':>8}"
    print(header)

    def line(key, s, tag=""):
        print(f"{key[0]:9.0f} {key[1]:8.0f} {key[2]:9.0f} {key[3]:5.1f} "
              f"{pct(s['same_cat_keep']):>8} {pct(s['same_family_keep']):>8} "
              f"{pct(s['diff_family_exclude']):>8} {pct(s['balanced']):>8} {tag}")

    for key, s in ranked[:10]:
        line(key, s)
    line(cur_key, cur, f"<- current config.py ({better} of {len(ranked)} eligible configs score higher)")
    print("\nA tuned config is only as good as the photo set. Check the top rows still")
    print("make sense before copying them into config.py.")


SWEEP_COLOR_WEIGHT = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)


def report_weight_sweep(photos: list[Photo], trials: int, rng: random.Random) -> None:
    """CLIP scores every cat in a narrow band, so the colour weight decides how
    much colour reorders the list. 0.0 means colour only filters."""
    limit = settings.DEFAULT_MATCH_LIMIT
    print(f"\n=== 5. Colour weight sweep (ranking, top {limit}) ===")
    print(f"{'colour wt':>9} {'wrong fam/top':>13} {'true cat in top':>15}")
    for cw in SWEEP_COLOR_WEIGHT:
        wrong, hits = [], []
        for q in photos:
            cands = [p for p in photos if p is not q]
            by_file = {p.file: p for p in cands}
            has_twin = any(p.cat_id == q.cat_id for p in cands)
            for _ in range(trials):
                pool = [{"id": p.file, "similarity": float(q.sighting_vec @ p.owner_vec),
                         "primary_color_hex": rgb_to_hex(p.taps[rng.randrange(len(p.taps))])}
                        for p in cands]
                top = rerank_by_color(
                    pool, q.sighting_hex,
                    clip_weight=1.0 - cw, color_weight=cw,
                    exclude_distance=settings.COLOR_EXCLUDE_DISTANCE,
                    lightness_weight=settings.COLOR_LIGHTNESS_WEIGHT,
                    neutral_chroma=settings.NEUTRAL_CHROMA_THRESHOLD,
                    neutral_lightness_exclude=settings.NEUTRAL_LIGHTNESS_EXCLUDE,
                    limit=limit,
                )
                wrong.append(sum(pair_kind(q, by_file[m["id"]]) == "diff_family" for m in top))
                if has_twin:
                    hits.append(any(by_file[m["id"]].cat_id == q.cat_id for m in top))
        tag = "  <- current" if abs(cw - settings.COLOR_MATCH_WEIGHT) < 1e-9 else ""
        print(f"{cw:9.2f} {np.mean(wrong):13.2f} {pct(float(np.mean(hits))) if hits else '  n/a':>15}{tag}")


def report_ranking(photos: list[Photo], trials: int, rng: random.Random,
                   query: str | None) -> None:
    print("\n=== 4. End-to-end ranking (CLIP + colour re-rank, top "
          f"{settings.DEFAULT_MATCH_LIMIT}) ===")
    print(f"every other photo is a candidate pet, {trials} trials\n")
    limit = settings.DEFAULT_MATCH_LIMIT
    print(f"{'':24} {'wrong family in top':^27} {'true cat in top':^27}")
    print(f"{'query':16} {'family':7} {'CLIP-only':>13} {'+colour':>13} {'CLIP-only':>13} {'+colour':>13}")
    totals = {"clip": [], "colour": [], "hit": [], "clip_hit": []}
    for qi, q in enumerate(photos):
        if query and q.file != query:
            continue
        cands = [p for p in photos if p is not q]
        sims = {p.file: float(q.sighting_vec @ p.owner_vec) for p in cands}
        by_file = {p.file: p for p in cands}
        clip_top = sorted(cands, key=lambda p: sims[p.file], reverse=True)[:limit]
        wrong_clip = sum(pair_kind(q, p) == "diff_family" for p in clip_top)
        wrong, hits, seen = [], [], {}
        has_twin = any(p.cat_id == q.cat_id for p in cands)
        for _ in range(trials):
            pool = [{"id": p.file, "similarity": sims[p.file],
                     "primary_color_hex": rgb_to_hex(p.taps[rng.randrange(len(p.taps))])}
                    for p in cands]
            pool.sort(key=lambda m: m["similarity"], reverse=True)
            top = rerank_by_color(
                pool[:settings.MATCH_CANDIDATE_POOL], q.sighting_hex,
                clip_weight=settings.CLIP_MATCH_WEIGHT,
                color_weight=settings.COLOR_MATCH_WEIGHT,
                exclude_distance=settings.COLOR_EXCLUDE_DISTANCE,
                lightness_weight=settings.COLOR_LIGHTNESS_WEIGHT,
                neutral_chroma=settings.NEUTRAL_CHROMA_THRESHOLD,
                neutral_lightness_exclude=settings.NEUTRAL_LIGHTNESS_EXCLUDE,
                limit=limit,
            )
            wrong.append(sum(pair_kind(q, by_file[m["id"]]) == "diff_family" for m in top))
            hits.append(any(by_file[m["id"]].cat_id == q.cat_id for m in top))
            for rank, m in enumerate(top):
                seen.setdefault(m["id"], []).append(rank + 1)
        clip_hit = any(p.cat_id == q.cat_id for p in clip_top)
        totals["clip"].append(wrong_clip)
        totals["colour"].append(float(np.mean(wrong)))
        if has_twin:
            totals["hit"].append(float(np.mean(hits)))
            totals["clip_hit"].append(float(clip_hit))
            hit_txt = f"{'yes' if clip_hit else 'no':>13} {pct(float(np.mean(hits))):>13}"
        else:
            hit_txt = f"{'-':>13} {'-':>13}"
        print(f"{q.file:16} {q.family:7} {wrong_clip:13d} {np.mean(wrong):13.2f} {hit_txt}")

        if query:
            print(f"\n  How often each candidate reached the top {limit} for {q.file} "
                  f"(sighting colour {q.sighting_hex}):")
            for fid, ranks in sorted(seen.items(), key=lambda kv: -len(kv[1])):
                c = by_file[fid]
                print(f"   {fid:16} {c.family:7} {pct(len(ranks) / trials)}  "
                      f"CLIP {sims[fid]:.3f}  avg rank {np.mean(ranks):.1f}  [{pair_kind(q, c)}]")

    if not query:
        print(f"\nmean wrong-family pets per top {limit}: CLIP-only "
              f"{np.mean(totals['clip']):.2f}, with colour {np.mean(totals['colour']):.2f}")
        if totals["hit"]:
            print(f"true cat in the top {limit}: CLIP-only "
                  f"{pct(float(np.mean(totals['clip_hit'])))}, with colour "
                  f"{pct(float(np.mean(totals['hit'])))}  "
                  f"({len(totals['hit'])} queries have a second photo of the same cat)")


# --- main --------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Accuracy evaluator for colour-aware matching")
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--labels", type=Path, help="CSV with file,cat_id,family")
    ap.add_argument("--taps", type=int, default=200, help="simulated eyedropper taps per photo")
    ap.add_argument("--trials", type=int, default=50, help="ranking trials per query")
    ap.add_argument("--tap-model", choices=("extract", "coat", "random"), default="extract",
                    help="extract = owner keeps the measured default, coat = owner "
                         "re-taps the main coat, random = owner re-taps any "
                         "mask pixel (worst case)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--query", help="only rank this file and show its candidates")
    args = ap.parse_args()

    labels = csv_labels(args.labels) if args.labels else default_labels(args.dir)
    if len(labels) < 2:
        raise SystemExit("❌ need at least two labelled photos")
    np_rng = np.random.default_rng(args.seed)
    rng = random.Random(args.seed)

    print(f"⏳ Loading YOLO + CLIP and processing {len(labels)} photos from {args.dir}…")
    photos = []
    for path in sorted(args.dir.iterdir()):
        key = unicodedata.normalize("NFC", path.name)
        if key in labels:
            cat_id, family = labels[key]
            photos.append(load_photo(path, cat_id, family, args.taps,
                                     args.tap_model, np_rng))
            photos[-1].file = key
    if args.query and args.query not in {p.file for p in photos}:
        raise SystemExit(f"❌ --query {args.query} is not a labelled photo")

    cfg = current_config()
    print("current config: " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
    print(f"owner tap model: {args.tap_model}")
    ps = build_pairs(photos)
    keep = keep_matrix(ps, **cfg)
    cross_check(ps, photos, keep, cfg, rng)

    if not args.query:
        report_extraction(photos, cfg)
        report_decisions(photos, ps, keep)
        if not args.no_sweep:
            report_sweep(ps, cfg)
    report_ranking(photos, args.trials, rng, args.query)
    if not args.query and not args.no_sweep:
        report_weight_sweep(photos, args.trials, rng)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
