"""
Sighting service — optimised 2-step pipeline.

Flow 3 in `Petty_Bounty_Brain/wiki/api_flow.md` mandates a manual species
correction gate, so we keep two endpoints but shift the heavy CLIP work
out of `POST /sightings/` and into `POST /sightings/analyze`:

    analyze:  download → YOLO-seg → mask-isolate → CLIP encode → cache
              everything (image, bbox, species, confidence, vector).
    save:     pull vector from cache → INSERT row with USER-confirmed
              species → call match_missing_pets RPC → return matches.

Hot path on confirm is just a DB write + RPC call (~100 ms).
On cache miss (>10 min idle or server restart) the save step re-runs the
full pipeline transparently.

All DB access goes through a SightingRepository port (app/repositories/); this
service holds zero supabase-py calls. Pure logic (payload build, threshold
filter, row mapping, activity assembly) lives in sighting_logic.py.

Threading note — why the `_*_sync` split
----------------------------------------
supabase-py is synchronous: every `.execute()` is a blocking HTTP round-trip
(~50 ms against the live project). Called straight from an `async def`, that
blocks the whole event loop, so concurrent requests serialise behind each other
instead of overlapping. The public methods here therefore stay `async def` but
delegate their DB work to a `_*_sync` counterpart run via `asyncio.to_thread`.

The split is per-METHOD, not per-query, on purpose: `get_hunter_stats` makes
four round-trips, and wrapping each one separately would pay four thread hops
and four event-loop re-entries to no benefit. One hop per logical operation
keeps the loop free for the same total time at a fraction of the overhead.

Methods that genuinely await (the AI pipeline) keep their awaits inline and
wrap only their DB steps.
"""
import asyncio
import logging

from app.core.config import settings
from app.repositories.sighting_repository import (
    SightingActionLocked,
    SightingRepository,
)
from app.schemas.sightings import SightingCreate, TargetedSightingCreate
from app.services.ai_cache import AnalyzeCache
from app.services.sighting_logic import (
    OWNER_DECISION_CONFIRMED,
    VERIFICATION_PENDING,
    assemble_hunter_activity,
    build_match_rows,
    build_sighting_payload,
    filter_by_threshold,
    normalize_action_type,
    normalize_owner_decision,
    rerank_by_color,
    strip_feature_vector,
)

logger = logging.getLogger(__name__)


class SightingService:
    """Coordinates the AI pipeline, sighting persistence, and pgvector match."""

    def __init__(self, repo: SightingRepository, ai_manager):
        self.repo = repo
        self.ai = ai_manager

    async def analyze_sighting_image(self, image_url: str, conf: float = 0.25):
        """
        Heavy step: download → YOLO-seg → mask-isolate → CLIP encode.
        Caches the full pipeline result keyed by image_url so the
        follow-up POST /sightings/ doesn't repeat any of it.

        Returns the same shape as before — `{species, confidence, bbox}` —
        so the Flutter verification screen contract is unchanged.
        """
        try:
            image = await self.ai.download_image(str(image_url))
            results = await self.ai.run_yolo_seg(image, conf=conf)
            iso = self.ai.isolate_subject(image, results)
            if iso is None:
                return {
                    "status": "not_found",
                    "message": "No target animals detected."
                }

            isolated_image, species, confidence, bbox = iso

            # Pre-compute CLIP NOW. This is the entire optimisation —
            # POST /sightings/ doesn't have to do CLIP, it just reads
            # the vector out of the cache.
            feature_vector = await self.ai.clip_encode(isolated_image)
            # Coat colour off the SAME isolated crop (safe: iso is not None here,
            # so the crop has a black background to strip). None for a near-black
            # subject → matching falls back to CLIP-only for this sighting.
            primary_color_hex = self.ai.extract_coat_color_hex(isolated_image)

            cache_key = str(image_url)
            AnalyzeCache.set(cache_key, {
                "pil_image": image,
                "isolated_image": isolated_image,
                "species": species,
                "bbox": bbox,
                "confidence": confidence,
                "feature_vector": feature_vector,
                "primary_color_hex": primary_color_hex,
            })
            # WARNING level so URL drift / worker isolation is easy to spot
            # — pair with the HIT/MISS logs in process_and_save_sighting.
            logger.warning(
                "Analyze-cache SET key=%r (species=%s)", cache_key, species
            )

            return {
                "status": "success",
                "message": f"AI detected a {species}.",
                "data": {
                    "species": species,
                    "confidence": round(confidence * 100, 2),
                    "bbox": bbox,
                }
            }
        except Exception as e:
            logger.error("Error in analyze_sighting_image: %s", e)
            raise

    def _insert_sighting_row(
        self, sighting, *, vector, target_pet_id: str | None,
        primary_color_hex: str | None = None,
    ) -> dict:
        """
        Build the sighting payload, INSERT it via the repo, and return the saved
        row with the 512-D feature_vector stripped out. Shared by the discovery
        and targeted paths so the INSERT contract lives in one place.

        `vector` is the CLIP embedding for discovery, or None for targeted
        (column left NULL). `target_pet_id` is set only for targeted reports.
        `primary_color_hex` is the discovery path's auto-extracted coat colour
        (None → NULL: targeted reports and unreadable/near-black subjects).
        """
        payload = build_sighting_payload(
            sighting, vector=vector, target_pet_id=target_pet_id,
            primary_color_hex=primary_color_hex,
        )
        sighting_row = self.repo.insert_sighting(payload)
        return strip_feature_vector(sighting_row)

    async def process_and_save_sighting(self, sighting: SightingCreate) -> dict:
        """
        Discovery hot path: pull the cached feature_vector (or re-run the
        pipeline on a cache miss), INSERT with the user-confirmed species,
        run the pgvector match RPC, and return {sighting, matches}.

        The targeted (pet-detail) flow does NOT come through here — it has its
        own endpoint/method (save_targeted_sighting), so this path is always
        discovery: it always computes a vector and always runs matching.
        """
        try:
            image_url = str(sighting.image_url)

            cached = AnalyzeCache.get(image_url)
            if cached is not None:
                vector = cached["feature_vector"]
                primary_color_hex = cached.get("primary_color_hex")
                logger.warning(
                    "Analyze-cache HIT key=%r — reusing CLIP vector",
                    image_url,
                )
            else:
                logger.warning(
                    "Analyze-cache MISS key=%r — re-running YOLO + CLIP",
                    image_url,
                )
                image = await self.ai.download_image(image_url)
                results = await self.ai.run_yolo_seg(image)
                # Forgiving re-run: don't constrain by user-confirmed species.
                # The user may have corrected YOLO's guess; the vector should
                # still represent whatever animal pixels YOLO actually finds
                # in the photo. The user's species choice is honoured at the
                # INSERT step regardless.
                iso = self.ai.isolate_subject(image, results)
                if iso is None:
                    raise ValueError(
                        "No target animal detected in image during re-run."
                    )
                isolated, _, _, _ = iso
                vector = await self.ai.clip_encode(isolated)
                # Re-extract colour too so a cache-miss save is colour-aware,
                # not silently CLIP-only.
                primary_color_hex = self.ai.extract_coat_color_hex(isolated)

            sighting_row = await asyncio.to_thread(
                self._insert_sighting_row,
                sighting, vector=vector, target_pet_id=None,
                primary_color_hex=primary_color_hex,
            )
            sighting_id = sighting_row["id"]
            logger.info(
                "Sighting %s saved (species=%s, discovery)",
                sighting_id, sighting.detected_species,
            )

            # Bundle matches in the same response (saves Flutter a round-trip).
            # If the RPC fails the row is still saved; client can re-query
            # via GET /sightings/{id}/matches.
            matches: list[dict] = []
            try:
                matches = await self.get_matches(
                    sighting_id, limit=5, threshold=0.0
                )
            except Exception as e:
                logger.warning(
                    "Sighting %s saved but match RPC failed: %s",
                    sighting_id, e,
                )

            # Persist match results into sighting_matches so the owner
            # timeline and F1 scoring have a sighting↔pet link to read.
            # Wrapped in its own try/except: a persist failure must NOT
            # clobber the matches we already fetched — the client still
            # gets the verify-screen data, and the sighting row itself
            # remains saved.
            if matches:
                try:
                    await asyncio.to_thread(
                        self._persist_matches, sighting_id, matches
                    )
                except Exception as e:
                    # Swallow so we don't clobber the matches already fetched
                    # for the verify screen — but log at ERROR, not WARNING.
                    # A silent warning is exactly why a broken upsert (missing
                    # UNIQUE on sighting_matches) dropped every AI match for
                    # ~10 days unnoticed: owner timelines + F1 scoring read
                    # this table.
                    logger.error(
                        "Sighting %s match-persist FAILED — matches NOT "
                        "written to sighting_matches (owner timeline + "
                        "scoring will miss them); response unaffected: %s",
                        sighting_id, e,
                        exc_info=True,
                    )

            return {"sighting": sighting_row, "matches": matches}

        except ValueError:
            raise
        except Exception as e:
            logger.error("Error in process_and_save_sighting: %s", e)
            raise

    async def save_targeted_sighting(
        self, sighting: TargetedSightingCreate
    ) -> dict:
        """
        Targeted path: the hunter is reporting ONE known pet straight to its
        owner. No CLIP vector, no pgvector match — just persist the row with
        initial_target_pet_id set so it surfaces on the owner's timeline via
        the initial_target_pet_id branch of sightings_for_pet.

        Returns {sighting, matches: []} — the same shape as the discovery
        endpoint so the client parses both responses identically.
        """
        try:
            sighting_row = await asyncio.to_thread(
                self._insert_sighting_row,
                sighting, vector=None, target_pet_id=sighting.target_pet_id,
            )
            logger.info(
                "Sighting %s saved (species=%s, targeted → pet %s)",
                sighting_row["id"], sighting.detected_species,
                sighting.target_pet_id,
            )
            return {"sighting": sighting_row, "matches": []}

        except ValueError:
            raise
        except Exception as e:
            logger.error("Error in save_targeted_sighting: %s", e)
            raise

    async def decide_match(
        self, pet_id: str, sighting_id: str, owner_id: str, decision: str,
    ) -> dict:
        """The owner's verdict on one AI match: "that is my pet" or "that isn't".

        This is the reply half of the loop. Until it existed the owner could
        read their timeline but say nothing back, so `owner_status` sat at
        Pending on every row and a wrong match kept counting as a real sighting
        of the pet.

        A Confirmed verdict also advances the sighting to `Confirmed`. A
        Rejected one deliberately does NOT touch the sighting: one photo can
        match several pets, and this owner saying "not mine" is no statement
        about anybody else's pet.

        Raises:
            ValueError: decision outside {Confirmed, Rejected} (API -> 400).
            LookupError: the pet is not the caller's, or the match does not
                exist (API -> 404 for both — a caller who does not own the pet
                must not be able to tell the two apart).
        """
        # Raises ValueError (400) before any I/O — cheap, so it stays on the
        # loop and a bad decision string never costs a thread hop.
        status = normalize_owner_decision(decision)
        return await asyncio.to_thread(
            self._decide_match_sync, pet_id, sighting_id, owner_id, status
        )

    def _decide_match_sync(
        self, pet_id: str, sighting_id: str, owner_id: str, status: str,
    ) -> dict:
        """DB half of `decide_match`; `status` is already normalised."""
        owners = self.repo.get_pet_owners([pet_id])
        if owners.get(pet_id) != owner_id:
            # Same 404-not-403 rule the owner-scoped PATCH uses: never confirm
            # that a pet exists to someone who does not own it.
            raise LookupError(f"Missing pet {pet_id} not found or not owned by you")

        updated = self.repo.update_match_owner_status(
            sighting_id, pet_id, status
        )
        if not updated:
            raise LookupError(
                f"Sighting {sighting_id} is not a match for pet {pet_id}"
            )

        sighting_status_written = False
        if status == OWNER_DECISION_CONFIRMED:
            try:
                self.repo.set_sighting_status(sighting_id, "Confirmed")
                sighting_status_written = True
            except Exception as e:
                # The owner's verdict IS recorded; the sighting's own lifecycle
                # column is secondary, so a failure here must not lose it.
                logger.error(
                    "Match %s/%s confirmed but sighting status write FAILED: %s",
                    sighting_id, pet_id, e,
                )

        logger.info(
            "Owner %s marked sighting %s as %s for pet %s",
            owner_id, sighting_id, status, pet_id,
        )
        return {
            "match": updated,
            "sighting_status_updated": sighting_status_written,
        }

    async def confirm_sighting_action(
        self, sighting_id: str, hunter_id: str, action_type: str,
    ) -> dict:
        """The hunter's final-review answer: did you just see it, or rescue it?

        This is the step AFTER "Confirm Match". The sighting itself was already
        written by `process_and_save_sighting` (with `action_type` defaulted to
        'Spotted'), and the owner was already pushed — so this endpoint exists
        only to persist the one choice that screen collects. That choice is not
        cosmetic: 'Caught' is the value the resolve RPC requires before a
        sighting can pay a bounty, so it is deliberately a separate,
        hunter-scoped write rather than a field on the create payload.

        Idempotent by design: re-confirming the value the row already holds —
        the common case, since 'Spotted' is both the UI default and the stored
        default — reports success without touching the row.

        Raises:
            ValueError: `action_type` outside the enum (API -> 400).
            LookupError: no such sighting, or it belongs to another hunter
                (API -> 404 for both — a caller who does not own the sighting
                must not be able to tell the two apart).
            SightingActionLocked: the sighting has already been reviewed
                (API -> 409). Subclasses ValueError, so the route must catch it
                FIRST.
        """
        # Raises ValueError (400) before any I/O, so a malformed choice never
        # costs a thread hop — same shape as decide_match.
        normalized = normalize_action_type(action_type)
        return await asyncio.to_thread(
            self._confirm_sighting_action_sync,
            sighting_id, hunter_id, normalized,
        )

    def _confirm_sighting_action_sync(
        self, sighting_id: str, hunter_id: str, action_type: str,
    ) -> dict:
        """DB half of `confirm_sighting_action`; `action_type` is normalised."""
        row = self.repo.get_sighting_for_action(sighting_id)
        if row is None or row.get("hunter_id") != hunter_id:
            # 404-not-403, the same rule the owner-scoped writes use: never
            # confirm that a sighting exists to someone who did not report it.
            raise LookupError(
                f"Sighting {sighting_id} not found or not reported by you"
            )

        # A row whose verification_status is missing is treated as Pending —
        # the column is NOT NULL DEFAULT 'Pending', so absence means "not yet
        # judged", and refusing the write there would lock a fresh sighting.
        verification = row.get("verification_status") or VERIFICATION_PENDING
        if verification != VERIFICATION_PENDING:
            raise SightingActionLocked(sighting_id, verification)

        if row.get("action_type") == action_type:
            logger.info(
                "Sighting %s action already %s — no write",
                sighting_id, action_type,
            )
            return {"sighting": row, "action_type": action_type,
                    "changed": False}

        updated = self.repo.set_sighting_action_type(
            sighting_id, hunter_id, action_type
        )
        if updated is None:
            # The read said it was theirs and the write matched nothing — the
            # row went away (or changed hands) in between. Same 404.
            raise LookupError(
                f"Sighting {sighting_id} not found or not reported by you"
            )

        logger.info(
            "Hunter %s confirmed sighting %s as %s",
            hunter_id, sighting_id, action_type,
        )
        return {
            "sighting": strip_feature_vector(updated),
            "action_type": action_type,
            "changed": True,
        }

    def _persist_matches(self, sighting_id: str, matches: list[dict]) -> None:
        """
        Log AI match results into sighting_matches via the repo's upsert (on the
        (sighting_id, missing_pet_id) unique constraint, so a retry refreshes
        similarity_score in place rather than duplicating). No-op when the
        mapped row set is empty.
        """
        rows = build_match_rows(sighting_id, matches)
        if rows:
            self.repo.upsert_sighting_matches(rows)

    async def get_matches(
        self,
        sighting_id: str,
        limit: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Find matching missing pets via the match_missing_pets RPC."""
        return await asyncio.to_thread(
            self._get_matches_sync, sighting_id, limit, threshold
        )

    def _get_matches_sync(
        self, sighting_id: str, limit: int, threshold: float,
    ) -> list[dict]:
        try:
            sighting = self.repo.get_sighting_for_match(sighting_id)
            if not sighting:
                raise ValueError(f"Sighting {sighting_id} not found")
            if not sighting.get("feature_vector"):
                raise ValueError(f"Sighting {sighting_id} has no feature vector")
            if not sighting.get("detected_species"):
                raise ValueError(f"Sighting {sighting_id} has no detected species")
            if not sighting.get("sighted_location"):
                raise ValueError(f"Sighting {sighting_id} has no location")

            # Pull a WIDER pool than the client's limit: the SQL side ranks by
            # CLIP only, so the colour re-rank must see candidates that sit just
            # outside the CLIP top-N before it reorders/excludes and trims.
            pool = self.repo.match_missing_pets(
                sighting_id, settings.MATCH_CANDIDATE_POOL
            )
            pool = filter_by_threshold(pool, threshold)
            matches = rerank_by_color(
                pool,
                sighting.get("primary_color_hex"),
                clip_weight=settings.CLIP_MATCH_WEIGHT,
                color_weight=settings.COLOR_MATCH_WEIGHT,
                exclude_distance=settings.COLOR_EXCLUDE_DISTANCE,
                lightness_weight=settings.COLOR_LIGHTNESS_WEIGHT,
                neutral_chroma=settings.NEUTRAL_CHROMA_THRESHOLD,
                neutral_lightness_exclude=settings.NEUTRAL_LIGHTNESS_EXCLUDE,
                limit=limit,
            )
            logger.info(
                "Found %d matches for sighting %s "
                "(pool=%d, threshold=%s, colour=%s)",
                len(matches), sighting_id, len(pool), threshold,
                sighting.get("primary_color_hex"),
            )
            return matches

        except ValueError:
            raise
        except Exception as e:
            logger.error("Error finding matches: %s", e)
            raise Exception(f"Failed to find matches: {e}")

    async def get_sighting_by_id(self, sighting_id: str) -> dict | None:
        return await asyncio.to_thread(self._get_sighting_by_id_sync, sighting_id)

    def _get_sighting_by_id_sync(self, sighting_id: str) -> dict | None:
        try:
            row = self.repo.get_sighting(sighting_id)
            if row is None:
                return None
            return strip_feature_vector(row)
        except Exception as e:
            logger.error("Error fetching sighting %s: %s", sighting_id, e)
            raise

    async def get_hunter_activity(
        self, hunter_id: str, limit: int = 50, offset: int = 0,
    ) -> dict:
        """
        Activity log for a single hunter — sightings (newest first) plus,
        per sighting, the AI match candidates and the score award (if the
        target pet has already been resolved).

        Three repo round-trips instead of a single big join: simpler to reason
        about, all three tables are small per-hunter, and there is no clean
        embedded-resource join that also covers the score_awards
        UNIQUE(pet, user) relation (which is not a FK to sightings).
        """
        return await asyncio.to_thread(
            self._get_hunter_activity_sync, hunter_id, limit, offset
        )

    def _get_hunter_activity_sync(
        self, hunter_id: str, limit: int, offset: int,
    ) -> dict:
        try:
            total_count = self.repo.count_sightings_for_hunter(hunter_id)

            sightings = self.repo.list_sightings_for_hunter(
                hunter_id, limit, offset
            )
            if not sightings:
                return {"sightings": [], "total_count": total_count}

            sighting_ids = [s["id"] for s in sightings]
            matches = self.repo.get_matches_for_sightings(sighting_ids)
            # Index awards by the sighting that earned them — a hunter has at
            # most one award per resolved pet, so fetching all of theirs is
            # cheap and avoids missing awards whose link was an AI match
            # (initial_target_pet_id NULL) rather than an explicit target.
            awards = self.repo.get_awards_for_hunter(hunter_id)

            sightings = assemble_hunter_activity(sightings, matches, awards)
            return {"sightings": sightings, "total_count": total_count}

        except Exception as e:
            # exception(), not error(): the useful part of a transport-level
            # failure (e.g. httpx.ReadError) is the traceback, and the route
            # turns this into an opaque 500 for the caller.
            logger.exception("Error fetching activity for hunter %s: %s",
                             hunter_id, e)
            raise

    async def get_hunter_stats(self, hunter_id: str) -> dict:
        """Cumulative stats card for the hunter profile screen."""
        return await asyncio.to_thread(self._get_hunter_stats_sync, hunter_id)

    def _get_hunter_stats_sync(self, hunter_id: str) -> dict:
        try:
            user = self.repo.get_user(hunter_id)
            total_score = user["total_score"] if user else 0

            return {
                "total_score": total_score,
                "sightings_submitted":
                    self.repo.count_sightings_for_hunter(hunter_id),
                "sightings_verified":
                    self.repo.count_verified_sightings_for_hunter(hunter_id),
                "resolutions_contributed_to":
                    self.repo.count_contributions_for_hunter(hunter_id),
            }
        except Exception as e:
            logger.exception("Error fetching stats for hunter %s: %s",
                             hunter_id, e)
            raise
