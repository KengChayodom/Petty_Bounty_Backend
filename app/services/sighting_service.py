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
"""
import logging

from app.schemas.sightings import SightingCreate, TargetedSightingCreate
from app.services.ai_cache import AnalyzeCache
from app.utils.postgis import create_postgis_point

logger = logging.getLogger(__name__)


class SightingService:
    """Coordinates the AI pipeline, sighting persistence, and pgvector match."""

    def __init__(self, db_client, ai_manager):
        self.db = db_client
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

            cache_key = str(image_url)
            AnalyzeCache.set(cache_key, {
                "pil_image": image,
                "isolated_image": isolated_image,
                "species": species,
                "bbox": bbox,
                "confidence": confidence,
                "feature_vector": feature_vector,
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
        self, sighting, *, vector, target_pet_id: str | None
    ) -> dict:
        """
        Build the sighting payload, INSERT it, and return the saved row with
        the 512-D feature_vector stripped out (clients never use it and it
        bloats payloads). Shared by the discovery and targeted paths so the
        INSERT contract lives in one place.

        CRITICAL: detected_species is the CLIENT-supplied (user-confirmed)
        value, NOT YOLO's guess — the whole point of the verification screen.

        `vector` is the CLIP embedding for discovery, or None for targeted
        (column left NULL). `target_pet_id` is set only for targeted reports.
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

        res = self.db.table("sightings").insert(payload).execute()
        if not res.data:
            raise ValueError("Insert failed: No data returned from Supabase")

        sighting_row = res.data[0]
        sighting_row.pop("feature_vector", None)
        return sighting_row

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

            sighting_row = self._insert_sighting_row(
                sighting, vector=vector, target_pet_id=None
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
                    self._persist_matches(sighting_id, matches)
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
            sighting_row = self._insert_sighting_row(
                sighting, vector=None, target_pet_id=sighting.target_pet_id
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

    def _persist_matches(self, sighting_id: str, matches: list[dict]) -> None:
        """
        Log AI match results into sighting_matches. The match RPC returns
        rows keyed `id` (the missing pet) and `similarity`; map them to the
        sighting_matches columns. No-op when there are no matches.

        Uses upsert on the (sighting_id, missing_pet_id) unique constraint so
        a retried request — or any future re-match path — refreshes the
        similarity_score in place instead of accumulating duplicate rows.
        """
        if not matches:
            return
        rows = [
            {
                "sighting_id": sighting_id,
                "missing_pet_id": m["id"],
                "similarity_score": m.get("similarity"),
            }
            for m in matches
            if m.get("id")
        ]
        if rows:
            (self.db.table("sighting_matches")
                    .upsert(rows, on_conflict="sighting_id,missing_pet_id")
                    .execute())

    async def get_matches(
        self,
        sighting_id: str,
        limit: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Find matching missing pets via the match_missing_pets RPC."""
        try:
            sighting_res = (self.db.table("sightings")
                                   .select("id, feature_vector, detected_species, sighted_location")
                                   .eq("id", sighting_id)
                                   .execute())
            if not sighting_res.data:
                raise ValueError(f"Sighting {sighting_id} not found")
            sighting = sighting_res.data[0]
            if not sighting.get("feature_vector"):
                raise ValueError(f"Sighting {sighting_id} has no feature vector")
            if not sighting.get("detected_species"):
                raise ValueError(f"Sighting {sighting_id} has no detected species")
            if not sighting.get("sighted_location"):
                raise ValueError(f"Sighting {sighting_id} has no location")

            res = self.db.rpc("match_missing_pets", {
                "p_sighting_id": sighting_id,
                "match_limit": limit,
            }).execute()

            matches = res.data or []
            if threshold:
                matches = [
                    m for m in matches
                    if (m.get("similarity") or 0.0) >= threshold
                ]
            logger.info(
                "Found %d matches for sighting %s (threshold=%s)",
                len(matches), sighting_id, threshold,
            )
            return matches

        except ValueError:
            raise
        except Exception as e:
            logger.error("Error finding matches: %s", e)
            raise Exception(f"Failed to find matches: {e}")

    async def get_sighting_by_id(self, sighting_id: str) -> dict | None:
        try:
            res = (self.db.table("sightings")
                          .select("*")
                          .eq("id", sighting_id)
                          .execute())
            if not res.data:
                return None
            row = res.data[0]
            row.pop("feature_vector", None)
            return row
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

        Three Supabase round-trips instead of a single big join: simpler to
        reason about, all three tables are small per-hunter, and supabase-py
        has no clean way to express a 3-table left-join with the embedded-
        resource syntax that also covers the score_awards UNIQUE(pet, user)
        relation (which is not a FK to sightings).
        """
        try:
            count_res = (self.db.table("sightings")
                                .select("id", count="exact")
                                .eq("hunter_id", hunter_id)
                                .execute())
            total_count = count_res.count or 0

            res = (self.db.table("sightings")
                          .select("id, image_url, detected_species, "
                                  "action_type, sighting_status, "
                                  "verification_status, sighted_location, "
                                  "initial_target_pet_id, created_at")
                          .eq("hunter_id", hunter_id)
                          .order("created_at", desc=True)
                          .range(offset, offset + limit - 1)
                          .execute())
            sightings = res.data or []
            if not sightings:
                return {"sightings": [], "total_count": total_count}

            sighting_ids = [s["id"] for s in sightings]

            matches_res = (self.db.table("sighting_matches")
                                  .select("sighting_id, missing_pet_id, "
                                          "similarity_score, owner_status")
                                  .in_("sighting_id", sighting_ids)
                                  .execute())
            matches_by_sighting: dict[str, list] = {}
            for m in (matches_res.data or []):
                matches_by_sighting.setdefault(m["sighting_id"], []).append(m)

            # Index awards by the sighting that earned them — a hunter has at
            # most one award per resolved pet, so fetching all of theirs is
            # cheap and avoids missing awards whose link was an AI match
            # (initial_target_pet_id NULL) rather than an explicit target.
            awards_res = (self.db.table("score_awards")
                                 .select("sighting_id, missing_pet_id, "
                                         "points, rank, awarded_at")
                                 .eq("user_id", hunter_id)
                                 .execute())
            awards_by_sighting: dict[str, dict] = {
                a["sighting_id"]: a
                for a in (awards_res.data or [])
                if a.get("sighting_id")
            }

            for s in sightings:
                s["matches"] = matches_by_sighting.get(s["id"], [])
                s["score_award"] = awards_by_sighting.get(s["id"])

            return {"sightings": sightings, "total_count": total_count}

        except Exception as e:
            logger.error("Error fetching activity for hunter %s: %s",
                         hunter_id, e)
            raise

    async def get_hunter_stats(self, hunter_id: str) -> dict:
        """Cumulative stats card for the hunter profile screen."""
        try:
            user_res = (self.db.table("users")
                               .select("total_score")
                               .eq("id", hunter_id)
                               .execute())
            total_score = (user_res.data[0]["total_score"]
                           if user_res.data else 0)

            submitted_res = (self.db.table("sightings")
                                    .select("id", count="exact")
                                    .eq("hunter_id", hunter_id)
                                    .execute())
            verified_res = (self.db.table("sightings")
                                   .select("id", count="exact")
                                   .eq("hunter_id", hunter_id)
                                   .eq("verification_status", "Verified")
                                   .execute())
            contributions_res = (self.db.table("score_awards")
                                        .select("missing_pet_id", count="exact")
                                        .eq("user_id", hunter_id)
                                        .execute())

            return {
                "total_score": total_score,
                "sightings_submitted": submitted_res.count or 0,
                "sightings_verified": verified_res.count or 0,
                "resolutions_contributed_to": contributions_res.count or 0,
            }
        except Exception as e:
            logger.error("Error fetching stats for hunter %s: %s",
                         hunter_id, e)
            raise
