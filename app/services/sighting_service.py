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

from app.schemas.sightings import SightingCreate
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

    async def process_and_save_sighting(self, sighting: SightingCreate) -> dict:
        """
        Hot path: pull cached feature_vector, INSERT with user-confirmed
        species, run pgvector match RPC, return {sighting, matches}.
        """
        try:
            image_url = str(sighting.image_url)
            cached = AnalyzeCache.get(image_url)

            if cached is not None:
                vector = cached["feature_vector"]
                logger.warning(
                    "Analyze-cache HIT key=%r — reusing CLIP vector", image_url
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

            # CRITICAL: detected_species is the CLIENT-supplied (user-confirmed)
            # value, NOT YOLO's cached guess. This is the entire reason for
            # the verification screen — the user can override misclassification.
            location = create_postgis_point(sighting.latitude, sighting.longitude)
            payload = {
                "hunter_id": sighting.hunter_id,
                "sighted_location": location,
                "image_url": image_url,
                "detected_species": sighting.detected_species,  # user-confirmed
                "feature_vector": vector,
                "action_type": "Spotted",
                "sighting_status": "Pending_Analysis",
            }
            res = self.db.table("sightings").insert(payload).execute()
            if not res.data:
                raise ValueError("Insert failed: No data returned from Supabase")

            sighting_row = res.data[0]
            sighting_id = sighting_row["id"]
            logger.info(
                "Sighting %s saved (species=%s)", sighting_id, sighting.detected_species
            )

            # Bundle matches in the same response — saves Flutter a round-trip.
            # If the RPC fails the row is still saved; client can re-query
            # via GET /sightings/{id}/matches.
            try:
                matches = await self.get_matches(
                    sighting_id, limit=5, threshold=0.0
                )
            except Exception as e:
                logger.warning(
                    "Sighting %s saved but matches failed: %s", sighting_id, e
                )
                matches = []

            # Strip the 512-D string from the response — clients don't use it
            # and it bloats payloads.
            sighting_row.pop("feature_vector", None)
            return {"sighting": sighting_row, "matches": matches}

        except ValueError:
            raise
        except Exception as e:
            logger.error("Error in process_and_save_sighting: %s", e)
            raise

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

    async def update_sighting_status(
        self, sighting_id: str, status: str,
    ) -> dict:
        valid_statuses = [
            "Pending_Analysis", "Notified_Owner", "Confirmed", "Closed"
        ]
        if status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        try:
            res = (self.db.table("sightings")
                          .update({"sighting_status": status})
                          .eq("id", sighting_id)
                          .execute())
            if not res.data:
                raise ValueError(f"Sighting {sighting_id} not found")
            return res.data[0]
        except ValueError:
            raise
        except Exception as e:
            logger.error("Error updating sighting status: %s", e)
            raise
