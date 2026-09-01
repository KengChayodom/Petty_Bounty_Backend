"""
Pet service for managing missing pet registrations.

This service handles:
- Creating missing pet reports
- Extracting AI feature vectors from pet photos
- Geospatial data storage with PostGIS

All DB access goes through a MissingPetRepository port (app/repositories/);
this service holds zero supabase-py calls. Payload construction lives in
pet_logic.py.

supabase-py is blocking, so every method here runs its repository work through
`asyncio.to_thread` rather than calling it inline from `async def` — see the
threading note in sighting_service.py for the full rationale.
"""
import asyncio
import logging

from app.repositories.missing_pet_repository import MissingPetRepository
from app.repositories.pagination import Page
from app.schemas.missing_pets import MissingPetCreate
from app.services.ai_service import AIManager
from app.services.pet_logic import (
    attach_sighting_counts,
    build_missing_pet_payload,
    count_sightings,
)

logger = logging.getLogger(__name__)


class PetService:
    """
    Service for managing missing pet reports.

    This service coordinates between the repository and AI models
    to provide feature extraction and spatial storage for missing pets.
    """

    @staticmethod
    async def register_missing_pet(
        repo: MissingPetRepository, pet: MissingPetCreate
    ) -> dict:
        """
        Register a new missing pet report with AI feature extraction.

        This method:
        1. Extracts feature vector from the pet photo using CLIP
        2. Creates PostGIS point for last seen location
        3. Saves missing pet record via the repository

        Raises:
            ValueError: If data validation fails or the INSERT returns no row
            Exception: If feature extraction or the DB operation fails
        """
        try:
            # Step 1: AI Feature Extraction — same mask-isolated pipeline the
            # live /sightings/ POST uses, so missing_pets and sightings vectors
            # remain directly comparable for pgvector similarity.
            logger.info(f"Extracting features from pet image: {pet.image_url}")
            image = await AIManager.download_image(str(pet.image_url))
            results = await AIManager.run_yolo_seg(image)
            # Off-thread for the same reason sighting_service does it: a
            # full-frame numpy pass on the event loop stalls every concurrent
            # request in the process.
            iso = await asyncio.to_thread(
                AIManager.isolate_subject,
                image, results, expected_species=pet.species,
            )
            if iso is None:
                logger.warning(
                    "YOLO found no %s in %s; falling back to full-frame embedding",
                    pet.species, pet.image_url,
                )
                target_image = image
            else:
                target_image, _, _, _ = iso
            feature_vector = await AIManager.clip_encode(target_image)

            # Step 2 + 3: build the payload (PostGIS point + status) ...
            data = build_missing_pet_payload(pet, feature_vector=feature_vector)

            # Step 4: Insert via the repository
            logger.info(f"Registering missing pet: {pet.pet_name}")
            created = await asyncio.to_thread(repo.insert_missing_pet, data)
            logger.info(f"Missing pet registered successfully: {created['id']}")
            return created

        except ValueError:
            # Re-raise ValueError with context
            raise
        except Exception as e:
            logger.error(f"Error registering missing pet: {e}")
            raise Exception(f"Failed to register missing pet: {str(e)}")

    @staticmethod
    async def get_missing_pet_by_id(
        repo: MissingPetRepository, pet_id: str
    ) -> dict | None:
        """
        Fetch a single missing pet by ID.

        Uses the get_missing_pet_by_id RPC (not select('*')) so the response
        projects latitude/longitude out of the last_seen_location geography
        (ST_Y/ST_X) and returns the SAME shape as get_nearby_missing_pets —
        honouring the MissingPetResponse contract. A raw select would omit
        lat/lng entirely and break clients that expect numeric coordinates.

        Carries the same derived `sighting_count` + `post_status` that the list
        endpoint attaches. The Status Tracker reads ONE pet, and without these
        it had to re-derive the badge from the raw `status` column — which it
        did with a different word list and a different closed-search test, so a
        pet at 'Resolved' read as still-searching in the app. One rule, one
        vocabulary, computed here.
        """
        try:
            return await asyncio.to_thread(
                PetService._get_missing_pet_by_id_sync, repo, pet_id
            )
        except Exception as e:
            logger.error(f"Error fetching missing pet {pet_id}: {e}")
            raise

    @staticmethod
    def _get_missing_pet_by_id_sync(
        repo: MissingPetRepository, pet_id: str
    ) -> dict | None:
        pet = repo.get_missing_pet_by_id(pet_id)
        if not pet:
            # Not found is not an error and must not cost a second query.
            return None
        links = repo.get_sighting_links_for_pets([pet_id])
        return attach_sighting_counts([pet], count_sightings(links))[0]

    @staticmethod
    async def get_my_missing_pets(
        repo: MissingPetRepository, owner_id: str
    ) -> list[dict]:
        """
        MD-34 / SRS-63 — the owner's "My Reports" list.

        Scoping is structural: the port takes an owner_id, so there is no query
        shape here that could return another owner's reports. Newest-first
        ordering is the repository's (SQL) job.

        Each report comes back with `sighting_count` and the derived
        `post_status` (Pending / Spotted / Rescued) attached, so the card can
        render its badge without a second round trip and without the client
        reimplementing the rule.
        """
        try:
            # Both round-trips as ONE thread hop — the second query's input is
            # the first's output, so they cannot overlap anyway and splitting
            # them would only pay the hand-off twice.
            return await asyncio.to_thread(
                PetService._get_my_missing_pets_sync, repo, owner_id
            )
        except Exception as e:
            logger.error("Error listing reports for owner %s: %s", owner_id, e)
            raise

    @staticmethod
    def _get_my_missing_pets_sync(
        repo: MissingPetRepository, owner_id: str
    ) -> list[dict]:
        pets = repo.get_by_owner(owner_id)
        if not pets:
            # No pets means nothing to count — skip the round trip rather
            # than asking the database about an empty list of ids.
            return []
        links = repo.get_sighting_links_for_pets(
            [p["id"] for p in pets if p.get("id")]
        )
        return attach_sighting_counts(pets, count_sightings(links))

    @staticmethod
    async def list_all_missing_pets(
        repo: MissingPetRepository,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
        species: str | None = None,
    ) -> Page:
        """
        MD-37 / SRS-66 — platform-wide browse for moderation.

        `status=None` means "every status", not "status IS NULL"; the filter is
        applied by the repository only when one was supplied. The admin gate
        itself lives in require_admin (MD-04), not here.

        `post_status` (Pending / Spotted / Expired / Rescued) is attached to
        every row so the admin console can display the derived badge instead of
        the raw `status` column (Searching / Found / Resolved). This mirrors
        exactly what `get_my_missing_pets` does for the owner's own list — the
        rule lives in `pet_logic.derive_post_status`, not here.

        Returns the page AND the total number of matching reports, so the admin
        console can draw numbered pages rather than infer a next page from a
        full one.
        """
        try:
            return await asyncio.to_thread(
                PetService._list_all_missing_pets_sync,
                repo, status, species, limit, offset,
            )
        except Exception as e:
            logger.error("Error listing all missing pets: %s", e)
            raise

    @staticmethod
    def _list_all_missing_pets_sync(
        repo: MissingPetRepository,
        status: str | None,
        species: str | None,
        limit: int,
        offset: int,
    ) -> Page:
        # "Spotted" and "Searching" (pure — no sightings yet) are both stored as
        # status='Searching' in the DB. Splitting them requires sighting_count,
        # which is only known after attach_sighting_counts(). We therefore fetch
        # ALL Searching rows first, attach counts, filter, then slice — so the
        # returned `total` is the count of the *derived* state, not the raw DB
        # bucket, and pagination arithmetic on the frontend is correct.
        #
        # For Found / Resolved / None the DB query already gives the right total
        # so we use the normal efficient path (DB-level limit + offset).
        if status in ("Spotted", "Searching"):
            raw = repo.list_all("Searching", species, limit=10_000, offset=0)
            if not raw.items:
                return Page([], 0)
            links = repo.get_sighting_links_for_pets(
                [p["id"] for p in raw.items if p.get("id")]
            )
            enriched = attach_sighting_counts(raw.items, count_sightings(links))
            if status == "Spotted":
                filtered = [p for p in enriched if p.get("sighting_count", 0) > 0]
            else:  # "Searching" — pure, no sightings yet
                filtered = [p for p in enriched if p.get("sighting_count", 0) == 0]
            total = len(filtered)
            paged = filtered[offset: offset + limit]
            return Page(paged, total)

        # Normal path: Found / Resolved / None — DB total is already correct.
        page = repo.list_all(status, species, limit, offset)
        if not page.items:
            return page
        links = repo.get_sighting_links_for_pets(
            [p["id"] for p in page.items if p.get("id")]
        )
        enriched = attach_sighting_counts(page.items, count_sightings(links))
        return Page(enriched, page.total)

    @staticmethod
    async def remove_missing_pet(
        repo: MissingPetRepository, pet_id: str, admin_id: str
    ) -> dict:
        """
        MD-38 / SRS-65 — remove a report that violates the guidelines.

        UD-14's postcondition is "removed from the database and the search map",
        so this is a hard delete rather than a hidden flag. The moderation
        action is recorded in the log at WARNING level — the same audit
        convention verify_sighting uses.

        Raises:
            ValueError: when no such report exists (the API layer maps it to 404).
        """
        try:
            removed = await asyncio.to_thread(repo.remove, pet_id)
        except Exception as e:
            logger.error("Error removing missing pet %s: %s", pet_id, e)
            raise
        if not removed:
            raise ValueError(f"Missing pet {pet_id} not found")
        logger.warning(
            "Admin %s removed missing pet %s (guideline violation)",
            admin_id, pet_id,
        )
        return removed

    @staticmethod
    async def get_sightings_for_pet(
        repo: MissingPetRepository,
        pet_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """
        Owner-facing chronological list. Delegates to the `sightings_for_pet`
        RPC which unions AI-matched (sighting_matches.missing_pet_id) with
        explicitly-targeted (sightings.initial_target_pet_id) and joins the
        hunter display name. Doing it in SQL means a single round-trip and
        no client-side dedupe.
        """
        try:
            # owner never sees Dismissed reports -> include_dismissed=False
            return await asyncio.to_thread(
                repo.sightings_for_pet,
                pet_id, limit, offset, include_dismissed=False,
            )
        except Exception as e:
            logger.error("Error fetching sightings for pet %s: %s", pet_id, e)
            raise

    @staticmethod
    async def get_nearby_missing_pets(
        repo: MissingPetRepository,
        latitude: float,
        longitude: float,
        radius_km: float = 10.0,
        limit: int = 20
    ) -> list[dict]:
        """
        Find missing pets within a radius using PostGIS spatial query.
        """
        try:
            # WKT for the RPC: longitude BEFORE latitude, space-separated
            # (no comma).
            center_point_wkt = f"POINT({longitude} {latitude})"
            radius_meters = radius_km * 1000.0

            return await asyncio.to_thread(
                repo.get_nearby_missing_pets,
                center_point_wkt, radius_meters, limit,
            )

        except Exception as e:
            logger.error(f"Error finding nearby missing pets: {e}")
            raise
