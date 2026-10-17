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
from app.core.config import settings
from app.services.ai_service import AIManager
from app.services.identity_index import IdentityIndex
from app.services.pet_logic import (
    OwnerEditRefused,
    apply_owner_edit_rules,
    attach_sighting_counts,
    build_missing_pet_payload,
    count_sightings,
    is_closed_pet_status,
    normalize_browse_species,
    normalize_browse_status,
)
from app.utils.postgis import create_postgis_point

logger = logging.getLogger(__name__)


class PetService:
    """
    Service for managing missing pet reports.

    This service coordinates between the repository and AI models
    to provide feature extraction and spatial storage for missing pets.

    Its collaborators arrive through the constructor, as SightingService's do,
    so a test hands in a fake repository or AI manager instead of patching a
    module. The AI defaults are the process-wide singletons the app runs on.
    """

    def __init__(
        self,
        repo: MissingPetRepository,
        ai_manager=AIManager,
        identity_index=IdentityIndex,
    ):
        self.repo = repo
        self.ai = ai_manager
        self.identity_index = identity_index

    async def register_missing_pet(
        self, pet: MissingPetCreate
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
            # Feature extraction runs the one shared embed pipeline
            # (AIManager.embed_image), so missing_pets and sightings vectors
            # stay directly comparable for pgvector similarity.
            logger.info(f"Extracting features from pet image: {pet.image_url}")
            # with_color: when the owner sent no colour, the one measured here
            # (the same way a sighting's is) is stored instead.
            embedding = await self.ai.embed_image(
                str(pet.image_url), expected_species=pet.species,
                with_color=True,
                with_identity=settings.IDENTITY_RERANK_ENABLED,
            )
            if embedding.used_full_frame:
                logger.warning(
                    "YOLO found no %s in %s; used a full-frame embedding",
                    pet.species, pet.image_url,
                )

            data = build_missing_pet_payload(
                pet, feature_vector=embedding.feature_vector,
                measured_color_hex=embedding.primary_color_hex,
            )

            logger.info(f"Registering missing pet: {pet.pet_name}")
            created = await asyncio.to_thread(self.repo.insert_missing_pet, data)
            if embedding.identity_vector is not None:
                # Ready for the very next sighting, without waiting for a
                # match to build it on demand.
                self.identity_index.put(
                    created["id"], str(pet.image_url), embedding.identity_vector,
                )
            logger.info(f"Missing pet registered successfully: {created['id']}")
            return created

        except ValueError:
            # Re-raise ValueError with context
            raise
        except Exception as e:
            logger.error(f"Error registering missing pet: {e}")
            raise Exception(f"Failed to register missing pet: {str(e)}")

    async def get_missing_pet_by_id(
        self, pet_id: str
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
                self._get_missing_pet_by_id_sync, pet_id
            )
        except Exception as e:
            logger.error(f"Error fetching missing pet {pet_id}: {e}")
            raise

    def _get_missing_pet_by_id_sync(
        self, pet_id: str
    ) -> dict | None:
        pet = self.repo.get_missing_pet_by_id(pet_id)
        if not pet:
            # Not found is not an error and must not cost a second query.
            return None
        links = self.repo.get_sighting_links_for_pets([pet_id])
        return attach_sighting_counts([pet], count_sightings(links))[0]

    async def get_my_missing_pets(
        self, owner_id: str
    ) -> list[dict]:
        """
        The owner's "My Reports" list.

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
                self._get_my_missing_pets_sync, owner_id
            )
        except Exception as e:
            logger.error("Error listing reports for owner %s: %s", owner_id, e)
            raise

    def _get_my_missing_pets_sync(
        self, owner_id: str
    ) -> list[dict]:
        pets = self.repo.get_by_owner(owner_id)
        if not pets:
            # No pets means nothing to count — skip the round trip rather
            # than asking the database about an empty list of ids.
            return []
        links = self.repo.get_sighting_links_for_pets(
            [p["id"] for p in pets if p.get("id")]
        )
        return attach_sighting_counts(pets, count_sightings(links))

    async def list_all_missing_pets(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
        species: str | None = None,
    ) -> Page:
        """
        Platform-wide browse for moderation.

        `status=None` means "every status", not "status IS NULL"; the filter is
        applied by the repository only when one was supplied. The admin gate
        itself lives in require_admin, not here.

        `post_status` (Pending / Spotted / Expired / Rescued) is attached to
        every row so the admin console can display the derived badge instead of
        the raw `status` column (Searching / Found / Resolved). This mirrors
        exactly what `get_my_missing_pets` does for the owner's own list — the
        rule lives in `pet_logic.derive_post_status`, not here.

        Returns the page AND the total number of matching reports, so the admin
        console can draw numbered pages rather than infer a next page from a
        full one.
        """
        # Normalised BEFORE any I/O, so an unrecognised filter is a 400 at the
        # edge rather than a failed enumeration cast surfacing as a 500 — the
        # same rule the flag queue listing follows.
        status = normalize_browse_status(status)
        species = normalize_browse_species(species)
        try:
            return await asyncio.to_thread(
                self._list_all_missing_pets_sync,
                status, species, limit, offset,
            )
        except Exception as e:
            logger.error("Error listing all missing pets: %s", e)
            raise

    def _list_all_missing_pets_sync(
        self,
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
            raw = self.repo.list_all("Searching", species, limit=10_000, offset=0)
            if not raw.items:
                return Page([], 0)
            links = self.repo.get_sighting_links_for_pets(
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
        page = self.repo.list_all(status, species, limit, offset)
        if not page.items:
            return page
        links = self.repo.get_sighting_links_for_pets(
            [p["id"] for p in page.items if p.get("id")]
        )
        enriched = attach_sighting_counts(page.items, count_sightings(links))
        return Page(enriched, page.total)

    async def remove_missing_pet(
        self, pet_id: str, admin_id: str
    ) -> dict:
        """
        Remove a report that violates the guidelines.

        The removal's postcondition is "removed from the database and the search map",
        so this is a hard delete rather than a hidden flag. The moderation
        action is recorded in the log at WARNING level — the same audit
        convention verify_sighting uses.

        Raises:
            ValueError: when no such report exists (the API layer maps it to 404).
        """
        try:
            removed = await asyncio.to_thread(self.repo.remove, pet_id)
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

    async def get_sightings_for_pet(
        self,
        pet_id: str,
        limit: int = 50,
        offset: int = 0,
        *,
        owner_id: str | None = None,
    ) -> list[dict]:
        """
        Owner-facing chronological list. Delegates to the `sightings_for_pet`
        RPC which unions AI-matched (sighting_matches.missing_pet_id) with
        explicitly-targeted (sightings.initial_target_pet_id) and joins the
        hunter username. Doing it in SQL means a single round-trip and
        no client-side dedupe.

        `owner_id` scopes the read to one account's own report. It is what makes
        this method owner-facing rather than merely authenticated: the rows carry
        the place a pet was seen and the hunter's own name and telephone number,
        so a signed-in stranger reading another owner's timeline would be reading
        a third party's contact details and location history. A report the caller
        does not own is answered as a missing report, never as a refusal, so the
        endpoint cannot be used to discover which identifiers exist — the same
        rule `update_missing_pet_owned` and `decide_match` already follow.

        Passing None keeps the unscoped read for callers that have already
        established the right to see it; today nothing does, and a caller that
        omits it is the bug this parameter exists to prevent.

        Raises:
            LookupError: no such report, or it belongs to another account
                (API -> 404 for both).
        """
        try:
            if owner_id is not None:
                pet = await asyncio.to_thread(self.repo.get_missing_pet_by_id, pet_id)
                if not pet or pet.get("owner_id") != owner_id:
                    raise LookupError(
                        f"Missing pet {pet_id} not found or not owned by you"
                    )
            # owner never sees Dismissed reports -> include_dismissed=False
            return await asyncio.to_thread(
                self.repo.sightings_for_pet,
                pet_id, limit, offset, include_dismissed=False,
            )
        except LookupError:
            raise
        except Exception as e:
            logger.error("Error fetching sightings for pet %s: %s", pet_id, e)
            raise

    async def get_nearby_missing_pets(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 10.0,
        limit: int = 20
    ) -> list[dict]:
        """
        Find missing pets within a radius using PostGIS spatial query.
        """
        try:
            center_point_wkt = create_postgis_point(latitude, longitude)
            radius_meters = radius_km * 1000.0

            return await asyncio.to_thread(
                self.repo.get_nearby_missing_pets,
                center_point_wkt, radius_meters, limit,
            )

        except Exception as e:
            logger.error(f"Error finding nearby missing pets: {e}")
            raise

    async def update_owned_missing_pet(
        self, pet_id: str, owner_id: str, payload: dict,
    ) -> dict:
        """
        Apply an owner's partial edit to a report they own.

        Owner-scoped: the read and the write both match on `owner_id`, so a
        report that does not exist and one owned by somebody else are the same
        LookupError. The caller cannot use it to learn which identifiers exist.

        A status or bounty change is first checked against the report's current
        state (`apply_owner_edit_rules`), and the write only lands if that state
        still holds. Ending the search also closes the report's sightings.

        Raises:
            LookupError: no such report, or it belongs to another account.
            OwnerEditRefused: the edit is not allowed in the report's state, or
                the report changed between the read and the write.
        """
        return await asyncio.to_thread(
            self._update_owned_missing_pet_sync, pet_id, owner_id, payload,
        )

    def _update_owned_missing_pet_sync(
        self, pet_id: str, owner_id: str, payload: dict,
    ) -> dict:
        not_found = f"Missing pet {pet_id} not found or not owned by you"
        expected_status = None

        if "status" in payload or "bounty_amount" in payload:
            current = self.repo.get_owned_pet_state(pet_id, owner_id)
            if not current:
                raise LookupError(not_found)
            caught_waiting = (
                payload.get("status") == "Found"
                and self.repo.has_caught_awaiting_decision(pet_id)
            )
            payload = apply_owner_edit_rules(
                current, payload, caught_awaiting_decision=caught_waiting
            )
            expected_status = current.get("status")

        updated = self.repo.update_missing_pet_owned(
            pet_id, owner_id, payload, expected_status=expected_status,
        )

        if not updated:
            if expected_status is not None:
                # The report was read a moment ago, so no row now means its
                # status changed in between (a Caught confirmation closing it).
                # Refuse rather than act on a state that no longer holds.
                raise OwnerEditRefused(
                    "The report changed while you were editing it. "
                    "Reload it and try again."
                )
            raise LookupError(not_found)

        if is_closed_pet_status(payload.get("status")):
            self._close_sightings_after_search_ended(pet_id)
        return updated

    def _close_sightings_after_search_ended(self, pet_id: str) -> None:
        """Ending the search ends its sightings too: they stop being live leads
        the moment the pet is home. A failure here is logged, not raised, since
        the owner's report IS already updated and failing the request would
        report a failure that did not happen."""
        try:
            closed = self.repo.close_sightings_for_pet(pet_id)
            logger.info(
                "Pet %s closed by owner — %d sighting(s) marked Closed.",
                pet_id, closed,
            )
        except Exception as e:
            logger.error(
                "Pet %s closed but its sightings could NOT be closed: %s",
                pet_id, e, exc_info=True,
            )
