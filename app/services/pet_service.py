"""
Pet service for managing missing pet registrations.

This service handles:
- Creating missing pet reports
- Extracting AI feature vectors from pet photos
- Geospatial data storage with PostGIS

All DB access goes through a MissingPetRepository port (app/repositories/);
this service holds zero supabase-py calls. Payload construction lives in
pet_logic.py.
"""
import logging

from app.repositories.missing_pet_repository import MissingPetRepository
from app.schemas.missing_pets import MissingPetCreate
from app.services.ai_service import AIManager
from app.services.pet_logic import build_missing_pet_payload

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
            iso = AIManager.isolate_subject(
                image, results, expected_species=pet.species
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
            created = repo.insert_missing_pet(data)
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
        """
        try:
            return repo.get_missing_pet_by_id(pet_id)
        except Exception as e:
            logger.error(f"Error fetching missing pet {pet_id}: {e}")
            raise

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
            return repo.sightings_for_pet(
                pet_id, limit, offset, include_dismissed=False
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

            return repo.get_nearby_missing_pets(
                center_point_wkt, radius_meters, limit
            )

        except Exception as e:
            logger.error(f"Error finding nearby missing pets: {e}")
            raise
