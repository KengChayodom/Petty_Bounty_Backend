"""
Pet service for managing missing pet registrations.

This service handles:
- Creating missing pet reports
- Extracting AI feature vectors from pet photos
- Geospatial data storage with PostGIS
"""
import logging
from datetime import datetime

from app.schemas.missing_pets import MissingPetCreate
from app.services.ai_service import AIManager
from app.utils.postgis import create_postgis_point

logger = logging.getLogger(__name__)


class PetService:
    """
    Service for managing missing pet reports.

    This service coordinates between the database and AI models
    to provide feature extraction and spatial storage for missing pets.
    """

    @staticmethod
    async def register_missing_pet(supabase, pet: MissingPetCreate) -> dict:
        """
        Register a new missing pet report with AI feature extraction.

        This method:
        1. Extracts feature vector from the pet photo using CLIP
        2. Creates PostGIS point for last seen location
        3. Saves missing pet record to database

        Args:
            supabase: Supabase client instance
            pet: MissingPetCreate schema with pet details

        Returns:
            Created missing pet record

        Raises:
            ValueError: If data validation fails or database insertion fails
            Exception: If feature extraction or database operation fails
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

            # Step 2: Create PostGIS point for location
            location_point = create_postgis_point(pet.latitude, pet.longitude)

            # Step 3: Prepare data payload
            data = {
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

            # Step 4: Insert into database
            logger.info(f"Registering missing pet: {pet.pet_name}")
            response = supabase.table("missing_pets").insert(data).execute()

            if not response.data:
                raise ValueError("Database insertion failed - no data returned")

            logger.info(f"Missing pet registered successfully: {response.data[0]['id']}")
            return response.data[0]

        except ValueError:
            # Re-raise ValueError with context
            raise
        except Exception as e:
            logger.error(f"Error registering missing pet: {e}")
            raise Exception(f"Failed to register missing pet: {str(e)}")

    @staticmethod
    async def get_missing_pet_by_id(supabase, pet_id: str) -> dict | None:
        """
        Fetch a single missing pet by ID.

        Args:
            supabase: Supabase client instance
            pet_id: UUID of the missing pet

        Returns:
            Missing pet record or None if not found
        """
        try:
            response = supabase.table("missing_pets")\
                .select("*")\
                .eq("id", pet_id)\
                .execute()

            return response.data[0] if response.data else None

        except Exception as e:
            logger.error(f"Error fetching missing pet {pet_id}: {e}")
            raise

    @staticmethod
    async def update_missing_pet_status(
        supabase,
        pet_id: str,
        status: str
    ) -> dict:
        """
        Update the status of a missing pet.

        Args:
            supabase: Supabase client instance
            pet_id: UUID of the missing pet
            status: New status (Searching, Spotted, Found)

        Returns:
            Updated missing pet record

        Raises:
            ValueError: If pet not found or status is invalid
        """
        valid_statuses = ["Searching", "Spotted", "Found"]

        if status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")

        try:
            response = supabase.table("missing_pets")\
                .update({
                    "status": status,
                    "updated_at": datetime.utcnow().isoformat()
                })\
                .eq("id", pet_id)\
                .execute()

            if not response.data:
                raise ValueError(f"Missing pet {pet_id} not found")

            logger.info(f"Missing pet {pet_id} status updated to {status}")
            return response.data[0]

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error updating missing pet status: {e}")
            raise

    @staticmethod
    async def get_sightings_for_pet(
        supabase,
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
            response = supabase.rpc("sightings_for_pet", {
                "p_pet_id":            pet_id,
                "p_limit":             limit,
                "p_offset":            offset,
                "p_include_dismissed": False,  # owner never sees Dismissed reports
            }).execute()
            return response.data or []
        except Exception as e:
            logger.error("Error fetching sightings for pet %s: %s", pet_id, e)
            raise

    @staticmethod
    async def get_nearby_missing_pets(
        supabase,
        latitude: float,
        longitude: float,
        radius_km: float = 10.0,
        limit: int = 20
    ) -> list[dict]:
        """
        Find missing pets within a radius using PostGIS spatial query.
        """
        try:
            # เปลี่ยนมาใช้ WKT Format สำหรับส่งเข้า RPC โดยเฉพาะ
            # ต้องเอา longitude ขึ้นก่อน latitude และคั่นด้วยช่องว่าง (ห้ามมีลูกน้ำ)
            center_point_wkt = f"POINT({longitude} {latitude})"
            
            radius_meters = radius_km * 1000.0

            # Execute via RPC
            response = supabase.rpc('get_nearby_missing_pets', {
                'center_location': center_point_wkt, # ส่งตัวแปร WKT เข้าไปแทน
                'radius_meters': radius_meters,
                'limit': limit
            }).execute()

            return response.data if response.data else []

        except Exception as e:
            logger.error(f"Error finding nearby missing pets: {e}")
            raise
