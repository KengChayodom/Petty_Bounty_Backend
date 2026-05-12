# services/pet_service.py
from app.schemas.missing_pets import MissingPetCreate
from app.services.ai_service import AIManager

class PetService:
    @staticmethod
    async def register_missing_pet(supabase, pet: MissingPetCreate):
        # 1. AI Extraction (สกัด Vector)
        feature_vector = await AIManager.extract_vector_from_bbox(str(pet.image_url))

        # 2. Data Preparation (เตรียมข้อมูล)
        location_point = f"POINT({pet.longitude} {pet.latitude})"
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
            "status": "Searching"
        }

        # 3. Database Operation
        response = supabase.table("missing_pets").insert(data).execute()
        if not response.data:
            raise ValueError("Database insertion failed")
        return response.data[0]