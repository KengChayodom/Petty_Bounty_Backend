# routers/missing_pets.py    
from fastapi import APIRouter, HTTPException
from app.schemas.missing_pets import MissingPetCreate
from app.core.database import supabase

router = APIRouter(prefix="/missing-pets", tags=["Missing Pets"])

@router.post("/")
def create_missing_pet(pet: MissingPetCreate):
    try:
   
        location_point = f"POINT({pet.longitude} {pet.latitude})"

        data = {
            "owner_id": pet.owner_id,
            "pet_name": pet.pet_name,
            "species": pet.species,
            "characteristics": pet.characteristics,
            "bounty_amount": pet.bounty_amount,
            "last_seen_location": location_point,
            "last_seen_time": pet.last_seen_time.isoformat(),
            "image_url": pet.image_url,
            "status": "Searching" 
        }

        # สั่งบันทึกลง Database
        response = supabase.table("missing_pets").insert(data).execute()
        
        return {"status": "success", "data": response.data}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))