from fastapi import APIRouter, HTTPException, Depends
from app.schemas.missing_pets import MissingPetCreate
from app.schemas.common import StandardResponse
from app.services.pet_service import PetService
from app.core.database import get_supabase_client

router = APIRouter(prefix="/missing-pets", tags=["Missing Pets"])

@router.post("/", response_model=StandardResponse)
async def create_missing_pet(
    pet: MissingPetCreate,
    supabase = Depends(get_supabase_client)
):
    try:
        data = await PetService.register_missing_pet(supabase, pet)
        return {"status": "success", "message": "Pet registered", "data": data}
    except Exception as e:
        # Senior จะดัก Error แยกเป็นประเภท แต่ถ้าพื้นฐานใช้ 500 สำหรับ Logic Error
        raise HTTPException(status_code=500, detail=str(e))