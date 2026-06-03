# app/api/missing_pets.py
from fastapi import APIRouter, HTTPException, Depends, Query
from app.schemas.missing_pets import MissingPetCreate, MissingPetUpdate
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
        return StandardResponse(
            status="success",
            message="Pet registered successfully.",
            data=data
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register pet: {str(e)}")
    

@router.get("/nearby", response_model=StandardResponse)
async def get_nearby_missing_pets(
    latitude: float = Query(..., ge=-90.0, le=90.0, description="Center point latitude"),
    longitude: float = Query(..., ge=-180.0, le=180.0, description="Center point longitude"),
    radius_km: float = Query(10.0, ge=0.1, le=100.0, description="Search radius in kilometers"),
    supabase = Depends(get_supabase_client)
):
    try:
        pets = await PetService.get_nearby_missing_pets(
            supabase, latitude, longitude, radius_km, limit=20
        )
        return StandardResponse(
            status="success",
            message=f"Found {len(pets)} nearby missing pets.",
            data=pets
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find nearby pets: {str(e)}")


@router.get("/{pet_id}/sightings", response_model=StandardResponse)
async def get_sightings_for_pet(
    pet_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    supabase = Depends(get_supabase_client)
):
    """
    Owner-facing list of sightings reported against this missing pet.
    Includes AI-matched sightings AND any sighting whose hunter explicitly
    tagged this pet via `initial_target_pet_id` (per product requirement).
    Newest first.

    Owner-only access check is deferred to Feature #6 (Auth).
    """
    try:
        data = await PetService.get_sightings_for_pet(
            supabase, pet_id, limit=limit, offset=offset,
        )
        return StandardResponse(
            status="success",
            message=f"Retrieved {len(data)} sightings.",
            data=data,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch sightings for pet: {str(e)}",
        )


@router.get("/{pet_id}", response_model=StandardResponse)
async def get_missing_pet(
    pet_id: str,
    supabase = Depends(get_supabase_client)
):
    try:
        pet = await PetService.get_missing_pet_by_id(supabase, pet_id)
        if not pet:
            raise HTTPException(status_code=404, detail=f"Missing pet {pet_id} not found")
        return StandardResponse(
            status="success",
            message="Pet retrieved successfully.",
            data=pet
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve pet: {str(e)}")

@router.patch("/{pet_id}", response_model=StandardResponse)
async def update_missing_pet(
    pet_id: str,
    update: MissingPetUpdate,
    supabase = Depends(get_supabase_client)
):
    try:
        # Build update payload with only provided fields
        payload = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}

        if not payload:
            raise HTTPException(status_code=400, detail="No fields to update")

        response = supabase.table("missing_pets").update(payload).eq("id", pet_id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail=f"Missing pet {pet_id} not found")

        return StandardResponse(
            status="success",
            message="Pet updated successfully.",
            data=response.data[0]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update pet: {str(e)}")

