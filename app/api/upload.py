from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.dependencies import get_upload_service
from app.core.auth import get_current_user_id
from app.schemas.common import StandardResponse
from app.services.upload_service import UploadService

# Every route here needs a signed-in caller, declared ONCE for the whole router so a route added later inherits it.
# The routes still ask for the id where they use it, which FastAPI resolves once per request.
router = APIRouter(
    prefix="/upload", tags=["Upload"],
    dependencies=[Depends(get_current_user_id)],
)


@router.post("/pet-image", response_model=StandardResponse)
async def upload_pet_image(
    file: UploadFile = File(...),
    service: UploadService = Depends(get_upload_service),
    user_id: str = Depends(get_current_user_id),
):
    """Store a lost-pet or profile photo and return its public URL.

    The sightings flow does not come through here (Flutter uploads straight to
    Storage). A file that fails validation answers 400 and nothing is written.
    """
    try:
        data = await service.upload_pet_image(
            file.filename, file.content_type, await file.read(),
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    return StandardResponse(
        status="success",
        message="Upload complete!",
        data=data,
    )
