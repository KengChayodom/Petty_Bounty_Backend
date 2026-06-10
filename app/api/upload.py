from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client
from app.schemas.common import StandardResponse
import uuid

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("/pet-image", response_model=StandardResponse)
async def upload_pet_image(
    file: UploadFile = File(...),
    supabase = Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    try:
        # Validate file exists
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No filename provided"
            )

        # Validate file type
        allowed_extensions = {"jpg", "jpeg", "png", "webp"}
        file_extension = file.filename.split(".")[-1].lower() if file.filename else ""

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )

        # Validate content type
        allowed_content_types = {"image/jpeg", "image/png", "image/webp"}
        if file.content_type not in allowed_content_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content type. Allowed: {', '.join(allowed_content_types)}"
            )

        file_bytes = await file.read()

        # Check file size (max 10MB)
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")

        unique_filename = f"{uuid.uuid4()}.{file_extension}"

        response = supabase.storage.from_("pet-images").upload(
            path=unique_filename,
            file=file_bytes,
            file_options={"content-type": file.content_type}
        )
        public_url = supabase.storage.from_("pet-images").get_public_url(unique_filename)

        return StandardResponse(
            status="success",
            message="Upload complete!",
            data={"image_url": public_url, "filename": unique_filename}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
