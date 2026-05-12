# routers/upload.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.core.database import get_supabase_client
import uuid

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("/pet-image")
async def upload_pet_image(
    file: UploadFile = File(...),
    supabase = Depends(get_supabase_client) 
):
    try:
        file_bytes = await file.read()
        
        file_extension = file.filename.split(".")[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        
        response = supabase.storage.from_("pet-images").upload(
            path=unique_filename,
            file=file_bytes,
            file_options={"content-type": file.content_type}
        )
        public_url = supabase.storage.from_("pet-images").get_public_url(unique_filename)
        
        return {
            "status": "success", 
            "message": "Upload Complete!",
            "image_url": public_url
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))