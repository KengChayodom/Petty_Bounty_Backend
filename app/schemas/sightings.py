from pydantic import BaseModel

class SightingCreate(BaseModel):
    hunter_id: str    
    image_url: str   
    latitude: float
    longitude: float