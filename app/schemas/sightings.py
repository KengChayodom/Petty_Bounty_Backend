# schemas/sighting
from pydantic import BaseModel
from typing import Optional, List

class AnalyzeRequest(BaseModel):
    image_url: str
    
class SightingCreate(BaseModel):
    hunter_id: str    
    image_url: str   
    latitude: float
    longitude: float
    detected_species: str 
    bbox: Optional[List[float]] = None