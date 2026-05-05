from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    image_url: str
    
class SightingCreate(BaseModel):
    hunter_id: str    
    image_url: str   
    latitude: float
    longitude: float