# app/schemas/missing_pets.py
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime

class MissingPetCreate(BaseModel):
    owner_id: str
    pet_name: str
    species: str
    characteristics: Dict[str, Any]  # เก็บเป็น JSON เช่น {"color": "black", "weight": "5kg"}
    bounty_amount: float
    longitude: float
    latitude: float
    last_seen_time: datetime
    image_url: str


 