"""Pure, I/O-free logic extracted from PetService."""
from app.utils.postgis import create_postgis_point


def build_missing_pet_payload(pet, *, feature_vector) -> dict:
    """The missing_pets INSERT payload — location as a PostGIS POINT, status
    seeded to 'Searching', and the CLIP feature vector attached."""
    location_point = create_postgis_point(pet.latitude, pet.longitude)
    return {
        "owner_id": pet.owner_id,
        "pet_name": pet.pet_name,
        "species": pet.species,
        "characteristics": pet.characteristics,
        "bounty_amount": pet.bounty_amount,
        "last_seen_location": location_point,
        "last_seen_time": pet.last_seen_time.isoformat(),
        "image_url": str(pet.image_url),
        "feature_vector": feature_vector,
        "status": "Searching",
        "primary_color_hex": pet.primary_color_hex,
        "pattern_id": pet.pattern_id,
    }
