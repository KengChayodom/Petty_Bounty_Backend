from fastapi import APIRouter, HTTPException
from app.schemas.sightings import SightingCreate
from app.core.database import supabase
from ultralytics import YOLO

router = APIRouter(prefix="/sightings", tags=["Sightings"])


yolo_model = YOLO("yolo11s-seg.pt")

TARGET_ANIMALS = {"bird","cat","dog"} 

@router.post("/")
async def report_sighting(sighting: SightingCreate):
    try:
      
        detected_species = "Other" 
        hightest_conf  = 0.0
        try: 
            results = yolo_model.predict(source= sighting.image_url, conf=0.25)
            if len(results) > 0:
                results = results[0]

                for box in results.boxes:
                    class_id = int(box.cls[0])
                    class_name = results.names[class_id].lower()
                    confidence = float(box.conf[0])

                    if class_name in TARGET_ANIMALS and confidence > hightest_conf:
                        detected_species = class_name.capitalize()
                        hightest_conf = confidence
        except Exception as ai_error:
            print(f"YOLO ERROR: {ai_error}")


            
        location_point = f"POINT({sighting.longitude} {sighting.latitude})"
        
        # จัดเตรียมข้อมูลให้ตรงกับตาราง sightings ใน Database
        data = {
            "hunter_id": sighting.hunter_id,
            "sighted_location": location_point,  
            "image_url": sighting.image_url,
            "detected_species": detected_species, 
            "action_type": "Spotted",           
            "sighting_status": "Pending_Analysis" 
        }
        
        response = supabase.table("sightings").insert(data).execute()

        return {
            "status": "success", 
            "message": "Tip received and AI has completed initial analysis!",
            "ai_result": {
                "species": detected_species,
                "confidence": hightest_conf
            },
            "data": response.data
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))