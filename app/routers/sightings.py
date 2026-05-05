from fastapi import APIRouter, HTTPException
from app.schemas.sightings import SightingCreate, AnalyzeRequest
from app.core.database import supabase
from ultralytics import YOLO

router = APIRouter(prefix="/sightings", tags=["Sightings"])

yolo_model = YOLO("yolo11s-seg.pt")
TARGET_ANIMALS = {"bird", "cat", "dog"} 

@router.post("/analyze")
async def analyze_sighting_image(request: AnalyzeRequest):
    try:
        detected_species = "Unknown"
        highest_conf = 0.0
        bounding_box = None 
        
        # Run YOLO prediction (Only happens here now!)
        results = yolo_model.predict(source=request.image_url, conf=0.25)
        
        if len(results) > 0:
            result = results[0]
            
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id].lower() 
                confidence = float(box.conf[0])
                
                if class_name in TARGET_ANIMALS and confidence > highest_conf:
                    detected_species = class_name.capitalize() 
                    highest_conf = confidence
                    bounding_box = box.xyxy[0].tolist() 
                    
        if detected_species == "Unknown":
            return {
                "status": "not_found",
                "message": "No target animals detected in this image.",
            }
            
        return {
            "status": "success",
            "message": f"AI detected a {detected_species}.",
            "prompt_question": f"Is this a {detected_species}?", 
            "data": {
                "species": detected_species,
                "confidence": round(highest_conf * 100, 2),
                "bbox": bounding_box 
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@router.post("/")
async def report_sighting(sighting: SightingCreate):
    try:
        location_point = f"POINT({sighting.longitude} {sighting.latitude})"
        
        # Prepare data for Supabase 'sightings' table
        # We now use the 'detected_species' directly from the frontend request!
        data = {
            "hunter_id": sighting.hunter_id,
            "sighted_location": location_point,  
            "image_url": sighting.image_url,
            "detected_species": sighting.detected_species, # Confirmed by user
            "action_type": "Spotted",           
            "sighting_status": "Pending_Analysis" 
        }
        
        response = supabase.table("sightings").insert(data).execute()

        return {
            "status": "success", 
            "message": "Sighting reported successfully with user confirmation!",
            "data": response.data
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))