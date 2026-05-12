import logging
from app.schemas.sightings import SightingCreate

logger = logging.getLogger(__name__)

class SightingService:
    TARGET_ANIMALS = {"bird", "cat", "dog"}

    def __init__(self, db_client, ai_manager):
        # Dependency Injection: รับ Client เข้ามาทำงาน
        self.db = db_client
        self.ai = ai_manager

    async def analyze_sighting_image(self, image_url: str):
        """สำหรับขั้นตอน /analyze เพื่อหา Bounding Box และชนิดสัตว์"""
        try:
            # ตรวจสอบชื่อฟังก์ชันใน AIManager ให้ตรงกัน (ใช้ analyze_sighting)
            results = await self.ai.analyze_sighting(str(image_url), conf=0.25)
            
            detected_species = "Unknown"
            highest_conf = 0.0
            bounding_box = None 
            
            if results and len(results) > 0:
                result = results[0]
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id].lower() 
                    confidence = float(box.conf[0])
                    
                    if class_name in self.TARGET_ANIMALS and confidence > highest_conf:
                        detected_species = class_name.capitalize() 
                        highest_conf = confidence
                        bounding_box = box.xyxy[0].tolist() 
                        
            if detected_species == "Unknown":
                return {"status": "not_found", "message": "No target animals detected."}
                
            return {
                "status": "success",
                "message": f"AI detected a {detected_species}.",
                "data": {
                    "species": detected_species,
                    "confidence": round(highest_conf * 100, 2),
                    "bbox": bounding_box 
                }
            }
        except Exception as e:
            logger.error(f"Error in analyze_sighting_image: {e}")
            raise e

    async def process_and_save_sighting(self, sighting: SightingCreate):
        """สำหรับขั้นตอน POST / เพื่อบันทึกเบาะแสลง Database"""
        try:
            # 1. สกัด Vector (ใช้ฟังก์ชันใน AIManager)
            vector = await self.ai.extract_vector_from_bbox(
                str(sighting.image_url), 
                sighting.bbox
            )
            
            # 2. เตรียมข้อมูลพิกัด (PostGIS POINT)
            location = f"POINT({sighting.longitude} {sighting.latitude})"
            
            payload = {
                "hunter_id": sighting.hunter_id,
                "sighted_location": location,
                "image_url": str(sighting.image_url),
                "detected_species": sighting.detected_species,
                "feature_vector": vector,
                "action_type": "Spotted",
                "sighting_status": "Pending_Analysis"
            }
            
            # 3. บันทึกผ่าน Supabase Client
            res = self.db.table("sightings").insert(payload).execute()
            
            if not res.data:
                raise ValueError("Insert failed: No data returned from Supabase")
                
            return res.data[0]
        except Exception as e:
            logger.error(f"Error in process_and_save_sighting: {e}")
            raise e
        
    async def get_matches(self, sighting_id: str, limit: int = 5):
      
        try:
          
            res = self.db.rpc('match_missing_pets', {
                'p_sighting_id': sighting_id,
                'match_limit': limit
            }).execute()
            
            return res.data if res.data else []
        except Exception as e:
            logger.error(f"Error finding matches: {e}")
            raise e