#services/ai_service.py
import io
import requests
import asyncio
from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class AIManager:
    _yolo = None
    _clip = None

    @classmethod
    def get_yolo(cls):
        if cls._yolo is None :
            logger.info("Loading YOLO model...")
            cls._yolo = YOLO("yolo11s-seg.pt")
        return cls._yolo

    @classmethod
    def get_clip(cls):
        if cls._clip is None:
            logger.info("Loading CLIP model...")
            cls._clip = SentenceTransformer('clip-ViT-B-32')
        return cls._clip

    @classmethod
    async def analyze_sighting(cls, image_url: str, conf: float = 0.25):
        """สำหรับตรวจจับชนิดสัตว์ด้วย YOLO"""
        model = cls.get_yolo()
        # สั่งรันแบบไม่บล็อกเซิร์ฟเวอร์
        return await asyncio.to_thread(model.predict, source=image_url, conf=conf)
    
    from PIL import Image

    @classmethod
    async def extract_vector_from_bbox(cls, image_url: str, bbox: list[float] = None) -> list[float]:
        """
        รับรูปภาพ และ Bounding Box จาก YOLO
        ตัดเฉพาะส่วนของสัตว์ (Crop) ก่อนนำไปสกัด Vector ด้วย CLIP
        """
        model = cls.get_clip()
        
        try:
            # โหลดรูปภาพ
            response = await asyncio.to_thread(requests.get, image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            
            # 🟢 โค้ดพระเอกของคุณ: การ Crop รูปด้วย Bounding Box จาก YOLO!
            if bbox and len(bbox) == 4:
                # bbox ปกติจะมาในรูปแบบ [x_min, y_min, x_max, y_max]
                # เราอาจจะขยายกรอบนิดหน่อย (Padding) เผื่อ YOLO ตัดหางแหว่ง
                padding = 10 
                x1 = max(0, bbox[0] - padding)
                y1 = max(0, bbox[1] - padding)
                x2 = min(image.width, bbox[2] + padding)
                y2 = min(image.height, bbox[3] + padding)
                
                # ตัดรูปเอาเฉพาะตัวสัตว์!
                image = image.crop((x1, y1, x2, y2))
                logger.info("Successfully cropped image using YOLO bounding box.")

            # นำรูป "สัตว์เพียวๆ" ไปแปลงเป็น Vector
            vector = await asyncio.to_thread(model.encode, image)
            return vector.tolist()
            
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            raise ValueError(f"Cannot load or process image: {e}")