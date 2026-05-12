import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def test_ranking_system():
    print("--- 🧪 Starting Ranking System Validation ---")

    # สมมติว่าคุณอัปโหลดรูปและได้ URL จาก Supabase Storage มาแล้ว
    # ในการเทสจริง ให้เรียก POST /upload/pet-image ก่อนเพื่อให้ได้ URL เหล่านี้
    missing_dog_url = "URL_ของ_dog_golden_1"
    missing_cat_url = "URL_ของ_cat_black_1"
    sighting_dog_url = "URL_ของ_dog_golden_2"

    print("1. บันทึกข้อมูลสัตว์ที่หายไป (Missing Pets)...")
    # ตัวที่ 1: โกลเด้น
    requests.post(f"{BASE_URL}/missing-pets/", json={
        "owner_id": "user_1", "pet_name": "Golden Boy", "species": "Dog",
        "characteristics": {"color": "gold"}, "bounty_amount": 1000,
        "longitude": 100.0, "latitude": 13.0, "last_seen_time": "2026-05-01T00:00:00Z",
        "image_url": missing_dog_url
    })
    
    # ตัวที่ 2: แมวดำ (Noise)
    requests.post(f"{BASE_URL}/missing-pets/", json={
        "owner_id": "user_2", "pet_name": "Shadow", "species": "Cat",
        "characteristics": {"color": "black"}, "bounty_amount": 500,
        "longitude": 100.1, "latitude": 13.1, "last_seen_time": "2026-05-02T00:00:00Z",
        "image_url": missing_cat_url
    })

    print("2. สร้างรายงานการพบเจอ (Sighting)...")
    # ส่งเบาะแสเป็นรูปโกลเด้นอีกมุม
    sighting_res = requests.post(f"{BASE_URL}/sightings/", json={
        "hunter_id": "hunter_1", "image_url": sighting_dog_url,
        "latitude": 13.05, "longitude": 100.05, "detected_species": "Dog",
        "bbox": [0, 0, 500, 500] # สมมติ BBox ครอบทั้งรูป
    })
    sighting_data = sighting_res.json()
    sighting_id = sighting_data["data"][0]["id"] # ดึง ID ที่เพิ่งสร้าง

    print("3. ดึงผลลัพธ์ Ranking...")
    ranking_res = requests.get(f"{BASE_URL}/sightings/{sighting_id}/matches?limit=5")
    matches = ranking_res.json()["matches"]

    print("\n--- 📊 Ranking Results ---")
    for i, match in enumerate(matches):
        print(f"Rank {i+1}: {match['pet_name']} (Similarity: {match['similarity']:.4f})")

    # --- การทำ Assertion (วัดผล) ---
    assert matches[0]["pet_name"] == "Golden Boy", "❌ TEST FAILED: ระบบ Ranking ผิดพลาด แมวดำดันเหมือนโกลเด้นมากกว่า!"
    assert matches[0]["similarity"] > 0.70, "❌ TEST WARNING: ความคล้ายต่ำเกินไป โมเดล CLIP อาจจะสกัดฟีเจอร์ไม่ดีพอ"
    
    print("✅ TEST PASSED: ระบบ Ranking ทำงานได้ถูกต้องตามหลักเรขาคณิตของ Vector!")

if __name__ == "__main__":
    test_ranking_system()