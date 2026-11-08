"""
End-to-end smoke test for the optimised 2-step pipeline.

Flow exercised:
  1. POST /missing-pets/  (×2 — seed two missing pets)
  2. POST /sightings/analyze  (heavy: YOLO + CLIP, caches vector)
  3. POST /sightings/  (light: pulls vector from cache, INSERTs with the
     species we send, runs the match RPC, returns {sighting, matches})

Assumes the FastAPI server is running at BASE_URL and the three image
URLs below have already been uploaded to Supabase Storage.
"""
import requests

BASE_URL = "http://127.0.0.1:8000"


def test_ranking_system():
    print("--- 🧪 Starting Ranking System Validation ---")

    missing_dog_url = "URL_ของ_dog_golden_1"
    missing_cat_url = "URL_ของ_cat_black_1"
    sighting_dog_url = "URL_ของ_dog_golden_2"

    print("1. Registering missing pets…")
    requests.post(f"{BASE_URL}/missing-pets/", json={
        "owner_id": "user_1", "pet_name": "Golden Boy", "species": "Dog",
        "characteristics": {"color": "gold"}, "bounty_amount": 1000,
        "longitude": 100.0, "latitude": 13.0,
        "last_seen_time": "2026-05-01T00:00:00Z",
        "image_url": missing_dog_url,
    })
    requests.post(f"{BASE_URL}/missing-pets/", json={
        "owner_id": "user_2", "pet_name": "Shadow", "species": "Cat",
        "characteristics": {"color": "black"}, "bounty_amount": 500,
        "longitude": 100.1, "latitude": 13.1,
        "last_seen_time": "2026-05-02T00:00:00Z",
        "image_url": missing_cat_url,
    })

    print("2. POST /sightings/analyze  (heavy step — YOLO + CLIP)…")
    analyze_res = requests.post(
        f"{BASE_URL}/sightings/analyze",
        json={"image_url": sighting_dog_url},
    )
    analyze_res.raise_for_status()
    analyze_data = analyze_res.json()["data"]
    detected_species = analyze_data["species"]
    print(f"   detected: {detected_species} "
          f"({analyze_data['confidence']}% confidence)")

    print("3. POST /sightings/  (light step — cache hit + DB insert + match)…")
    # In the real Flutter flow the user can override `detected_species`
    # here. For the test we just echo back what YOLO said.
    sighting_res = requests.post(f"{BASE_URL}/sightings/", json={
        "hunter_id": "hunter_1",
        "image_url": sighting_dog_url,
        "latitude": 13.05,
        "longitude": 100.05,
        "detected_species": detected_species,
    })
    sighting_res.raise_for_status()
    body = sighting_res.json()
    sighting_id = body["data"]["sighting"]["id"]
    matches = body["data"]["matches"]
    print(f"   sighting_id: {sighting_id}")

    print("\n--- 📊 Ranking Results ---")
    for i, m in enumerate(matches):
        print(f"Rank {i+1}: {m['pet_name']} (similarity: {m['similarity']:.4f})")

    assert matches, "❌ TEST FAILED: no matches returned"
    assert matches[0]["pet_name"] == "Golden Boy", (
        "❌ TEST FAILED: Ranking misordered — black cat shouldn't beat golden dog."
    )
    assert matches[0]["similarity"] > 0.70, (
        "❌ TEST WARNING: CLIP similarity too low for a same-breed match."
    )

    print("✅ TEST PASSED: 2-step pipeline + ranking work end-to-end.")


if __name__ == "__main__":
    test_ranking_system()
