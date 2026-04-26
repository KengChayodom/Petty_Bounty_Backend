import os
from fastapi import FastAPI
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

app = FastAPI(title="Petty Bounty API")

# ตั้งค่าเชื่อมต่อ Supabase
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

@app.get("/")
def root():
    return {"message": "Petty Bounty API is running! http://127.0.0.1:8000/docs"}

# API ทดสอบดึงข้อมูลจากตาราง users
@app.get("/test-db")
def test_db():
    try:
        response = supabase.table("users").select("*").execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)