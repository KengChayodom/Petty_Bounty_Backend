# core/database.py
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")

# สร้างตัวเชื่อมต่อ (Client) ที่ไฟล์อื่นสามารถดึงไปใช้ได้
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)