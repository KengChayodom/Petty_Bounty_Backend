"""
Petty Bounty FastAPI Application.

This is the main entry point for the FastAPI backend.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import close_supabase_client
from app.core.firebase import init_firebase
from app.api import (
    admin, auth, dev_auth, devices, hunters, me, missing_pets, reports,
    sightings, upload,
)
# 💡 นำเข้า AIManager เพื่อใช้โหลดโมเดลตอนสตาร์ทเซิร์ฟเวอร์
from app.services.ai_service import AIManager

logger = logging.getLogger(__name__)


# Pin OMP/MKL thread count to the host's CPU count BEFORE torch / onnxruntime
# import any further. Without this, ORT and torch each spin up their own pool
# and oversubscribe the CPU, which on small servers thrashes more than it
# parallelises. Safe-default; override with the env var per deploy.
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # --- Startup (ทำงานก่อนที่เซิร์ฟเวอร์จะเปิดรับ Request แรก) ---
    if settings.ENABLE_UNAUTHED_ADMIN:
        logger.warning(
            "ENABLE_UNAUTHED_ADMIN=true — /admin/* endpoints are open with "
            "NO authentication. Acceptable for local dev only. Disable "
            "before any shared or public deploy."
        )
    else:
        logger.info(
            "Admin endpoints gated: require_admin will 503 until "
            "ENABLE_UNAUTHED_ADMIN=true or Feature #6 ships real auth."
        )

    # Initialize Firebase Admin for FCM push (SRS-FR-12). Non-fatal: logs a
    # warning and disables push if FIREBASE_CREDENTIALS is unset/invalid.
    init_firebase()

    print("Load Ai model & Warm up")
    import asyncio
    await asyncio.to_thread(AIManager.warmup_models)

    print("Loaded and Warmed up AI models successfully")

    yield  # จุดนี้คือช่วงที่เซิร์ฟเวอร์รันทำงานปกติ

    # --- Shutdown (ทำงานตอนที่เรากดปิดเซิร์ฟเวอร์) ---
    await close_supabase_client()
   


# Create FastAPI app instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure proper origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(missing_pets.router)
app.include_router(upload.router)
app.include_router(sightings.router)
app.include_router(hunters.router)
app.include_router(admin.router)
app.include_router(devices.router)
app.include_router(me.router)
app.include_router(reports.router)

# DEV-ONLY: mount auth convenience endpoints ONLY when explicitly enabled.
# When the flag is off these routes are never registered (404), not just
# disabled. Keep ENABLE_DEV_AUTH false/unset in any shared or public deploy.
if settings.ENABLE_DEV_AUTH:
    app.include_router(dev_auth.router)
    logger.warning(
        "DEV AUTH endpoints enabled (/dev/login, /dev/register) — "
        "do not use in production."
    )


@app.get("/", tags=["Root"])
def root():
    """Root endpoint providing API information."""
    return {
        "message": "Petty Bounty API is running!",
        "docs": "/docs",
        "redoc": "/redoc",
        "version": settings.VERSION
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}