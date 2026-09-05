"""Moderation flags — a user reporting a sighting for review (MD-43, SRS-71).

"Flag" is the term used throughout for a `reports` row, so it is never confused
with a missing-pet report (see app/repositories/report_repository.py).
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client
from app.repositories.report_repository import FlagTargetNotFound
from app.repositories.supabase_report_repository import SupabaseReportRepository
from app.repositories.supabase_sighting_repository import (
    SupabaseSightingRepository,
)
from app.schemas.common import StandardResponse
from app.services.moderation_logic import FLAG_REASONS
from app.services.report_service import ReportService

router = APIRouter(prefix="/reports", tags=["Reports"])


def get_report_service(
    supabase=Depends(get_supabase_client),
) -> ReportService:
    return ReportService(
        repo=SupabaseReportRepository(supabase),
        sighting_repo=SupabaseSightingRepository(supabase),
    )


class FlagSightingRequest(BaseModel):
    """Body of POST /reports.

    `reporter_id` is deliberately absent — the reporter is always the verified
    JWT identity, never a value the client can choose.
    """

    sighting_id: str = Field(
        ..., description="UUID of the sighting being flagged.",
    )
    reason: str = Field(
        ...,
        description=(
            "Why the sighting violates the guidelines. One of "
            f"{', '.join(FLAG_REASONS)} (the spaced spellings — "
            "'Not a pet', 'Inappropriate image' — are accepted too)."
        ),
    )


@router.post("", response_model=StandardResponse)
async def flag_sighting(
    payload: FlagSightingRequest,
    service: ReportService = Depends(get_report_service),
    reporter_id: str = Depends(get_current_user_id),
):
    """Record a flag against a sighting for moderator review."""
    try:
        flag = await service.flag_sighting(
            payload.sighting_id, payload.reason, reporter_id,
        )
    except FlagTargetNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to flag sighting: {e}"
        )

    return StandardResponse(
        status="success",
        message="Sighting flagged for review.",
        data=flag,
    )
