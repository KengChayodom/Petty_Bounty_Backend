"""
Auth-adjacent routes — Feature #6.

Registration / login / logout happen client-side directly against Supabase
Auth (the one Supabase service Flutter is allowed to call directly). The
backend's job here is just to expose the caller's own profile so the client
can confirm the session is valid and read the server-authoritative `role`.
"""
from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client
from app.schemas.common import StandardResponse

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.get("/me", response_model=StandardResponse)
async def get_me(
    supabase=Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    """Return the authenticated user's public.users profile row."""
    try:
        result = (
            supabase.table("users")
            .select("id, display_name, phone, role, total_score, profile_image_url, created_at")
            .eq("id", user_id)
            .single()
            .execute()
        )
    except Exception as e:
        # `.single()` raises when no row exists (e.g. profile trigger missed).
        raise HTTPException(
            status_code=404,
            detail=f"Profile not found for the authenticated user: {e}",
        )

    return StandardResponse(
        status="success",
        message="Profile retrieved successfully.",
        data=result.data,
    )
