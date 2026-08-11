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
from app.repositories.supabase_user_repository import SupabaseUserRepository
from app.repositories.user_repository import UserProfileNotFound
from app.schemas.common import StandardResponse

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.get("/me", response_model=StandardResponse)
async def get_me(
    supabase=Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    """Return the authenticated user's public.users profile row."""
    try:
        data = SupabaseUserRepository(supabase).get_user_profile(user_id)
    except UserProfileNotFound as e:
        raise HTTPException(
            status_code=404,
            detail=f"Profile not found for the authenticated user: {e}",
        )

    return StandardResponse(
        status="success",
        message="Profile retrieved successfully.",
        data=data,
    )
