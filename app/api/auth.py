"""
Auth-adjacent routes — Feature #6.

Registration / login / logout happen client-side directly against Supabase
Auth (the one Supabase service Flutter is allowed to call directly). The
backend's job here is just to expose the caller's own profile so the client
can confirm the session is valid and read the server-authoritative `role`.
"""
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_user_service
from app.core.auth import get_current_user_id
from app.repositories.user_repository import UserProfileNotFound
from app.schemas.common import StandardResponse
from app.services.user_service import UserService

# Every route here needs a signed-in caller, declared ONCE for the whole router so a route added later inherits it.
# The routes still ask for the id where they use it, which FastAPI resolves once per request.
router = APIRouter(
    prefix="/auth", tags=["Auth"],
    dependencies=[Depends(get_current_user_id)],
)


@router.get("/me", response_model=StandardResponse)
async def get_me(
    service: UserService = Depends(get_user_service),
    user_id: str = Depends(get_current_user_id),
):
    """Return the authenticated user's public.users profile row."""
    try:
        data = await service.get_profile(user_id)
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
