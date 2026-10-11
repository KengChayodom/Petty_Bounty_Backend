"""Leaderboards — read-only rankings for the mobile app's "Rank List" screen.

Two boards, both paginated:
  * /leaderboard/users    — hunters ranked by cumulative `total_score`.
  * /leaderboard/bounties — active missing pets ranked by `bounty_amount`.

No schema of its own: every field already exists (`users.total_score`,
`missing_pets.bounty_amount`). The queries are in `SupabaseLeaderboardRepository`
and the ranking rules in `LeaderboardService`, like every other route.
"""
from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_leaderboard_service
from app.core.auth import get_current_user_id
from app.schemas.common import StandardResponse
from app.services.leaderboard_service import LeaderboardService

# Every route here needs a signed-in caller, declared ONCE for the whole router so a route added later inherits it.
# The routes still ask for the id where they use it, which FastAPI resolves once per request.
router = APIRouter(
    prefix="/leaderboard", tags=["Leaderboard"],
    dependencies=[Depends(get_current_user_id)],
)


@router.get("/users", response_model=StandardResponse)
async def leaderboard_users(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: LeaderboardService = Depends(get_leaderboard_service),
    user_id: str = Depends(get_current_user_id),
):
    """One page of the hunter board (highest score first) plus the caller's own
    standing.

    `me` carries the caller's GLOBAL rank, computed as "how many users strictly
    outscore me, + 1" — not their position within the returned page — so the
    screen can pin "YOUR STANDING" no matter which page is on show. A tie shares
    a rank; the id tiebreak on the page keeps pagination from overlapping or
    skipping rows when scores are equal.
    """
    try:
        data = await service.users_page(user_id, limit, offset)
        return StandardResponse(
            status="success",
            message=f"Retrieved {len(data['entries'])} ranked users.",
            data=data,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load user leaderboard: {e}"
        )


@router.get("/bounties", response_model=StandardResponse)
async def leaderboard_bounties(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: LeaderboardService = Depends(get_leaderboard_service),
    user_id: str = Depends(get_current_user_id),
):
    """One page of active missing pets ranked by bounty (highest first).

    "Active" = the search is still open. The owner flow only ever moves a pet's
    status column from 'Searching' to 'Found', so filtering out the closed
    states leaves exactly the pets still worth chasing; a 'Resolved' guard is
    kept in case an admin settlement uses that word.
    """
    try:
        data = await service.bounties_page(limit, offset)
        return StandardResponse(
            status="success",
            message=f"Retrieved {len(data['entries'])} bounties.",
            data=data,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load bounty leaderboard: {e}"
        )
