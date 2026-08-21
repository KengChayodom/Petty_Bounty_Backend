"""Leaderboards — read-only rankings for the mobile app's "Rank List" screen.

Two boards, both paginated:
  * /leaderboard/users    — hunters ranked by cumulative `total_score`.
  * /leaderboard/bounties — active missing pets ranked by `bounty_amount`.

No schema of its own: every field already exists (`users.total_score`,
`missing_pets.bounty_amount`). Straight ordered reads, so they live in the
route rather than a repository/service layer.
"""
from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client
from app.schemas.common import StandardResponse

router = APIRouter(prefix="/leaderboard", tags=["Leaderboard"])


@router.get("/users", response_model=StandardResponse)
async def leaderboard_users(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    supabase=Depends(get_supabase_client),
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
        res = (
            supabase.table("users")
            .select("id, display_name, profile_image_url, total_score")
            .order("total_score", desc=True)
            .order("id")
            .range(offset, offset + limit - 1)
            .execute()
        )
        entries = [
            {
                "rank": offset + i + 1,
                "user_id": r["id"],
                "display_name": r.get("display_name"),
                "profile_image_url": r.get("profile_image_url"),
                "total_score": r.get("total_score") or 0,
            }
            for i, r in enumerate(res.data or [])
        ]

        me_res = (
            supabase.table("users")
            .select("display_name, profile_image_url, total_score")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        me_row = (me_res.data or [{}])[0]
        my_score = me_row.get("total_score") or 0
        higher = (
            supabase.table("users")
            .select("id", count="exact")
            .gt("total_score", my_score)
            .execute()
        )
        me_standing = {
            "rank": (higher.count or 0) + 1,
            "user_id": user_id,
            "display_name": me_row.get("display_name"),
            "profile_image_url": me_row.get("profile_image_url"),
            "total_score": my_score,
        }

        return StandardResponse(
            status="success",
            message=f"Retrieved {len(entries)} ranked users.",
            data={"entries": entries, "me": me_standing},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load user leaderboard: {e}"
        )


@router.get("/bounties", response_model=StandardResponse)
async def leaderboard_bounties(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    supabase=Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    """One page of active missing pets ranked by bounty (highest first).

    "Active" = the search is still open. The owner flow only ever moves a pet's
    status column from 'Searching' to 'Found', so filtering out the closed
    states leaves exactly the pets still worth chasing; a 'Resolved' guard is
    kept in case an admin settlement uses that word.
    """
    try:
        res = (
            supabase.table("missing_pets")
            .select("id, pet_name, image_url, bounty_amount")
            .not_.in_("status", ["Found", "Resolved"])
            .order("bounty_amount", desc=True)
            .order("id")
            .range(offset, offset + limit - 1)
            .execute()
        )
        entries = [
            {
                "rank": offset + i + 1,
                "pet_id": r["id"],
                "pet_name": r.get("pet_name"),
                "image_url": r.get("image_url"),
                "bounty_amount": float(r.get("bounty_amount") or 0),
            }
            for i, r in enumerate(res.data or [])
        ]
        return StandardResponse(
            status="success",
            message=f"Retrieved {len(entries)} bounties.",
            data={"entries": entries},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load bounty leaderboard: {e}"
        )
