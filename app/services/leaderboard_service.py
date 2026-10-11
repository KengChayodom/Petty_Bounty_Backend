"""Leaderboard service: assembles the two Rank List boards.

The queries are behind a `LeaderboardRepository`, the arithmetic is the pure
`assign_ranks`. Each public method does its blocking reads in one worker thread,
as the other services do (see the threading note in sighting_service.py).
"""
import asyncio

from app.repositories.leaderboard_repository import LeaderboardRepository
from app.services.leaderboard_logic import assign_ranks


class LeaderboardService:
    def __init__(self, repo: LeaderboardRepository):
        self.repo = repo

    async def users_page(self, viewer_id: str, limit: int, offset: int) -> dict:
        """One page of the hunter board plus the caller's own standing.

        `me` carries the caller's GLOBAL rank, "how many users strictly outscore
        me, plus one", so it reads the same whichever page is on show.
        """
        return await asyncio.to_thread(
            self._users_page_sync, viewer_id, limit, offset
        )

    def _users_page_sync(self, viewer_id: str, limit: int, offset: int) -> dict:
        rows = self.repo.list_users_page(limit, offset)
        scores = [r.get("total_score") or 0 for r in rows]
        first_rank = (self.repo.count_users_above(scores[0]) + 1) if rows else 1
        ranks = assign_ranks(scores, offset=offset, first_rank=first_rank)

        entries = [
            {
                "rank": rank,
                "user_id": r["id"],
                "username": r.get("username"),
                "profile_image_url": r.get("profile_image_url"),
                "total_score": score,
            }
            for r, score, rank in zip(rows, scores, ranks)
        ]

        me_row = self.repo.get_user_standing(viewer_id) or {}
        my_score = me_row.get("total_score") or 0
        me = {
            "rank": self.repo.count_users_above(my_score) + 1,
            "user_id": viewer_id,
            "username": me_row.get("username"),
            "profile_image_url": me_row.get("profile_image_url"),
            "total_score": my_score,
        }
        return {"entries": entries, "me": me}

    async def bounties_page(self, limit: int, offset: int) -> dict:
        """One page of open missing pets, highest bounty first."""
        return await asyncio.to_thread(self._bounties_page_sync, limit, offset)

    def _bounties_page_sync(self, limit: int, offset: int) -> dict:
        rows = self.repo.list_open_bounties_page(limit, offset)
        amounts = [float(r.get("bounty_amount") or 0) for r in rows]
        first_rank = (
            self.repo.count_open_bounties_above(amounts[0]) + 1 if rows else 1
        )
        ranks = assign_ranks(amounts, offset=offset, first_rank=first_rank)
        entries = [
            {
                "rank": rank,
                "pet_id": r["id"],
                "pet_name": r.get("pet_name"),
                "image_url": r.get("image_url"),
                "bounty_amount": amount,
            }
            for r, amount, rank in zip(rows, amounts, ranks)
        ]
        return {"entries": entries}
