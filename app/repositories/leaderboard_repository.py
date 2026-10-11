"""The leaderboard's repository port.

Four reads over `users` and `missing_pets`, no schema of their own. One method
per real query, returning plain dicts and ints, never a vendor response.
"""
from typing import Protocol


class LeaderboardRepository(Protocol):
    def list_users_page(self, limit: int, offset: int) -> list[dict]:
        """Hunters, highest `total_score` first, `id` as the tiebreak so rows
        with equal scores keep one stable order across pages."""
        ...

    def count_users_above(self, score: float) -> int:
        """How many users have a total_score strictly greater than `score`."""
        ...

    def get_user_standing(self, user_id: str) -> dict | None:
        """`{username, profile_image_url, total_score}` for one user, or None."""
        ...

    def list_open_bounties_page(self, limit: int, offset: int) -> list[dict]:
        """Missing pets whose search is still open, highest `bounty_amount`
        first, `id` as the tiebreak."""
        ...

    def count_open_bounties_above(self, amount: float) -> int:
        """How many open searches carry a bounty strictly greater than `amount`."""
        ...
