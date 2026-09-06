"""Report service — a user flagging a sighting for moderator review (MD-43).

"Flag" is deliberate terminology (see report_repository): a row of `reports` is
one user's complaint about a *sighting*, never a missing-pet report.

All DB access goes through the ReportRepository / SightingRepository ports; this
service holds zero supabase-py calls. Payload construction and reason
normalisation are pure functions in moderation_logic.
"""
import asyncio
import logging

from app.repositories.report_repository import (
    FlagTargetNotFound,
    ReportRepository,
)
from app.repositories.sighting_repository import SightingRepository
from app.services.moderation_logic import build_flag_payload

logger = logging.getLogger(__name__)


class ReportService:
    """Write side of the moderation queue (UD-15)."""

    def __init__(
        self, repo: ReportRepository, sighting_repo: SightingRepository
    ):
        self.repo = repo
        self.sighting_repo = sighting_repo

    async def flag_sighting(
        self, sighting_id: str, reason: str, reporter_id: str
    ) -> dict:
        """
        MD-43 — record a flag against a sighting.

        The reason is validated BEFORE anything is read or written, so a bad
        reason costs no round-trip and can never reach the enum cast. The target
        sighting's existence is then checked, because `reports.sighting_id` is
        nullable in the schema: without this check a flag against a ghost id
        would insert happily and sit in the queue pointing at nothing.

        Raises:
            ValueError: reason outside the permitted set (API -> 400).
            FlagTargetNotFound: no such sighting (API -> 404).
            Exception: any repository failure (API -> 500).
        """
        # Raises ValueError (400) before any I/O.
        payload = build_flag_payload(sighting_id, reason, reporter_id)
        # Both round-trips run as one unit off the event loop — supabase-py is
        # blocking. Same rationale as SightingService's module docstring.
        return await asyncio.to_thread(
            self._flag_sighting_sync, sighting_id, reporter_id, payload
        )

    def _flag_sighting_sync(
        self, sighting_id: str, reporter_id: str, payload: dict,
    ) -> dict:
        target = self.sighting_repo.get_sighting(sighting_id)
        if not target:
            raise FlagTargetNotFound(sighting_id)

        created = self.repo.create_flag(payload)
        logger.warning(
            "User %s flagged sighting %s as %s",
            reporter_id, sighting_id, payload["reason"],
        )
        return created
