"""
FCM device-token registration for push notifications.

Upsert is keyed by the UNIQUE fcm_token, so a token that moves to a different
account is reassigned instead of duplicated. DB access goes through the
DeviceTokenRepository port, run off the event loop because supabase-py is
blocking.
"""
import asyncio
from datetime import datetime, timezone

from app.repositories.device_token_repository import DeviceTokenRepository


class DeviceService:
    def __init__(self, repo: DeviceTokenRepository):
        self.repo = repo

    async def register(self, user_id: str, fcm_token: str, platform: str) -> dict:
        """MD-09 — upsert the caller's FCM token.

        Raises:
            RuntimeError: the upsert returned no row.
        """
        row = {
            "user_id": user_id,
            "fcm_token": fcm_token,
            "platform": platform,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        created = await asyncio.to_thread(self.repo.upsert_device_token, row)
        if not created:
            raise RuntimeError("Token upsert returned no row")
        return created

    async def unregister(self, user_id: str, fcm_token: str) -> None:
        """MD-10 — drop the caller's FCM token on logout.

        Scoped by BOTH user_id and fcm_token, so a client can only delete its
        own token. Deleting a token that is already gone is a benign no-op.
        """
        await asyncio.to_thread(self.repo.delete_device_token, user_id, fcm_token)
