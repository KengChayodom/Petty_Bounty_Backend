"""The device-token aggregate's repository port (FCM registration routes)."""
from typing import Protocol


class DeviceTokenRepository(Protocol):
    def upsert_device_token(self, row: dict) -> dict | None: ...
    def delete_device_token(self, user_id: str, fcm_token: str) -> None: ...
