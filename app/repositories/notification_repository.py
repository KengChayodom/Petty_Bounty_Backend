"""Repository port for the FCM fan-out (notification_service).

Covers the geo-hunter query and the device-token reads/prunes that the push
fan-out performs. Returns plain lists of ids/tokens — no vendor rows cross it.
"""
from typing import Protocol


class NotificationRepository(Protocol):
    def get_nearby_hunters(
        self,
        center_wkt: str,
        radius_meters: float,
        max_age_hours: int,
        exclude_user_id: str,
    ) -> list[str]: ...
    def get_fcm_tokens_for_users(self, user_ids: list[str]) -> list[str]: ...
    def delete_device_tokens(self, tokens: list[str]) -> None: ...
