"""Supabase adapter for NotificationRepository (FCM fan-out)."""


class SupabaseNotificationRepository:
    def __init__(self, db):
        self._db = db

    def get_nearby_hunters(
        self,
        center_wkt: str,
        radius_meters: float,
        max_age_hours: int,
        exclude_user_id: str,
    ) -> list[str]:
        res = self._db.rpc(
            "get_nearby_hunters",
            {
                "center_location": center_wkt,
                "radius_meters": radius_meters,
                "max_age_hours": max_age_hours,
                "exclude_user_id": exclude_user_id,
            },
        ).execute()
        return [r["user_id"] for r in (res.data or []) if r.get("user_id")]

    def get_fcm_tokens_for_users(self, user_ids: list[str]) -> list[str]:
        res = (
            self._db.table("device_tokens")
            .select("fcm_token")
            .in_("user_id", user_ids)
            .execute()
        )
        return [r["fcm_token"] for r in (res.data or []) if r.get("fcm_token")]

    def delete_device_tokens(self, tokens: list[str]) -> None:
        self._db.table("device_tokens").delete().in_("fcm_token", tokens).execute()
