"""Supabase adapter for DeviceTokenRepository (POST /devices/*)."""


class SupabaseDeviceTokenRepository:
    def __init__(self, db):
        self._db = db

    def upsert_device_token(self, row: dict) -> dict | None:
        res = (
            self._db.table("device_tokens")
            .upsert(row, on_conflict="fcm_token")
            .execute()
        )
        return res.data[0] if res.data else None

    def delete_device_token(self, user_id: str, fcm_token: str) -> None:
        # Scoped by BOTH user_id and fcm_token so a client can only drop its
        # own token. Deleting an already-absent token is a benign no-op.
        (
            self._db.table("device_tokens")
            .delete()
            .eq("user_id", user_id)
            .eq("fcm_token", fcm_token)
            .execute()
        )
