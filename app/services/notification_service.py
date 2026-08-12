"""
FCM push fan-out (SRS-FR-12).

Everything here is guarded by `is_firebase_ready()` so it is a safe no-op when
FIREBASE_CREDENTIALS is unset (the rest of the API still runs). Functions are
synchronous (firebase-admin + supabase-py are sync); call them from a FastAPI
BackgroundTask so the request that triggered them is never blocked.
"""
import logging

from firebase_admin import messaging

from app.core.firebase import is_firebase_ready
from app.repositories.notification_repository import NotificationRepository
from app.repositories.supabase_notification_repository import (
    SupabaseNotificationRepository,
)

logger = logging.getLogger(__name__)

# Android notification channel id — must match the manifest meta-data
# `default_notification_channel_id` so background/terminated messages render
# on Android 8+.
_ANDROID_CHANNEL_ID = "pet_alerts"


def send_to_users(
    repo: NotificationRepository,
    user_ids: list[str],
    title: str,
    body: str,
    data: dict | None = None,
) -> None:
    """Send one notification to every device token of the given users.

    Prunes tokens that FCM reports as UNREGISTERED (uninstalled / rotated away).
    No-op when Firebase is not configured or there are no recipients/tokens.
    """
    if not user_ids:
        return
    if not is_firebase_ready():
        logger.warning(
            "FCM not configured — skipping push to %d user(s).", len(user_ids)
        )
        return

    try:
        tokens = repo.get_fcm_tokens_for_users(user_ids)
    except Exception as e:
        logger.warning("Failed to load device tokens: %s", e)
        return

    if not tokens:
        logger.info("No device tokens for %d user(s) — nothing to send.", len(user_ids))
        return

    # FCM data values must be strings.
    str_data = {k: str(v) for k, v in (data or {}).items()}

    message = messaging.MulticastMessage(
        tokens=tokens,
        notification=messaging.Notification(title=title, body=body),
        data=str_data,
        android=messaging.AndroidConfig(
            priority="high",
            notification=messaging.AndroidNotification(
                channel_id=_ANDROID_CHANNEL_ID
            ),
        ),
    )

    try:
        response = messaging.send_each_for_multicast(message)
    except Exception as e:
        logger.warning("FCM multicast send failed: %s", e)
        return

    # Prune tokens FCM says are dead so they don't accumulate.
    stale: list[str] = []
    for token, result in zip(tokens, response.responses):
        if result.success:
            continue
        if isinstance(result.exception, messaging.UnregisteredError):
            stale.append(token)
        else:
            logger.warning("FCM send error (kept token): %s", result.exception)

    if stale:
        try:
            repo.delete_device_tokens(stale)
            logger.info("Pruned %d stale FCM token(s).", len(stale))
        except Exception as e:
            logger.warning("Failed to prune stale tokens: %s", e)

    logger.info(
        "FCM push: %d delivered, %d failed.",
        response.success_count,
        response.failure_count,
    )


def notify_nearby_hunters(
    db,
    pet_id: str,
    latitude: float,
    longitude: float,
    owner_id: str,
    radius_km: float,
    pet_name: str,
    species: str,
    max_age_hours: int = 24,
) -> None:
    """Find fresh hunters within radius (excluding the owner) and push them.

    Designed to run in a FastAPI BackgroundTask after a missing-pet INSERT.
    """
    if not is_firebase_ready():
        return

    # `db` is the raw request-scoped client (this runs as a BackgroundTask
    # scheduled straight from the route); wrap it in the port here — the
    # composition point for the fan-out.
    repo = SupabaseNotificationRepository(db)

    # WKT 'POINT(lng lat)' — Postgres casts to geography(4326); mirrors the
    # convention used by get_nearby_missing_pets.
    center_wkt = f"POINT({longitude} {latitude})"
    try:
        user_ids = repo.get_nearby_hunters(
            center_wkt, radius_km * 1000.0, max_age_hours, owner_id
        )
    except Exception as e:
        logger.warning("get_nearby_hunters failed for pet %s: %s", pet_id, e)
        return

    if not user_ids:
        logger.info("No fresh nearby hunters for pet %s.", pet_id)
        return

    send_to_users(
        repo,
        user_ids,
        title="Missing pet nearby 🐾",
        body=f"{pet_name} ({species}) was just reported near you. Can you help?",
        data={"petId": pet_id},
    )
