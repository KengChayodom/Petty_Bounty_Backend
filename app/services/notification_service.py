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
from app.repositories.supabase_sighting_repository import (
    SupabaseSightingRepository,
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
) -> int:
    """Send one notification to every device token of the given users.

    Prunes tokens that FCM reports as UNREGISTERED (uninstalled / rotated away).
    No-op when Firebase is not configured or there are no recipients/tokens.

    Returns how many messages FCM actually delivered — 0 for every no-op and
    every failure. Callers that record "we notified them" must gate on this
    rather than on the call returning, so a status never claims a push that
    was skipped (unconfigured Firebase) or dropped (no tokens, send error).
    """
    if not user_ids:
        return 0
    if not is_firebase_ready():
        logger.warning(
            "FCM not configured — skipping push to %d user(s).", len(user_ids)
        )
        return 0

    try:
        tokens = repo.get_fcm_tokens_for_users(user_ids)
    except Exception as e:
        logger.warning("Failed to load device tokens: %s", e)
        return 0

    if not tokens:
        logger.info("No device tokens for %d user(s) — nothing to send.", len(user_ids))
        return 0

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
        return 0

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
    return response.success_count


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


def notify_pet_owners(
    db,
    sighting_id: str,
    pet_ids: list[str],
    hunter_id: str | None = None,
) -> list[str]:
    """Tell the owners of `pet_ids` that someone has reported seeing their pet.

    This is the missing half of the loop: until now a sighting landed in the
    database and nobody told the owner, so `sighting_status` never left
    `Pending_Analysis` and the owner had to open the app and go looking. It
    serves both report paths — the AI-matched pets of a discovery sighting, and
    the single pet a targeted sighting names.

    On a successful push the sighting advances to `Notified_Owner`. The status
    is written **only** when FCM actually delivered something, so it never
    claims a notification that was skipped (Firebase unconfigured) or dropped
    (owner has no device token) — an owner reading "we told you" who was never
    told is worse than a status that stays put.

    Runs as a FastAPI BackgroundTask, so nothing here may raise into the
    request; every failure is logged and swallowed.

    Returns the owner ids that were pushed to (empty when nothing was sent).
    """
    if not pet_ids:
        return []
    if not is_firebase_ready():
        return []

    # `db` is the raw request-scoped client (this runs as a BackgroundTask
    # scheduled from the route), so the ports are constructed here.
    notif_repo = SupabaseNotificationRepository(db)
    sighting_repo = SupabaseSightingRepository(db)

    try:
        owners_by_pet = sighting_repo.get_pet_owners(pet_ids)
    except Exception as e:
        logger.warning(
            "Owner lookup failed for sighting %s: %s", sighting_id, e
        )
        return []

    # One push per owner even when several of their pets matched, and never a
    # push to the hunter about their own pet (an owner who spots their own pet
    # and reports it does not need telling).
    owner_ids = sorted({
        owner_id for owner_id in owners_by_pet.values()
        if owner_id and owner_id != hunter_id
    })
    if not owner_ids:
        logger.info("No owners to notify for sighting %s.", sighting_id)
        return []

    delivered = send_to_users(
        notif_repo,
        owner_ids,
        title="Someone spotted your pet 🐾",
        body="A hunter just reported a sighting that matches your missing pet.",
        data={"sightingId": sighting_id},
    )
    if not delivered:
        logger.info(
            "Sighting %s: no push delivered to %d owner(s) — status left at "
            "Pending_Analysis.", sighting_id, len(owner_ids),
        )
        return []

    try:
        sighting_repo.set_sighting_status(sighting_id, "Notified_Owner")
    except Exception as e:
        # The owners DID get the push; failing to record it is a bookkeeping
        # problem, not a reason to report the notification as failed.
        logger.error(
            "Sighting %s: owners notified but status write FAILED: %s",
            sighting_id, e,
        )
    return owner_ids
