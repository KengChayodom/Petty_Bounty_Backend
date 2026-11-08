"""
Firebase Admin SDK initialization — FCM push (SRS-FR-12).

NON-FATAL by design: if ``FIREBASE_CREDENTIALS`` is unset or invalid we log a
warning and skip init, so auth / matching / the rest of the API still boot
(the decision agreed for Checkpoint A). Any push-send path MUST call
``is_firebase_ready()`` before touching the messaging API.

Creds path follows the same env pattern as ``SUPABASE_SERVICE_KEY`` in
``config.py``; the JSON file itself is git-ignored and loaded via env only.
"""
import logging
import os

import firebase_admin
from firebase_admin import credentials

from app.core.config import settings

logger = logging.getLogger(__name__)

_initialized = False


def init_firebase() -> bool:
    """Initialize the Firebase Admin app once. Returns True if FCM is ready.

    Safe to call multiple times (idempotent) and never raises — a missing or
    bad credentials file disables push without breaking startup.
    """
    global _initialized
    if _initialized or firebase_admin._apps:
        _initialized = True
        return True

    cred_path = settings.FIREBASE_CREDENTIALS
    if not cred_path:
        logger.warning(
            "FIREBASE_CREDENTIALS not set — skipping Firebase Admin init. "
            "FCM push (SRS-FR-12) is DISABLED; the rest of the API runs "
            "normally."
        )
        return False
    if not os.path.isfile(cred_path):
        logger.warning(
            "FIREBASE_CREDENTIALS=%r points to no file — skipping Firebase "
            "Admin init. FCM push DISABLED.",
            cred_path,
        )
        return False

    try:
        firebase_admin.initialize_app(credentials.Certificate(cred_path))
        _initialized = True
        logger.info("Firebase Admin SDK initialized — FCM push enabled.")
        return True
    except Exception as exc:  # malformed JSON, wrong project, etc.
        logger.warning(
            "Firebase Admin init failed (%s) — FCM push DISABLED.", exc
        )
        return False


def is_firebase_ready() -> bool:
    """True when the Admin SDK is initialized and the messaging API is usable."""
    return _initialized or bool(firebase_admin._apps)
