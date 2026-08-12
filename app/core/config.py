"""
Core configuration settings for the Petty Bounty API.

Loads environment variables and provides application-wide constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # Supabase Configuration
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY", "")
    # Anon (publishable) key — same one the Flutter app uses. Only needed by the
    # DEV-ONLY auth helpers (app/api/dev_auth.py) to proxy Supabase GoTrue.
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")

    # Firebase Admin SDK — path to the service-account JSON used for FCM push
    # (SRS-FR-12). Same load pattern as the Supabase keys, but intentionally
    # NON-FATAL: if unset/invalid the app still boots and only FCM is disabled
    # (see app/core/firebase.py). Never commit the JSON — load via env.
    FIREBASE_CREDENTIALS: str = os.getenv("FIREBASE_CREDENTIALS", "")

    # Matching Defaults
    DEFAULT_SEARCH_RADIUS_KM: float = 10.0
    DEFAULT_MATCH_THRESHOLD: float = 0.7
    DEFAULT_MATCH_LIMIT: int = 5

    # Colour-aware matching (see app/services/sighting_logic.rerank_by_color).
    # The final match score blends CLIP cosine with coat-colour similarity:
    #   combined = CLIP_MATCH_WEIGHT * clip_sim + COLOR_MATCH_WEIGHT * color_sim
    # The two weights should sum to 1.0. CLIP stays dominant — colour is a
    # tie-breaker/penalty, not the primary signal.
    CLIP_MATCH_WEIGHT: float = 0.7
    COLOR_MATCH_WEIGHT: float = 0.3

    # Hard exclusion for clearly-wrong colours. Colour distance is measured in
    # CIELab with lightness down-weighted (COLOR_LIGHTNESS_WEIGHT) so a bright
    # owner-picked swatch and the same coat photographed under shade still count
    # as the same colour "family" — what separates grey from orange is chroma
    # (a*/b*), not brightness. A candidate whose colour distance exceeds
    # COLOR_EXCLUDE_DISTANCE is dropped from the results entirely. Tuned lenient
    # on purpose: only obviously-different colours (grey vs orange) get cut;
    # shade/lighting variants of the same colour survive. Colour is only ever
    # applied when BOTH sides have a colour — a missing colour never excludes.
    COLOR_EXCLUDE_DISTANCE: float = 45.0
    COLOR_LIGHTNESS_WEIGHT: float = 0.25

    # Candidate pool the match RPC is asked for before the colour re-rank trims
    # to the client's limit. Must exceed DEFAULT_MATCH_LIMIT so the colour pass
    # can reorder/exclude before the top-N is cut (otherwise SQL's CLIP-only
    # LIMIT would hide colour-better candidates ranked just outside it).
    MATCH_CANDIDATE_POOL: int = 50

    # API Configuration
    API_PREFIX: str = "/api"
    PROJECT_NAME: str = "Petty Bounty API"
    VERSION: str = "1.0.0"

    # Admin bypass — must be explicitly opted into per-environment. When False
    # (the default), `require_admin` short-circuits to 503 instead of returning
    # TEST_USER_ID, so a deploy without real auth (Feature #6) cannot expose
    # the bounty-payout endpoints.
    ENABLE_UNAUTHED_ADMIN: bool = os.getenv(
        "ENABLE_UNAUTHED_ADMIN", "false"
    ).lower() in ("1", "true", "yes")

    # Auth dev escape hatch — when True, requests with a missing or invalid
    # bearer token fall back to the hardcoded TEST_USER_ID instead of being
    # rejected with 401. Keeps the existing CLI smoke scripts and pytest suite
    # working without a real Supabase session. MUST stay False (the default)
    # in any shared or public deploy.
    AUTH_DEV_BYPASS: bool = os.getenv(
        "AUTH_DEV_BYPASS", "false"
    ).lower() in ("1", "true", "yes")

    # DEV-ONLY: when True, main.py mounts app/api/dev_auth.py (/dev/login,
    # /dev/register) — thin proxies to Supabase GoTrue so a token can be minted
    # from /docs without curl. When False (the default), those routes are not
    # registered at all (404). MUST stay False/unset in any shared/public deploy.
    ENABLE_DEV_AUTH: bool = os.getenv(
        "ENABLE_DEV_AUTH", "false"
    ).lower() in ("1", "true", "yes")

    def __init__(self):
        """Validate required settings on initialization."""
        if not self.SUPABASE_URL:
            raise ValueError("SUPABASE_URL environment variable must be set")
        if not self.SUPABASE_SERVICE_KEY:
            raise ValueError("SUPABASE_SERVICE_KEY environment variable must be set")


# Create a singleton instance
settings = Settings()
