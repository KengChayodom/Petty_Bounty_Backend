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

    # Matching Defaults
    DEFAULT_SEARCH_RADIUS_KM: float = 10.0
    DEFAULT_MATCH_THRESHOLD: float = 0.7
    DEFAULT_MATCH_LIMIT: int = 5

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

    def __init__(self):
        """Validate required settings on initialization."""
        if not self.SUPABASE_URL:
            raise ValueError("SUPABASE_URL environment variable must be set")
        if not self.SUPABASE_SERVICE_KEY:
            raise ValueError("SUPABASE_SERVICE_KEY environment variable must be set")


# Create a singleton instance
settings = Settings()
