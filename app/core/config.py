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

    def __init__(self):
        """Validate required settings on initialization."""
        if not self.SUPABASE_URL:
            raise ValueError("SUPABASE_URL environment variable must be set")
        if not self.SUPABASE_SERVICE_KEY:
            raise ValueError("SUPABASE_SERVICE_KEY environment variable must be set")


# Create a singleton instance
settings = Settings()
