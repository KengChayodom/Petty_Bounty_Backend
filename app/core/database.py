"""
Database connection management for Supabase.

This module provides a singleton Supabase client and dependency injection
for FastAPI endpoints.
"""
import os
from logging import getLogger
from typing import Annotated

from fastapi import Depends
from supabase import create_client, Client

from app.core.config import settings

logger = getLogger(__name__)

# Singleton Supabase client instance
_supabase_client: Client | None = None


def get_supabase_client() -> Client:
    """
    Get or create the singleton Supabase client.

    Uses the service role key for backend operations.

    Returns:
        Supabase client instance

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_SERVICE_KEY are not set
    """
    global _supabase_client

    if _supabase_client is None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables"
            )

        logger.info("Creating new Supabase client")
        _supabase_client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY
        )

    return _supabase_client


# Type alias for dependency injection
SupabaseDep = Annotated[Client, Depends(get_supabase_client)]


async def close_supabase_client() -> None:
    """
    Close the Supabase client connection.

    This should be called on application shutdown.
    """
    global _supabase_client
    if _supabase_client is not None:
        logger.info("Closing Supabase client")
        _supabase_client = None
