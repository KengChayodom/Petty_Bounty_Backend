"""
Authentication utilities - COMPLETE BYPASS MODE

This module provides a complete auth bypass for testing.
All JWT validation is DISABLED. Hardcoded user ID is always returned.

TODO: Re-enable proper JWT verification after testing is complete.
"""

# Hardcoded test user ID - ALWAYS returned, no validation performed
TEST_USER_ID = "024dd692-8b4a-44b7-968c-f6f3ddac3f4c"


def verify_token(authorization: str = "") -> str:
    """
    COMPLETE BYPASS: Ignores input, always returns hardcoded user ID.

    This function:
    - Does NOT validate the token
    - Does NOT check if authorization header exists
    - Does NOT raise ANY exceptions
    - ALWAYS returns the hardcoded test user ID

    Args:
        authorization: Authorization header (completely ignored)

    Returns:
        TEST_USER_ID (hardcoded)
    """
    return TEST_USER_ID


def verify_token_optional(authorization: str = "") -> str | None:
    """
    COMPLETE BYPASS: Always returns hardcoded user ID.

    Args:
        authorization: Authorization header (completely ignored)

    Returns:
        TEST_USER_ID (hardcoded)
    """
    return TEST_USER_ID
