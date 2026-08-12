"""
Admin service — sighting verification, sighting-timeline, and the atomic
pet-resolution call.

The hard part (transferring the bounty + distributing F1 clue scores +
flipping pet status, all atomically) is delegated to the
`resolve_missing_pet` PostgreSQL function. This module is just the thin
Python wrapper.

All DB access goes through an AdminRepository port (app/repositories/); this
service holds zero supabase-py calls.
"""
import logging
from typing import Optional

from app.repositories.admin_repository import AdminRepository
from app.services.sighting_logic import strip_feature_vector

logger = logging.getLogger(__name__)


class AdminService:
    """Verification + resolution operations available to admins only."""

    def __init__(self, repo: AdminRepository):
        self.repo = repo

    async def verify_sighting(
        self, sighting_id: str, verification_status: str,
    ) -> dict:
        if verification_status not in ("Verified", "Dismissed"):
            raise ValueError(
                "verification_status must be 'Verified' or 'Dismissed'"
            )
        try:
            row = self.repo.update_sighting_verification(
                sighting_id, verification_status
            )
            if not row:
                raise ValueError(f"Sighting {sighting_id} not found")
            row = strip_feature_vector(row)
            logger.warning(
                "Admin set sighting %s verification_status=%s",
                sighting_id, verification_status,
            )
            return row
        except ValueError:
            raise
        except Exception as e:
            logger.error("Error verifying sighting %s: %s", sighting_id, e)
            raise

    async def get_sighting_timeline(
        self, pet_id: str, limit: int = 100, offset: int = 0,
    ) -> list[dict]:
        """
        Full audit timeline for admin verification — same shape as the
        owner endpoint but unfiltered by verification_status (admins need
        to see Dismissed entries too).
        """
        try:
            return self.repo.get_sighting_timeline(pet_id, limit, offset)
        except Exception as e:
            logger.error("Error fetching timeline for pet %s: %s", pet_id, e)
            raise

    async def resolve_missing_pet(
        self,
        pet_id: str,
        final_sighting_id: str,
        slip_image_url: str,
        reference_no: Optional[str],
        verified_by: str,
    ) -> dict:
        """
        Single-RPC resolution. The DB function raises on duplicate resolve
        and on invalid final_sighting; both surface as PostgrestError which
        we re-raise as ValueError so the API layer can return 400.
        """
        try:
            result = self.repo.resolve_missing_pet(
                pet_id, final_sighting_id, slip_image_url,
                reference_no, verified_by,
            )
            logger.warning(
                "Pet %s resolved by %s — final_sighting=%s",
                pet_id, verified_by, final_sighting_id,
            )
            return result
        except Exception as e:
            msg = str(e)
            # The DB function raises with these prefixes — translate to a
            # client-friendly 400 instead of a 500.
            if ("already resolved" in msg
                    or "not a verified Caught sighting" in msg
                    or "not found" in msg):
                raise ValueError(msg) from e
            logger.error("Error resolving pet %s: %s", pet_id, e)
            raise
