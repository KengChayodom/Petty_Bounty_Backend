"""
Admin service — sighting verification, sighting-timeline, and the atomic
pet-resolution call.

The hard part (transferring the bounty + distributing F1 clue scores +
flipping pet status, all atomically) is delegated to the
`resolve_missing_pet` PostgreSQL function. This module is just the thin
Python wrapper.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AdminService:
    """Verification + resolution operations available to admins only."""

    def __init__(self, db_client):
        self.db = db_client

    async def verify_sighting(
        self, sighting_id: str, verification_status: str,
    ) -> dict:
        if verification_status not in ("Verified", "Dismissed"):
            raise ValueError(
                "verification_status must be 'Verified' or 'Dismissed'"
            )
        try:
            res = (self.db.table("sightings")
                          .update({"verification_status": verification_status})
                          .eq("id", sighting_id)
                          .execute())
            if not res.data:
                raise ValueError(f"Sighting {sighting_id} not found")
            row = res.data[0]
            row.pop("feature_vector", None)
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
            response = self.db.rpc("sightings_for_pet", {
                "p_pet_id":            pet_id,
                "p_limit":             limit,
                "p_offset":            offset,
                "p_include_dismissed": True,  # admins must see Dismissed entries
            }).execute()
            return response.data or []
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
            res = self.db.rpc("resolve_missing_pet", {
                "p_pet_id":            pet_id,
                "p_final_sighting_id": final_sighting_id,
                "p_slip_image_url":    slip_image_url,
                "p_reference_no":      reference_no,
                "p_verified_by":       verified_by,
            }).execute()
            logger.warning(
                "Pet %s resolved by %s — final_sighting=%s",
                pet_id, verified_by, final_sighting_id,
            )
            return res.data
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
