"""
Admin service — sighting verification, sighting-timeline, the atomic
pet-resolution call, and review of the moderation flag queue.

The hard part of resolution (transferring the bounty + distributing F1 clue
scores + flipping pet status, all atomically) is delegated to the
`resolve_missing_pet` PostgreSQL function. This module is just the thin
Python wrapper.

All DB access goes through repository ports (app/repositories/); this service
holds zero supabase-py calls. `AdminRepository` covers verification / timeline /
resolution; flag review additionally needs the queue itself (`ReportRepository`),
which is an optional constructor argument so the rest of this service keeps its
one-argument construction.

**Scope of admin power (decided 2026-08-17).** An administrator moderates by
REMOVING things — dismissing a flagged sighting, deleting a report that breaks
the guidelines. There is deliberately no ban, no account suspension, and no
account dashboard: an earlier design had all three, and they were withdrawn
because moderation that touches accounts needs a review process the project does
not have. Admins are also no longer expected to verify every sighting; they look
at what users flag.
"""
import asyncio
import logging
from typing import Optional

from app.repositories.admin_repository import AdminRepository
from app.repositories.report_repository import (
    ReportAlreadyModerated,
    ReportNotFound,
    ReportRepository,
)
from app.services.moderation_logic import (
    DECISION_UPHOLD,
    normalize_flag_decision,
    normalize_flag_status_filter,
)
from app.services.sighting_logic import strip_feature_vector

logger = logging.getLogger(__name__)


class AdminService:
    """Verification, resolution, and flag moderation — admins only."""

    def __init__(
        self,
        repo: AdminRepository,
        report_repo: Optional[ReportRepository] = None,
    ):
        self.repo = repo
        self.report_repo = report_repo

    async def verify_sighting(
        self, sighting_id: str, verification_status: str,
    ) -> dict:
        if verification_status not in ("Verified", "Dismissed"):
            raise ValueError(
                "verification_status must be 'Verified' or 'Dismissed'"
            )
        # supabase-py is blocking; off the event loop it goes. Same rationale
        # as SightingService — see its module docstring.
        return await asyncio.to_thread(
            self._verify_sighting_sync, sighting_id, verification_status
        )

    def _verify_sighting_sync(
        self, sighting_id: str, verification_status: str,
    ) -> dict:
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
        return await asyncio.to_thread(
            self._get_sighting_timeline_sync, pet_id, limit, offset
        )

    def _get_sighting_timeline_sync(
        self, pet_id: str, limit: int, offset: int,
    ) -> list[dict]:
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
        return await asyncio.to_thread(
            self._resolve_missing_pet_sync, pet_id, final_sighting_id,
            slip_image_url, reference_no, verified_by,
        )

    def _resolve_missing_pet_sync(
        self,
        pet_id: str,
        final_sighting_id: str,
        slip_image_url: str,
        reference_no: Optional[str],
        verified_by: str,
    ) -> dict:
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

    # ---------------------------------------------------------------- #
    # MD-51 — read the moderation flag queue (the listing MD-40 acts from)
    # ---------------------------------------------------------------- #
    async def list_reports(
        self, status: str | None = None, limit: int = 20, offset: int = 0,
    ) -> list[dict]:
        """
        Paginated read of the `reports` moderation queue, newest first.

        Exists because `review_report` (MD-40) takes a `report_id` an
        administrator previously had no way to obtain: the flag queue could be
        written to and decided on, but never enumerated.

        The filter is normalised **before** any I/O, so an unrecognised status
        is a 400 at the edge rather than a failed enum cast surfacing as a 500.

        Raises:
            ValueError: status outside the `report_status` enum (API -> 400).
        """
        normalized = normalize_flag_status_filter(status)
        return await asyncio.to_thread(
            self._list_reports_sync, normalized, limit, offset
        )

    def _list_reports_sync(
        self, status: str | None, limit: int, offset: int,
    ) -> list[dict]:
        try:
            return self.report_repo.list_reports(status, limit, offset)
        except Exception as e:
            logger.error(
                "Error listing moderation flags (status=%s): %s", status, e
            )
            raise

    # ---------------------------------------------------------------- #
    # MD-40 — review a moderation flag (SRS-68, UD-14)
    # ---------------------------------------------------------------- #
    async def review_report(
        self, report_id: str, decision: str, admin_id: str,
    ) -> dict:
        """
        Resolve a pending flag: dismiss it, or uphold it and drop the sighting.

        Upholding is two writes: the flagged sighting is set Dismissed so it
        stops reaching owners, and the flag itself is closed as Reviewed_Ban.
        Despite that enum value's name, **nobody is banned** — the hunter keeps
        their account (see the module docstring); only the offending sighting is
        withdrawn.

        The flag's own status is written **last** on purpose. It is the
        idempotency key — while it is still Pending the whole decision can be
        retried, so a failure part-way through leaves work to redo rather than a
        flag marked handled whose consequences never landed.

        Raises:
            ValueError: decision outside {Dismissed, Reviewed_Ban} (API -> 400).
            ReportNotFound: no such flag (API -> 404).
            ReportAlreadyModerated: another admin already decided (API -> 409).
        """
        # Raises ValueError (400) before any I/O.
        status = normalize_flag_decision(decision)
        return await asyncio.to_thread(
            self._review_report_sync, report_id, status, admin_id
        )

    def _review_report_sync(
        self, report_id: str, status: str, admin_id: str,
    ) -> dict:
        """DB half of `review_report`; `status` is already normalised.

        Runs as one unit in a single worker thread, which also keeps the
        read-then-write idempotency check described above from being
        interleaved with another admin's decision by the event loop.
        """
        flag = self.report_repo.get_report(report_id)
        if not flag:
            raise ReportNotFound(report_id)

        current = flag.get("status") or "Pending"
        if current != "Pending":
            # UD-14 [E1] Action Conflict — do not overwrite the first decision.
            raise ReportAlreadyModerated(report_id, current)

        sighting_dismissed = False

        if status == DECISION_UPHOLD:
            sighting_id = flag.get("sighting_id")
            if sighting_id:
                sighting = self.repo.update_sighting_verification(
                    sighting_id, "Dismissed"
                )
                sighting_dismissed = bool(sighting)
            else:
                logger.warning(
                    "Flag %s upheld but carries no sighting_id — nothing to "
                    "dismiss", report_id,
                )

        updated = self.report_repo.update_report(report_id, {"status": status})
        if not updated:
            raise ReportNotFound(report_id)

        logger.warning(
            "Admin %s reviewed flag %s -> %s (sighting_dismissed=%s)",
            admin_id, report_id, status, sighting_dismissed,
        )
        return {
            "report": updated,
            "sighting_dismissed": sighting_dismissed,
        }
