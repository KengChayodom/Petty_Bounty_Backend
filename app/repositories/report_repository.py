"""The moderation-flag aggregate's repository port (`public.reports`).

A **flag** is one user's complaint that a sighting violates the platform
guidelines (MD-43 terminology note). It is a row of `reports`, and it is never a
missing-pet report — the word "report" means a missing pet everywhere else in
this codebase.

Column-name note: the method spec calls the fields `report_reason` and
`report_status`, but the deployed table (sql-update.txt:88, verified live
2026-08-16) names them `reason` and `status`. The DB names win inside the
adapter; the service works in domain terms.

`ReportService.flag_sighting` (MD-43) and `AdminService.review_report` (MD-44)
depend on this port; tests double it with MagicMock(spec=ReportRepository).
"""
from typing import Protocol

from app.repositories.pagination import Page


class ReportNotFound(ValueError):
    """No `reports` row with that id — MD-44's 404."""

    def __init__(self, report_id: str):
        super().__init__(f"Report {report_id} not found")
        self.report_id = report_id


class ReportAlreadyModerated(ValueError):
    """The flag has already been dismissed or upheld — MD-44's 409.

    UD-16 [E1] "Action Conflict": another admin got there first, so the second
    decision must not silently overwrite the first.
    """

    def __init__(self, report_id: str, current_status: str):
        super().__init__(
            f"Report {report_id} has already been moderated "
            f"(status={current_status})"
        )
        self.report_id = report_id
        self.current_status = current_status


class FlagTargetNotFound(ValueError):
    """The sighting being flagged does not exist — MD-43's 404."""

    def __init__(self, sighting_id: str):
        super().__init__(f"Sighting {sighting_id} not found")
        self.sighting_id = sighting_id


class FlagNotSaved(ValueError):
    """Raised by the adapter when the reports INSERT returns no row.

    Subclasses ValueError to match the MissingPetNotSaved / SightingNotSaved
    convention already used by the other ports.
    """

    def __init__(self, payload: dict):
        super().__init__("Database insertion failed - no data returned")
        self.payload = payload


class ReportRepository(Protocol):
    def create_flag(self, payload: dict) -> dict: ...
    def get_report(self, report_id: str) -> dict | None: ...
    def update_report(self, report_id: str, patch: dict) -> dict | None: ...
    def list_reports(
        self, status: str | None, limit: int, offset: int
    ) -> Page: ...
