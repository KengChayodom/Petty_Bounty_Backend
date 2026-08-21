"""Supabase adapter for ReportRepository — the `public.reports` moderation queue."""
from app.repositories.pagination import Page
from app.repositories.report_repository import FlagNotSaved


class SupabaseReportRepository:
    def __init__(self, db):
        self._db = db

    def create_flag(self, payload: dict) -> dict:
        res = self._db.table("reports").insert(payload).execute()
        if not res.data:
            raise FlagNotSaved(payload)
        return res.data[0]

    def get_report(self, report_id: str) -> dict | None:
        # maybe_single(): a missing flag yields data=None rather than raising,
        # so "not found" stays a domain condition (404) and only a genuine
        # transport failure surfaces as 500.
        res = (self._db.table("reports")
                       .select("*")
                       .eq("id", report_id)
                       .maybe_single()
                       .execute())
        return getattr(res, "data", None) or None

    def list_reports(
        self, status: str | None, limit: int, offset: int
    ) -> Page:
        # MD-51 moderation queue, mirroring the missing-pet browse of MD-37:
        # the status predicate is applied ONLY when one was supplied, so `None`
        # means every status rather than "status IS NULL". Newest first,
        # because a queue is worked from the most recent complaint backwards.
        #
        # count="exact" carries the queue DEPTH back with the page. A moderator
        # needs to know how much is waiting, and the console cannot infer it
        # from a page that happens to be full.
        query = self._db.table("reports").select("*", count="exact")
        if status is not None:
            query = query.eq("status", status)
        res = (query.order("created_at", desc=True)
                    .range(offset, offset + limit - 1)
                    .execute())
        rows = res.data or []
        total = getattr(res, "count", None)
        return Page(rows, len(rows) if total is None else total)

    def update_report(self, report_id: str, patch: dict) -> dict | None:
        res = (self._db.table("reports")
                       .update(patch)
                       .eq("id", report_id)
                       .execute())
        return res.data[0] if res.data else None
