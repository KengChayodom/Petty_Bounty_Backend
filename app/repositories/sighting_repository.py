"""The sighting aggregate's repository port.

`SightingRepository` is a Protocol owned by THIS codebase: the service depends
on it, tests double it. One method per real DB operation the service performs,
derived from the actual call sites in sighting_service.py — never speculative.

NON-NEGOTIABLE: no vendor type crosses this boundary. Methods return plain
dicts / lists / ints / None — never an APIResponse, never a `.data`/`.count`
the caller has to unwrap. Vendor errors are translated inside the adapter.

Methods are SYNC on purpose: supabase-py's `.execute()` is a blocking call, so
the honest signature is `def`, not `async def`. (Making them async would be a
behaviour change, deferred to a later phase.)
"""
from typing import Protocol


class SightingNotSaved(ValueError):
    """Raised by the adapter when the sightings INSERT returns no row.

    Subclasses ValueError during the migration so the service's existing
    `except ValueError: raise` contract — and the tests asserting ValueError —
    stay intact. Tighten to a standalone domain error in a later phase.
    """

    def __init__(self, payload: dict):
        super().__init__("Insert failed: No data returned from Supabase")
        self.payload = payload


class SightingRepository(Protocol):
    # --- writes ---------------------------------------------------------- #
    def insert_sighting(self, payload: dict) -> dict: ...
    def upsert_sighting_matches(self, rows: list[dict]) -> None: ...

    # --- discovery / match reads ---------------------------------------- #
    def get_sighting_for_match(self, sighting_id: str) -> dict | None: ...
    def match_missing_pets(self, sighting_id: str, limit: int) -> list[dict]: ...
    def get_sighting(self, sighting_id: str) -> dict | None: ...

    # --- hunter activity / stats reads ---------------------------------- #
    def count_sightings_for_hunter(self, hunter_id: str) -> int: ...
    def list_sightings_for_hunter(
        self, hunter_id: str, limit: int, offset: int
    ) -> list[dict]: ...
    def get_matches_for_sightings(self, sighting_ids: list[str]) -> list[dict]: ...
    def get_awards_for_hunter(self, hunter_id: str) -> list[dict]: ...
    def get_user(self, hunter_id: str) -> dict | None: ...
    def count_verified_sightings_for_hunter(self, hunter_id: str) -> int: ...
    def count_contributions_for_hunter(self, hunter_id: str) -> int: ...
