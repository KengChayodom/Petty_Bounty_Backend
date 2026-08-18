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


class SightingActionLocked(ValueError):
    """The hunter tried to change `action_type` on a sighting that has already
    been adjudicated — the API's 409.

    Once an owner or admin has set `verification_status` away from 'Pending',
    the report has been judged as it stands. Letting the hunter flip 'Spotted'
    to 'Caught' afterwards would retro-fit a VERIFIED sighting into the one
    shape the resolve RPC pays a bounty for (Caught + Verified), so the column
    is frozen at that point.

    Subclasses ValueError like the moderation errors do, so route handlers MUST
    catch it before their generic `except ValueError` 400 (see
    tests/test_sighting_action_api.py, which pins that ordering).
    """

    def __init__(self, sighting_id: str, verification_status: str):
        super().__init__(
            f"Sighting {sighting_id} has already been reviewed "
            f"(verification_status={verification_status}); its action type "
            f"can no longer be changed"
        )
        self.sighting_id = sighting_id
        self.verification_status = verification_status


class SightingRepository(Protocol):
    # --- writes ---------------------------------------------------------- #
    def insert_sighting(self, payload: dict) -> dict: ...
    def upsert_sighting_matches(self, rows: list[dict]) -> None: ...
    def set_sighting_status(
        self, sighting_id: str, status: str
    ) -> dict | None: ...
    def update_match_owner_status(
        self, sighting_id: str, pet_id: str, status: str
    ) -> dict | None: ...
    def set_sighting_action_type(
        self, sighting_id: str, hunter_id: str, action_type: str
    ) -> dict | None: ...

    # --- owner side of the loop ------------------------------------------ #
    def get_pet_owners(self, pet_ids: list[str]) -> dict[str, str]: ...

    # --- discovery / match reads ---------------------------------------- #
    def get_sighting_for_match(self, sighting_id: str) -> dict | None: ...
    def match_missing_pets(self, sighting_id: str, limit: int) -> list[dict]: ...
    def get_sighting(self, sighting_id: str) -> dict | None: ...
    def get_sighting_for_action(self, sighting_id: str) -> dict | None: ...

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
