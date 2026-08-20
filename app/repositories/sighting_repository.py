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

    Once an administrator has set `verification_status` away from 'Pending' the
    report has been judged as it stands (in practice: Dismissed by moderation).
    Letting the hunter flip 'Spotted' to 'Caught' afterwards would re-shape a
    report somebody has already ruled on, so the column is frozen at that point.

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


class OwnerDecisionRefused(ValueError):
    """Base for the ways `owner_decide_sighting` can refuse a verdict.

    The RPC is the single authority on the owner's queue rules, so its refusals
    arrive as one PostgREST error each and the adapter turns them back into
    these types by matching on the message the function raises. Every subclass
    is a distinct HTTP status at the route, which is why they are separate
    types rather than one error carrying a string.

    Subclasses ValueError like the other domain errors here, so a route's
    generic `except ValueError -> 400` still catches anything new; the specific
    handlers must be listed FIRST (see tests/test_owner_loop_api.py).
    """


class SightingAlreadyDecided(OwnerDecisionRefused):
    """The owner already ruled on this card — the API's 409.

    Re-deciding is refused rather than overwritten: a verdict is what the
    scoring reads, and a card flipped from Confirmed to Rejected after the fact
    silently changes who gets paid.
    """


class SightingOutOfOrder(OwnerDecisionRefused):
    """An older card on this pet's queue is still undecided — the API's 409.

    The owner rules oldest-first so that everyone who helped is ruled on before
    the rescue closes the case. Enforced in the database, not only in the app:
    the app is code running on someone else's device.
    """


class SearchAlreadyClosed(OwnerDecisionRefused):
    """The pet is already Found/Resolved, so its queue is over — the API's 409.

    This is also the guard that makes the payout unrepeatable: scores are
    distributed exactly once, by the verdict that closes the search.
    """


class SightingRepository(Protocol):
    # --- writes ---------------------------------------------------------- #
    def insert_sighting(self, payload: dict) -> dict: ...
    def upsert_sighting_matches(self, rows: list[dict]) -> None: ...
    def set_sighting_status(
        self, sighting_id: str, status: str
    ) -> dict | None: ...
    def set_sighting_action_type(
        self, sighting_id: str, hunter_id: str, action_type: str
    ) -> dict | None: ...

    # --- owner side of the loop ------------------------------------------ #
    def get_pet_owners(self, pet_ids: list[str]) -> dict[str, str]: ...

    def owner_decide_sighting(
        self, pet_id: str, sighting_id: str, owner_id: str, decision: str,
    ) -> dict:
        """The owner's verdict on one card, via the `owner_decide_sighting` RPC.

        One method, not the four writes it performs, because the verdict, the
        end of the search, the score distribution and the closing of the
        remaining sightings have to land together or not at all — a payout that
        half-happened cannot be reconstructed once the search is closed.

        Returns the RPC's JSON: `{pet_id, sighting_id, owner_status,
        search_closed, pet_status, awards[]}`.

        Raises: LookupError (no such pet/card, or not the caller's),
        SightingAlreadyDecided, SightingOutOfOrder, SearchAlreadyClosed.
        """
        ...

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
    def get_penalties_for_hunter(self, hunter_id: str) -> list[dict]: ...
    def get_user(self, hunter_id: str) -> dict | None: ...
    def count_owner_confirmed_sightings_for_hunter(self, hunter_id: str) -> int: ...
    def count_contributions_for_hunter(self, hunter_id: str) -> int: ...
