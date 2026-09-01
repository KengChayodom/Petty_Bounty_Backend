"""Supabase adapter for MissingPetRepository.

The ONLY place the missing-pet flow touches supabase-py. Each method is the
`.table(...)/.rpc(...).execute()` chain moved verbatim out of PetService, plus
the `.data` unwrap and the one INSERT error-translation (MissingPetNotSaved).
"""
from app.repositories.missing_pet_repository import MissingPetNotSaved
from app.repositories.pagination import Page


class SupabaseMissingPetRepository:
    def __init__(self, db):
        self._db = db

    def insert_missing_pet(self, payload: dict) -> dict:
        res = self._db.table("missing_pets").insert(payload).execute()
        if not res.data:
            raise MissingPetNotSaved(payload)
        return res.data[0]

    def get_missing_pet_by_id(self, pet_id: str) -> dict | None:
        res = self._db.rpc(
            "get_missing_pet_by_id", {"p_pet_id": pet_id}
        ).execute()
        return res.data[0] if res.data else None

    def get_by_owner(self, owner_id: str) -> list[dict]:
        # MD-34 "My Reports": owner-scoped, newest first. Ordering is done in
        # SQL, not in Python, so paging can be added without re-sorting client
        # side.
        res = (self._db.table("missing_pets")
                       .select("*")
                       .eq("owner_id", owner_id)
                       .order("created_at", desc=True)
                       .execute())
        return res.data or []

    def get_sighting_links_for_pets(self, pet_ids: list[str]) -> list[dict]:
        """Every sighting recorded against each pet, as `{pet_id, sighting_id,
        owner_status}` rows.

        Reads the SAME two sources the `sightings_for_pet` RPC unions, because
        these rows become the "RECEIVED n ENTRIES" label on a card whose button
        opens that RPC's list — read one source only and the label would
        disagree with what the owner sees when they tap it:

          1. `sighting_matches` — what the AI matched to this pet;
          2. `sightings.initial_target_pet_id` — a hunter reporting this exact
             pet from its detail page (the targeted flow, SRS-50), which never
             produces a match row and so carries no owner verdict.

        Deliberately NO counting, de-duplication, or filtering here: what counts
        as a sighting of a pet is a product rule (`pet_logic.count_sightings`),
        and keeping it out of the adapter is what lets a unit test pin it down —
        this layer only runs against a real database.
        """
        if not pet_ids:
            return []

        links: list[dict] = []

        matches = (self._db.table("sighting_matches")
                           .select("missing_pet_id, sighting_id, owner_status")
                           .in_("missing_pet_id", pet_ids)
                           .execute())
        for row in matches.data or []:
            links.append({
                "pet_id": row.get("missing_pet_id"),
                "sighting_id": row.get("sighting_id"),
                "owner_status": row.get("owner_status"),
            })

        targeted = (self._db.table("sightings")
                            .select("id, initial_target_pet_id")
                            .in_("initial_target_pet_id", pet_ids)
                            .execute())
        for row in targeted.data or []:
            links.append({
                "pet_id": row.get("initial_target_pet_id"),
                "sighting_id": row.get("id"),
                "owner_status": None,
            })

        return links

    def list_all(
        self, status: str | None, species: str | None, limit: int, offset: int
    ) -> Page:
        # MD-37 admin browse. The status filter is applied ONLY when supplied —
        # `None` must mean "every status", not "status IS NULL".
        #
        # count="exact" makes PostgREST report the number of rows matching the
        # filter in the Content-Range header, so the page and its total arrive
        # in ONE round trip rather than a list query plus a count query.
        query = self._db.table("missing_pets").select("*", count="exact")
        if status is not None:
            query = query.eq("status", status)
        if species is not None:
            query = query.eq("species", species)
        res = (query.order("created_at", desc=True)
                    .range(offset, offset + limit - 1)
                    .execute())
        rows = res.data or []
        # `count` is None if the header is missing (an old PostgREST, or a
        # double that does not model it). Falling back to the page length keeps
        # the caller working; it only understates the total.
        total = getattr(res, "count", None)
        return Page(rows, len(rows) if total is None else total)

    def remove(self, pet_id: str) -> dict | None:
        # MD-38 / SRS-65: UD-14's postcondition is "removed from the database
        # and the search map", so this is a hard DELETE (FKs cascade to
        # sighting_matches). Returns the deleted row, or None when nothing
        # matched — the caller turns that into 404.
        res = (self._db.table("missing_pets")
                       .delete()
                       .eq("id", pet_id)
                       .execute())
        return res.data[0] if res.data else None

    def close_sightings_for_pet(self, pet_id: str) -> int:
        """Mark every sighting recorded against a pet as `Closed`.

        Called when the owner ends the search: the entries stay readable in the
        timeline, but their lifecycle is over, so nothing downstream treats them
        as live leads on a pet that is already home.

        Covers the same two sources as `count_sightings_for_pets` — matched and
        targeted — because those are exactly the entries the owner was shown.
        PostgREST cannot express the `IN (subquery)` this would be in SQL, so
        the match ids are read first and the update is issued against them.

        Returns how many sighting rows were closed.
        """
        sighting_ids: set[str] = set()

        matches = (self._db.table("sighting_matches")
                           .select("sighting_id")
                           .eq("missing_pet_id", pet_id)
                           .execute())
        for row in matches.data or []:
            if row.get("sighting_id"):
                sighting_ids.add(row["sighting_id"])

        closed = 0
        if sighting_ids:
            res = (self._db.table("sightings")
                           .update({"sighting_status": "Closed"})
                           .in_("id", sorted(sighting_ids))
                           .execute())
            closed += len(res.data or [])

        # Targeted reports carry no match row, so they need their own update.
        res = (self._db.table("sightings")
                       .update({"sighting_status": "Closed"})
                       .eq("initial_target_pet_id", pet_id)
                       .execute())
        for row in res.data or []:
            # A targeted sighting can ALSO have been matched to the same pet;
            # count it once.
            if row.get("id") not in sighting_ids:
                closed += 1

        return closed

    def update_missing_pet_owned(
        self, pet_id: str, owner_id: str, patch: dict
    ) -> dict | None:
        # Owner-scoped: matches a row only if it belongs to owner_id. A pet that
        # doesn't exist OR isn't theirs both yield no rows (caller -> 404).
        res = (self._db.table("missing_pets")
                       .update(patch)
                       .eq("id", pet_id)
                       .eq("owner_id", owner_id)
                       .execute())
        return res.data[0] if res.data else None

    def sightings_for_pet(
        self, pet_id: str, limit: int, offset: int, include_dismissed: bool
    ) -> list[dict]:
        res = self._db.rpc("sightings_for_pet", {
            "p_pet_id":            pet_id,
            "p_limit":             limit,
            "p_offset":            offset,
            "p_include_dismissed": include_dismissed,
        }).execute()
        return res.data or []

    def get_nearby_missing_pets(
        self, center_wkt: str, radius_meters: float, limit: int
    ) -> list[dict]:
        res = self._db.rpc('get_nearby_missing_pets', {
            'center_location': center_wkt,
            'radius_meters': radius_meters,
            'limit': limit,
        }).execute()
        return res.data if res.data else []
