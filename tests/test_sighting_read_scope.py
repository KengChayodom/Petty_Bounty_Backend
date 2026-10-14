"""
Who may read the matches computed for one sighting.

`GET /sightings/{id}/matches` was authenticated on 2026-09-09 but never scoped:
any signed-in account holding a sighting id could read which missing pets a
sighting resembles. The owner timeline had already been scoped for that reason.
The single-sighting read that shared this rule was removed on 2026-10-14,
nothing calling it. Administrators read a flagged sighting through
`GET /admin/sightings/{id}` instead.

Allowed: the hunter who reported it, and the owner of a pet it is linked to
(matched by the AI, or named by a targeted report). Everyone else gets the same
404 as a sighting that does not exist, so the endpoint cannot be used to find
out which ids are real. A sighting moderation has withdrawn is invisible to the
owner, as it is on their timeline, and stays visible to its own hunter.

Real `SightingService` over a doubled `SightingRepository`; the route cases
prove the endpoints actually ask.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import sightings as sightings_api
from app.core.auth import get_current_user_id
from app.core.database import get_supabase_client
from app.repositories.sighting_repository import SightingRepository
from app.services.sighting_service import SightingService


def run(coro):
    return asyncio.run(coro)


def _service(*, sighting, links=(), owners=None):
    """`links` are the pet ids the AI matched to the sighting; `owners` maps
    pet id to owner id for every pet that exists."""
    repo = MagicMock(spec=SightingRepository)
    repo.get_sighting.return_value = sighting
    repo.get_matches_for_sightings.return_value = [
        {"sighting_id": "s1", "missing_pet_id": pid} for pid in links
    ]
    known = owners or {}
    repo.get_pet_owners.side_effect = lambda ids: {
        i: known[i] for i in ids if i in known
    }
    return SightingService(repo=repo, ai_manager=MagicMock()), repo


def _sighting(**over):
    row = {
        "id": "s1", "hunter_id": "hunter-1", "verification_status": "Pending",
        "initial_target_pet_id": None, "feature_vector": [0.1],
    }
    row.update(over)
    return row


def _matches_service(**kw):
    """The real visibility rule, with the matching itself stood in for."""
    service, repo = _service(**kw)
    service.get_matches = AsyncMock(return_value=[{"id": "p1"}])
    return service, repo


class TestWhoMayReadTheMatches:
    def test_the_owner_of_a_matched_pet_may(self):
        service, _ = _matches_service(
            sighting=_sighting(), links=["p1"], owners={"p1": "owner-1"},
        )
        assert run(service.get_matches_for_viewer("s1", "owner-1")) == [{"id": "p1"}]

    def test_the_owner_of_the_targeted_pet_may(self):
        service, _ = _matches_service(
            sighting=_sighting(initial_target_pet_id="p9"),
            owners={"p9": "owner-9"},
        )
        assert run(service.get_matches_for_viewer("s1", "owner-9")) == [{"id": "p1"}]

    def test_a_stranger_gets_the_same_answer_as_a_missing_sighting(self):
        service, _ = _matches_service(
            sighting=_sighting(), links=["p1"], owners={"p1": "owner-1"},
        )
        with pytest.raises(LookupError) as stranger:
            run(service.get_matches_for_viewer("s1", "stranger"))

        missing, _ = _matches_service(sighting=None)
        with pytest.raises(LookupError) as absent:
            run(missing.get_matches_for_viewer("s1", "stranger"))

        assert str(stranger.value) == str(absent.value)

    def test_a_withdrawn_sighting_is_hidden_from_the_owner_but_not_the_hunter(self):
        row = _sighting(verification_status="Dismissed")
        service, _ = _matches_service(sighting=row, links=["p1"], owners={"p1": "owner-1"})

        with pytest.raises(LookupError):
            run(service.get_matches_for_viewer("s1", "owner-1"))
        assert run(service.get_matches_for_viewer("s1", "hunter-1")) == [{"id": "p1"}]


class TestReadingTheMatches:
    def test_a_stranger_is_refused_before_any_matching_runs(self):
        """Matching is the expensive part and the part that names other pets.
        A refused caller must not trigger either."""
        service, repo = _service(
            sighting=_sighting(), links=["p1"], owners={"p1": "owner-1"},
        )
        service.get_matches = AsyncMock(return_value=[{"id": "p1"}])

        with pytest.raises(LookupError):
            run(service.get_matches_for_viewer("s1", "stranger"))

        service.get_matches.assert_not_called()
        repo.match_missing_pets.assert_not_called()

    def test_the_hunter_gets_the_ranked_matches(self):
        service, _ = _service(sighting=_sighting())
        service.get_matches = AsyncMock(return_value=[{"id": "p1"}])

        out = run(service.get_matches_for_viewer("s1", "hunter-1", 3, 0.5))

        assert out == [{"id": "p1"}]
        service.get_matches.assert_awaited_once_with("s1", 3, 0.5)

    def test_a_sighting_that_cannot_be_matched_is_still_a_bad_request(self):
        """The authorised caller of a targeted sighting (no vector) still gets
        the ValueError the route maps to 400, not a 404."""
        service, _ = _service(sighting=_sighting())
        service.get_matches = AsyncMock(side_effect=ValueError("no vector"))
        with pytest.raises(ValueError):
            run(service.get_matches_for_viewer("s1", "hunter-1"))


def _client(service, user_id):
    app = FastAPI()
    app.include_router(sightings_api.router)
    app.dependency_overrides[get_supabase_client] = lambda: MagicMock()
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[sightings_api.get_sighting_service] = lambda: service
    return TestClient(app, raise_server_exceptions=False)


class TestTheRoutesAskWhoIsReading:
    """This goes through the real service, so it fails if the route is ever
    changed back to the unscoped read."""

    def test_a_stranger_cannot_read_the_matches(self):
        service, repo = _service(sighting=_sighting())
        r = _client(service, "stranger").get("/sightings/s1/matches")
        assert r.status_code == 404
        repo.match_missing_pets.assert_not_called()
