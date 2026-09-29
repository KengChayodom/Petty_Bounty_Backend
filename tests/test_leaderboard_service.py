"""
The ranking rules of the Rank List, now that they no longer live inside a route.

`test_leaderboard_api.py` keeps pinning the queries themselves through the route.
What is pinned here is the arithmetic and the assembly, which need no database:

  * a rank is a GLOBAL position ("how many strictly outscore me, plus one"), so a
    later page continues the numbering instead of restarting at 1;
  * equal scores share a rank, and the next distinct score takes its true
    position (two tied at 1, the next is 3, not 2);
  * the caller's own standing is counted the same way and does not depend on
    which page is on show;
  * a missing score reads as zero, never as null.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from app.repositories.leaderboard_repository import LeaderboardRepository
from app.services.leaderboard_logic import assign_ranks
from app.services.leaderboard_service import LeaderboardService


def run(coro):
    return asyncio.run(coro)


class TestAssignRanks:
    def test_an_empty_page_has_no_ranks(self):
        assert assign_ranks([], offset=0, first_rank=1) == []

    def test_one_row_takes_the_rank_it_was_given(self):
        assert assign_ranks([50], offset=0, first_rank=7) == [7]

    def test_distinct_scores_count_up_from_the_first_rank(self):
        assert assign_ranks([90, 80, 70], offset=0, first_rank=1) == [1, 2, 3]

    def test_equal_scores_share_a_rank_and_the_next_takes_its_true_position(self):
        assert assign_ranks([90, 90, 80], offset=0, first_rank=1) == [1, 1, 3]

    def test_a_later_page_continues_the_numbering(self):
        # Page 2 of size 3: the caller counted 3 people above the first row.
        assert assign_ranks([70, 60, 50], offset=3, first_rank=4) == [4, 5, 6]

    def test_a_tie_that_runs_across_the_first_row_keeps_its_shared_rank(self):
        # The first row of page 2 ties with the last row of page 1: both rank 3.
        assert assign_ranks([80, 70], offset=3, first_rank=3) == [3, 5]

    def test_a_drop_after_a_tie_jumps_by_the_size_of_the_tie(self):
        assert assign_ranks([90, 80, 80, 80, 60], offset=0, first_rank=1) == [
            1, 2, 2, 2, 5]

    def test_zero_scores_rank_like_any_other_score(self):
        assert assign_ranks([5, 0, 0], offset=0, first_rank=1) == [1, 2, 2]


def _service(*, users=(), above=0, me=None, bounties=(), bounties_above=0):
    repo = MagicMock(spec=LeaderboardRepository)
    repo.list_users_page.return_value = list(users)
    repo.count_users_above.return_value = above
    repo.get_user_standing.return_value = me
    repo.list_open_bounties_page.return_value = list(bounties)
    repo.count_open_bounties_above.return_value = bounties_above
    return LeaderboardService(repo), repo


def _user(uid, score, name=None):
    return {"id": uid, "username": name or uid, "profile_image_url": None,
            "total_score": score}


class TestUsersPage:
    def test_ranks_come_from_the_global_count_not_the_page_position(self):
        """Page 2 starts at position 21, but two people on page 1 tie with the
        first row of this page, so only 18 strictly outscore it and it ranks 19.
        Numbering by page position would say 21."""
        service, repo = _service(
            users=[_user("a", 70), _user("b", 60)], above=18, me=None)

        page = run(service.users_page("viewer", limit=2, offset=20))

        assert [e["rank"] for e in page["entries"]] == [19, 22]
        repo.count_users_above.assert_any_call(70)

    def test_entry_shape_is_the_one_the_app_reads(self):
        service, _ = _service(users=[_user("a", 70, "Somchai")], above=0)
        entry = run(service.users_page("v", limit=5, offset=0))["entries"][0]
        assert entry == {
            "rank": 1, "user_id": "a", "username": "Somchai",
            "profile_image_url": None, "total_score": 70,
        }

    def test_a_null_score_reads_as_zero(self):
        service, _ = _service(users=[_user("a", None)], above=4)
        entry = run(service.users_page("v", limit=5, offset=0))["entries"][0]
        assert entry["total_score"] == 0

    def test_my_standing_counts_who_outscores_me(self):
        service, repo = _service(
            users=[_user("a", 90)],
            me={"username": "me", "profile_image_url": "u", "total_score": 40},
        )
        repo.count_users_above.side_effect = lambda s: {90: 0, 40: 6}[s]

        me = run(service.users_page("viewer-1", limit=5, offset=0))["me"]

        assert me == {"rank": 7, "user_id": "viewer-1", "username": "me",
                      "profile_image_url": "u", "total_score": 40}

    def test_a_caller_with_no_row_reads_as_zero_and_still_gets_a_standing(self):
        service, repo = _service(users=[], me=None, above=12)
        page = run(service.users_page("ghost", limit=5, offset=0))
        assert page["entries"] == []
        assert page["me"]["total_score"] == 0
        assert page["me"]["rank"] == 13
        assert page["me"]["username"] is None

    def test_an_empty_page_asks_for_no_first_row_rank(self):
        service, repo = _service(users=[], me=None)
        run(service.users_page("v", limit=5, offset=0))
        # only the caller's own standing is counted
        assert repo.count_users_above.call_count == 1


class TestBountiesPage:
    def test_ranks_and_amounts_are_numbers(self):
        rows = [
            {"id": "p1", "pet_name": "A", "image_url": "u", "bounty_amount": "500"},
            {"id": "p2", "pet_name": "B", "image_url": None, "bounty_amount": None},
        ]
        service, repo = _service(bounties=rows, bounties_above=2)

        entries = run(service.bounties_page(limit=5, offset=2))["entries"]

        assert [e["rank"] for e in entries] == [3, 4]
        assert [e["bounty_amount"] for e in entries] == [500.0, 0.0]
        assert entries[0] == {"rank": 3, "pet_id": "p1", "pet_name": "A",
                              "image_url": "u", "bounty_amount": 500.0}
        repo.count_open_bounties_above.assert_called_once_with(500.0)

    def test_equal_bounties_share_a_rank(self):
        rows = [{"id": f"p{i}", "pet_name": "x", "image_url": None,
                 "bounty_amount": 100} for i in range(3)]
        service, _ = _service(bounties=rows)
        entries = run(service.bounties_page(limit=5, offset=0))["entries"]
        assert [e["rank"] for e in entries] == [1, 1, 1]

    def test_an_empty_board_is_an_empty_list(self):
        service, repo = _service(bounties=[])
        assert run(service.bounties_page(limit=5, offset=0)) == {"entries": []}
        repo.count_open_bounties_above.assert_not_called()

    def test_a_repository_failure_propagates_for_the_route_to_report(self):
        service, repo = _service()
        repo.list_open_bounties_page.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            run(service.bounties_page(limit=5, offset=0))
