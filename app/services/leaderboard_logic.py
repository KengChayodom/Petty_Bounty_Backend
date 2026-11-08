"""Pure ranking arithmetic for the Rank List. No I/O."""


def assign_ranks(scores: list[float], *, offset: int, first_rank: int) -> list[int]:
    """Competition ranks ("1224") for one page of scores, best first.

    `first_rank` is the GLOBAL rank of the first row, worked out by counting how
    many rows outside the page strictly outscore it. That is what lets a later
    page continue the numbering and lets a tie that straddles a page boundary
    keep its shared rank. Equal scores share a rank, and the next lower score
    takes its true position, which is `offset + index + 1`.
    """
    ranks: list[int] = []
    current = first_rank
    for i, score in enumerate(scores):
        if i > 0 and score < scores[i - 1]:
            current = offset + i + 1
        ranks.append(current)
    return ranks
