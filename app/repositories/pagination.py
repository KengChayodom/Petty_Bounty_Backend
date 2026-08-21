"""One page of rows, plus how many rows there are in total.

A listing endpoint that returns only rows forces its client to guess: the
console could offer "next" whenever a page came back full and find out it was
wrong by loading an empty page. Numbered pages cannot be drawn from a guess at
all. `total` is the count of rows matching the filter *before* limit/offset are
applied, which is what PostgREST returns for `count="exact"`.
"""
from typing import NamedTuple


class Page(NamedTuple):
    items: list[dict]
    total: int
