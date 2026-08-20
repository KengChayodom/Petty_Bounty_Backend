"""
Unit tests for the translation of `owner_decide_sighting`'s refusals into
domain errors — `supabase_sighting_repository._translate_decision_error`.

Why this file exists at all: the adapter layer is omitted from coverage on the
grounds that it is thin vendor translation verified against a real database.
This function is the exception. It is not a query — it is a lookup table
mapping the TEXT of a plpgsql RAISE onto an HTTP status, and PostgreSQL offers
no machine-readable code that survives PostgREST, so those message fragments
are duplicated: once in the migration that raises them, once here. Nothing
about that duplication is visible to either the integration suite (which calls
the function directly through psycopg and never touches the adapter) or the
service tests (which double the port).

So the last test in this file reads the migration and asserts that every
fragment still appears in it. Without it, rewording a RAISE turns an ordinary
out-of-turn tap into a 500 and no existing test notices.
"""
from pathlib import Path

import pytest

from app.repositories.sighting_repository import (
    SearchAlreadyClosed,
    SightingAlreadyDecided,
    SightingOutOfOrder,
)
from app.repositories.supabase_sighting_repository import (
    _DECISION_ERRORS,
    _translate_decision_error,
)

MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "migrations"
    / "2026_08_21_owner_driven_resolution.sql"
)


class _PostgrestError(Exception):
    """Shaped like the vendor error: carries the text on `.message`."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


@pytest.mark.parametrize("message, expected", [
    ("Missing pet abc not found or not owned by you", LookupError),
    ("Sighting s1 is not a match for pet p1", LookupError),
    ("Sighting s1 has already been decided (Confirmed)", SightingAlreadyDecided),
    ("Sighting s1 is out of order: 2 earlier sighting(s) are still undecided",
     SightingOutOfOrder),
    ("Search for pet p1 is already closed", SearchAlreadyClosed),
    ("decision must be Confirmed or Rejected (got Maybe)", ValueError),
])
def test_each_refusal_maps_to_its_status(message, expected):
    assert type(_translate_decision_error(_PostgrestError(message))) is expected


def test_the_original_text_is_preserved():
    """The message names which card and how many are ahead of it. That is what
    the client shows the owner, so it must not be replaced with a generic."""
    text = "Sighting s1 is out of order: 2 earlier sighting(s) are still undecided"

    assert str(_translate_decision_error(_PostgrestError(text))) == text


def test_an_unrecognised_failure_is_handed_back_untouched():
    """A dropped connection or a bug in the function must surface as a 500. A
    catch-all here would tell the owner their perfectly valid request was
    invalid, and would hide a broken procedure behind a 400 forever."""
    original = RuntimeError("connection reset by peer")

    assert _translate_decision_error(original) is original


def test_a_plain_exception_without_a_message_attribute_still_matches():
    """psycopg and postgrest raise different shapes; the text is the contract,
    not the attribute it arrives on."""
    result = _translate_decision_error(
        Exception("Search for pet p1 is already closed")
    )

    assert isinstance(result, SearchAlreadyClosed)


def test_every_fragment_is_still_raised_by_the_migration():
    """The duplication guard. These fragments are the only link between what
    the database refuses and what the API answers; if a RAISE is reworded and
    this list is not, every refusal of that kind silently becomes a 500."""
    sql = MIGRATION.read_text(encoding="utf-8")

    missing = [f for f, _ in _DECISION_ERRORS if f not in sql]

    assert missing == [], (
        f"fragments no longer raised by {MIGRATION.name}: {missing} — "
        "reword the adapter's table to match, or the refusals become 500s"
    )
