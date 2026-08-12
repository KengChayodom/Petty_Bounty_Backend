"""
Integration tests for the owner-scoped missing_pets UPDATE behind
PATCH /missing-pets/{id} (SupabaseMissingPetRepository.update_missing_pet_owned).

The scoping — WHERE id AND owner_id — is a security boundary: a non-owner's
update must match ZERO rows, which is what lets the route return 404 without
leaking whether the pet exists. Only a real UPDATE can prove the row count.
"""
import pytest

pytestmark = pytest.mark.integration


def test_owner_scoped_update_applies_to_owner(conn, seed):
    owner = seed.user()
    pet = seed.missing_pet(owner_id=owner, pet_name="Old")

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE missing_pets SET pet_name = %s "
            "WHERE id = %s AND owner_id = %s RETURNING pet_name",
            ("New", pet, owner),
        )
        rows = cur.fetchall()

    assert len(rows) == 1              # the owner's row matched
    assert rows[0][0] == "New"         # and reflects the change


def test_owner_scoped_update_matches_no_rows_for_non_owner(conn, seed):
    owner = seed.user()
    other = seed.user()
    pet = seed.missing_pet(owner_id=owner, pet_name="Original")

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE missing_pets SET pet_name = %s "
            "WHERE id = %s AND owner_id = %s RETURNING id",
            ("Hacked", pet, other),
        )
        # 0 rows -> the adapter returns None -> the route 404s (no existence leak)
        assert cur.fetchall() == []

        # the pet is untouched
        cur.execute("SELECT pet_name FROM missing_pets WHERE id = %s", (pet,))
        assert cur.fetchone()[0] == "Original"
