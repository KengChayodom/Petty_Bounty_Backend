"""Tiny read-back helpers for asserting real persisted state.

Every assertion in the integration suite reads state back through a real query
(skill §2.2). table/column args are test-controlled literals, never user input.
"""


def pet_status(conn, pet_id):
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM missing_pets WHERE id = %s", (pet_id,))
        return cur.fetchone()[0]


def total_score(conn, user_id):
    with conn.cursor() as cur:
        cur.execute("SELECT total_score FROM users WHERE id = %s", (user_id,))
        return cur.fetchone()[0]


def row_count(conn, table, column, value):
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table} WHERE {column} = %s", (value,))
        return cur.fetchone()[0]
