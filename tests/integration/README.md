# Integration tests — real ephemeral Postgres (pgvector + PostGIS)

These tests run the deployed SQL — `match_missing_pets`, `sightings_for_pet`,
`resolve_missing_pet`, and the `sighting_matches` UNIQUE constraint — against a
**real, disposable Postgres** spun up per session via Testcontainers. No SQLite,
no mocked DB: vector cosine, geography radius, schema constraints and the
transactional payout are exactly what this layer exists to verify.

## Running

```bash
# one-time: install test deps into the test interpreter (Python 3.10+)
python -m pip install -r requirements-test.txt

# requires the Docker daemon running
pytest tests/integration --integration          # this suite
pytest --integration                             # unit + integration
pytest                                           # unit only (integration auto-skipped)
```

The combined pgvector+PostGIS image (`Dockerfile.pg`) is built automatically on
first run and reused thereafter. Without `--integration`, these tests are
skipped and **no container is started**, so the unit suite stays fast.

## How it stays deterministic and isolated

- **Hand-built vectors** (`_helpers.py`) with known dot products — no randomness,
  so similarity values are exact and reproducible.
- **Per-test transaction rollback** (`conn` fixture): every test seeds + calls +
  reads inside one transaction that is always rolled back. Order-independent;
  safe to run 1000× with no data drift.

## Schema provenance

`conftest._apply_schema` applies, in order: `sql/00_prelude.sql` (base schema +
Supabase `auth` shim, RLS omitted — bypassed by the service key in prod) → the
**real** `migrations/2026_05_27_feature2_logging_scoring.sql` → `sql/20_live_match_rpc.sql`
(the deployed 2-arg `match_missing_pets` overload, which lives only in the live
DB — pulled via `pg_get_functiondef`). See the header comments in each file.

## CI staging

| Layer | When to run | Gate |
| :-- | :-- | :-- |
| Unit (`pytest`) | every push | required merge gate |
| **Integration (`--integration`)** | **on PR** (build/cache the image) | **required for any change touching `app/services/*`, `migrations/`, or `sql*.txt`** |
| Mutation / full E2E | nightly | non-blocking report |

Integration belongs on **PR**, not every push: it gates schema/migration/RPC
changes before they reach a shared environment, but a Docker spin-up is too
heavy for a push gate. Keep mutation testing and full E2E on a nightly schedule.
