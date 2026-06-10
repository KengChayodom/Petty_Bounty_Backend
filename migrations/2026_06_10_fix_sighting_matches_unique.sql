-- ============================================================
-- Fix: sighting_matches must have UNIQUE (sighting_id, missing_pet_id).
-- Apply via Supabase SQL Editor or `supabase db push`.
--
-- Why: SightingService._persist_matches upserts AI match results with
--   .upsert(rows, on_conflict="sighting_id,missing_pet_id")
-- which generates INSERT ... ON CONFLICT (sighting_id, missing_pet_id). That
-- requires a UNIQUE constraint/index on exactly those columns — but it was
-- never created (not in sql.txt, not in any migration; the Feature#2 migration
-- only adds UNIQUE on the *score_awards* table). On the live DB Postgres
-- rejected every such upsert with 42P10 ("no unique or exclusion constraint
-- matching the ON CONFLICT specification"), and the error was swallowed by the
-- persist try/except — so NO match rows were written. Verified live: writes
-- stopped 2026-05-31 despite 16 sightings created afterward; owner timelines
-- (get_sightings_for_pet) and F1 scoring (resolve_missing_pet) silently missed
-- every AI-matched sighting for ~10 days.
--
-- Safe to apply: verified 0 existing duplicate (sighting_id, missing_pet_id)
-- pairs on live, so no dedup is needed. Idempotent — guarded so re-running is a
-- no-op. After this, _persist_matches upserts succeed and re-matches refresh
-- similarity_score in place instead of erroring.
--
-- NOTE: this does NOT backfill the ~16 sightings whose matches were dropped
-- while the upsert was broken — they re-persist only on a new analyze/save.
-- Backfill is an optional, separate follow-up.
-- ============================================================

BEGIN;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'public.sighting_matches'::regclass
          AND conname  = 'sighting_matches_sighting_pet_uniq'
    ) THEN
        ALTER TABLE public.sighting_matches
            ADD CONSTRAINT sighting_matches_sighting_pet_uniq
            UNIQUE (sighting_id, missing_pet_id);
    END IF;
END
$$;

COMMIT;
