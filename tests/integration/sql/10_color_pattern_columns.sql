-- Test-only shim: primary_color_hex + pattern_id columns.
--
-- These columns exist on the DEPLOYED missing_pets table (added directly in
-- Supabase) but are NOT defined in any repo migration or in sql.txt. The
-- get_missing_pet_by_id RPC (migrations/2026_06_10_fix_get_missing_pet_by_id.sql)
-- projects them, so the integration schema must include them for the RPC to
-- execute. This mirrors production state; it adds no new behaviour.
--
-- Applied AFTER the prelude (which creates missing_pets) and BEFORE the by-id
-- fix migration (which references these columns).
ALTER TABLE missing_pets
    ADD COLUMN IF NOT EXISTS primary_color_hex character varying,
    ADD COLUMN IF NOT EXISTS pattern_id        character varying;

-- sightings gets its OWN auto-extracted coat colour (migration
-- migrations/2026-08-13_colour_matching.sql on prod). The live match RPC in
-- 20_live_match_rpc.sql projects missing_pets.primary_color_hex; the sightings
-- column feeds the query side (repo get_sighting_for_match). Mirror it here.
ALTER TABLE sightings
    ADD COLUMN IF NOT EXISTS primary_color_hex character varying;
