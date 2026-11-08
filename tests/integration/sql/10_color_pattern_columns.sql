-- Test-only shim: the primary_color_hex column.
--
-- It exists on the DEPLOYED missing_pets table (added directly in Supabase) but
-- is NOT defined in any repo migration or in sql.txt. The get_missing_pet_by_id
-- RPC projects it, so the integration schema must include it for the RPC to
-- execute. This mirrors production state and adds no new behaviour.
--
-- Applied AFTER the prelude (which creates missing_pets) and BEFORE the by-id
-- fix migration (which references the column).
--
-- pattern_id was added here for the same reason and removed on 2026-09-05 with
-- migration 2026_09_05_drop_pattern_id.sql. The harness applies that migration's
-- rebuilt functions, so a column re-added here would only mask a stale one.
ALTER TABLE missing_pets
    ADD COLUMN IF NOT EXISTS primary_color_hex character varying;

-- sightings gets its OWN auto-extracted coat colour (migration
-- migrations/2026-08-13_colour_matching.sql on prod). The live match RPC in
-- 20_live_match_rpc.sql projects missing_pets.primary_color_hex; the sightings
-- column feeds the query side (repo get_sighting_for_match). Mirror it here.
ALTER TABLE sightings
    ADD COLUMN IF NOT EXISTS primary_color_hex character varying;
