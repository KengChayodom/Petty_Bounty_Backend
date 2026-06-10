-- ============================================================
-- Fix: GET /missing-pets/{id} must return the SAME shape as
--      get_nearby_missing_pets (honour the MissingPetResponse contract).
-- Apply via Supabase SQL Editor or `supabase db push`.
--
-- Why: get_missing_pet_by_id previously did `select('*')`, returning the raw
-- row — which has NO latitude/longitude (location is the last_seen_location
-- GEOGRAPHY) and exposes bounty_amount inconsistently. Clients that expect
-- numeric lat/lng/bounty (the Flutter MissingPetModel) crashed. This RPC
-- projects lat/lng with ST_Y/ST_X and returns the identical column set/types
-- as get_nearby_missing_pets (minus distance_meters), so by-id and /nearby
-- are one consistent shape.
-- ============================================================

BEGIN;

CREATE OR REPLACE FUNCTION get_missing_pet_by_id(p_pet_id uuid)
RETURNS TABLE (
    id                uuid,
    owner_id          uuid,
    pet_name          character varying,
    species           character varying,
    characteristics   jsonb,
    bounty_amount     numeric,
    latitude          double precision,
    longitude         double precision,
    last_seen_time    timestamp with time zone,
    image_url         text,
    status            character varying,
    created_at        timestamp with time zone,
    primary_color_hex character varying,
    pattern_id        character varying
) LANGUAGE plpgsql STABLE AS $$
BEGIN
    RETURN QUERY
    SELECT
        mp.id,
        mp.owner_id,
        mp.pet_name,
        mp.species::character varying,
        mp.characteristics,
        mp.bounty_amount,
        ST_Y(mp.last_seen_location::geometry) AS latitude,   -- mirror nearby
        ST_X(mp.last_seen_location::geometry) AS longitude,  -- mirror nearby
        mp.last_seen_time,
        mp.image_url,
        mp.status::character varying,
        mp.created_at,
        mp.primary_color_hex,
        mp.pattern_id
    FROM public.missing_pets mp
    WHERE mp.id = p_pet_id;
END;
$$;

COMMIT;
