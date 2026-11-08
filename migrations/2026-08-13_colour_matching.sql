-- Colour-aware matching: make the coat colour flow end-to-end.
--
-- Context (2026-08-13): the colour re-rank in app/services/sighting_logic.py
-- was dead in production because the colour never reached it:
--   1. `sightings` had no `primary_color_hex` column (the query side's colour
--      was never stored — and the INSERT of it would error);
--   2. the deployed match_missing_pets(uuid,int) RPC did not RETURN the
--      candidate's `primary_color_hex`.
-- Both are fixed here. `get_sighting_for_match` (repo) now also SELECTs the
-- column. Run this whole file once against the live Supabase project.

-- 1) Query-side colour storage --------------------------------------------- --
ALTER TABLE sightings ADD COLUMN IF NOT EXISTS primary_color_hex varchar(7);

COMMENT ON COLUMN sightings.primary_color_hex IS
  'Coat colour auto-extracted from the isolated subject (#RRGGBB); NULL when '
  'extraction was skipped (full-frame fallback or near-black subject). Feeds '
  'the colour-aware match re-rank.';

-- 2) Candidate-side colour in the RPC output ------------------------------- --
-- Adding a column to RETURNS TABLE changes the function's return type, so the
-- existing (uuid,int) overload must be dropped before recreation. Only that
-- overload is touched; the 6-arg variant in sql-update.txt is untouched.
DROP FUNCTION IF EXISTS match_missing_pets(uuid, integer);

CREATE OR REPLACE FUNCTION public.match_missing_pets(
    p_sighting_id uuid,
    match_limit   integer
)
RETURNS TABLE(
    id                 uuid,
    pet_name           text,
    species            text,
    characteristics    jsonb,
    bounty_amount      numeric,
    last_seen_location text,
    last_seen_time     text,
    image_url          text,
    similarity         double precision,
    distance_meters    double precision,
    status             text,
    primary_color_hex  text          -- NEW: lets the Python re-rank judge colour
)
LANGUAGE plpgsql
AS $function$
DECLARE
    v_embedding vector(512);
    v_species   pet_species;
    v_location  GEOGRAPHY;
BEGIN
    SELECT feature_vector, detected_species, sighted_location
    INTO v_embedding, v_species, v_location
    FROM sightings
    WHERE sightings.id = p_sighting_id;

    IF v_embedding IS NULL THEN
        RAISE EXCEPTION 'Sighting % not found or missing vector', p_sighting_id;
    END IF;

    RETURN QUERY
    SELECT
        mp.id,
        mp.pet_name::text,
        mp.species::text,
        mp.characteristics,
        mp.bounty_amount,
        ST_AsText(mp.last_seen_location::geometry)::text     AS last_seen_location,
        to_char(mp.last_seen_time AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS"Z"')                AS last_seen_time,
        mp.image_url,
        COALESCE(1 - (mp.feature_vector <=> v_embedding), 0)::float AS similarity,
        ST_Distance(mp.last_seen_location, v_location)::float       AS distance_meters,
        mp.status::text,
        mp.primary_color_hex::text
    FROM missing_pets mp
    WHERE LOWER(mp.species::text) = LOWER(v_species::text)
      AND mp.status = 'Searching'
      AND ST_DWithin(mp.last_seen_location, v_location, 10000)
    ORDER BY mp.feature_vector <=> v_embedding ASC NULLS LAST
    LIMIT match_limit;
END;
$function$;

-- DROP FUNCTION wiped the old EXECUTE grants — restore them so the same roles
-- can call the recreated overload (the backend calls it as service_role).
GRANT EXECUTE ON FUNCTION match_missing_pets(uuid, integer)
    TO anon, authenticated, service_role;
