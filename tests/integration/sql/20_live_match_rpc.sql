-- The DEPLOYED matcher overload the app actually calls.
--
-- PROVENANCE: copied verbatim from the live Supabase project (last re-pulled
--   2026-08-13, after the colour-matching migration) via
--   SELECT pg_get_functiondef(oid) FROM pg_proc WHERE proname='match_missing_pets'
-- and the signature is match_missing_pets(uuid, integer). The final RETURN
-- column `primary_color_hex` feeds the Python colour re-rank (sighting_logic).
--
-- WHY THIS FILE EXISTS: this 2-arg `p_sighting_id` overload is NOT in any repo
-- SQL file — sql-update.txt only defines a different 6-arg variant. CLAUDE.md
-- documents that this overload was deployed directly to Supabase. The service
-- calls it as rpc("match_missing_pets", {"p_sighting_id":…, "match_limit":…})
-- (sighting_service.py:234), so it is the function under test. Keep in sync
-- with the live DB; re-pull with pg_get_functiondef if the deployment changes.
--
-- NOTE the real behaviours this pins: (1) RAISES if the sighting's vector is
-- NULL; (2) radius is HARDCODED at 10 000 m; (3) there is NO similarity
-- threshold, so a pet with a NULL feature_vector within range is still
-- returned (similarity COALESCEd to 0, ordered last via NULLS LAST).

CREATE OR REPLACE FUNCTION public.match_missing_pets(p_sighting_id uuid, match_limit integer)
 RETURNS TABLE(id uuid, pet_name text, species text, characteristics jsonb, bounty_amount numeric, last_seen_location text, last_seen_time text, image_url text, similarity double precision, distance_meters double precision, status text, primary_color_hex text)
 LANGUAGE plpgsql
AS $function$
DECLARE
    v_embedding vector(512);
    v_species pet_species;
    v_location GEOGRAPHY;
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
