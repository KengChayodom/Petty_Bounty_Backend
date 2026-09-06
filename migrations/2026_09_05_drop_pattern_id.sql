-- Drop missing_pets.pattern_id.
--
-- The coat-pattern field is gone from the product. No screen writes it: the
-- create form has no pattern picker and LostPetPostRequest stopped sending the
-- key, so every row written since then carries NULL. Its validation requirement
-- was withdrawn on 2026-09-04 and, because that renumbering closed the gap, the
-- rule now carries no SRS identifier at all.
--
-- Verified empty before writing this migration:
--     SELECT count(*) FROM missing_pets WHERE pattern_id IS NOT NULL;  -- 0
-- Re-run that on the target database before applying. A non-zero count means
-- somebody wrote the field after this was authored, and the read path that used
-- to display it has already been removed from the client.
--
-- Two functions project the column and must be rebuilt first. Neither can be
-- CREATE OR REPLACE'd, because removing a column from a RETURNS TABLE is a
-- return-type change. Postgres does not track a plpgsql body as a dependency of
-- the column, so dropping the column without rebuilding them would leave both
-- functions to fail at call time with "column mp.pattern_id does not exist"
-- rather than failing here.
--
-- These bodies are copied from the latest migration that defines each function
-- (get_nearby_missing_pets from 2026-09-01_post_expires_at.sql,
-- get_missing_pet_by_id from 2026_09_02_pet_owner_details.sql) with the one
-- column removed and nothing else changed. Confirm against the deployed
-- definitions before applying:
--     SELECT pg_get_functiondef(oid) FROM pg_proc
--      WHERE proname IN ('get_nearby_missing_pets', 'get_missing_pet_by_id');

BEGIN;

-- ---------- 1. get_nearby_missing_pets — the Home Map list ----------

DROP FUNCTION IF EXISTS get_nearby_missing_pets(TEXT, DOUBLE PRECISION, INTEGER);

CREATE FUNCTION get_nearby_missing_pets(
    center_location TEXT,
    radius_meters   DOUBLE PRECISION,
    "limit"         INTEGER DEFAULT 20
) RETURNS TABLE (
    id                UUID,
    owner_id          UUID,
    pet_name          VARCHAR,
    species           VARCHAR,
    characteristics   JSONB,
    bounty_amount     NUMERIC,
    latitude          DOUBLE PRECISION,
    longitude         DOUBLE PRECISION,
    last_seen_time    TIMESTAMP WITH TIME ZONE,
    image_url         TEXT,
    status            VARCHAR,
    created_at        TIMESTAMP WITH TIME ZONE,
    primary_color_hex VARCHAR,
    distance_meters   DOUBLE PRECISION
) LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        mp.id,
        mp.owner_id,
        mp.pet_name,
        mp.species::character varying,
        mp.characteristics,
        mp.bounty_amount,
        ST_Y(mp.last_seen_location::geometry) AS latitude,
        ST_X(mp.last_seen_location::geometry) AS longitude,
        mp.last_seen_time,
        mp.image_url,
        mp.status::character varying,
        mp.created_at,
        mp.primary_color_hex,
        ST_Distance(
            mp.last_seen_location::geography,
            ST_GeomFromText(center_location, 4326)::geography
        ) AS distance_meters
    FROM public.missing_pets mp
    WHERE mp.status::character varying = 'Searching'
      AND mp.expires_at > NOW()
      AND ST_DWithin(
          mp.last_seen_location::geography,
          ST_GeomFromText(center_location, 4326)::geography,
          radius_meters
      )
    ORDER BY distance_meters
    LIMIT "limit";
END;
$$;

-- ---------- 2. get_missing_pet_by_id — the single-pet detail read ----------

DROP FUNCTION IF EXISTS get_missing_pet_by_id(uuid);

CREATE FUNCTION get_missing_pet_by_id(p_pet_id uuid)
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
    expires_at        timestamp with time zone,
    primary_color_hex character varying,
    owner_display_name character varying,
    owner_phone       character varying,
    owner_profile_image_url text
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
        ST_Y(mp.last_seen_location::geometry) AS latitude,
        ST_X(mp.last_seen_location::geometry) AS longitude,
        mp.last_seen_time,
        mp.image_url,
        mp.status::character varying,
        mp.created_at,
        mp.expires_at,
        mp.primary_color_hex,
        u.display_name,
        u.phone,
        u.profile_image_url
    FROM public.missing_pets mp
    LEFT JOIN public.users u ON u.id = mp.owner_id
    WHERE mp.id = p_pet_id;
END;
$$;

-- ---------- 3. the column itself ----------

ALTER TABLE public.missing_pets DROP COLUMN IF EXISTS pattern_id;

COMMIT;
