-- ============================================================
-- SRS-87 rework: per-post expiry as an absolute timestamp.
--
-- BEFORE: the seven-day rule was the predicate
--   `mp.created_at > NOW() - INTERVAL '7 days'`
-- copy-pasted into match_missing_pets and get_nearby_missing_pets, and mirrored
-- a third time in Python as pet_logic.POST_LIFETIME_DAYS (for the owner's
-- "Expired" badge). One rule, three literals, kept in step only by a comment.
--
-- AFTER: a single missing_pets.expires_at column. The read paths filter
--   `mp.expires_at > NOW()`
-- and the badge reads the column directly. The seven-day grant lives once, in
-- this column's DEFAULT. Extending one post (the paid-extension feature planned
-- for the next progress round, MD-57) becomes a plain
--   `UPDATE missing_pets SET expires_at = ... WHERE id = ...`
-- with no schema change and no touch to the read paths.
--
-- Apply via the Supabase SQL Editor or `supabase db push`.
-- ============================================================

BEGIN;

-- ---------- 1. the column ----------
ALTER TABLE missing_pets
    ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP WITH TIME ZONE;

-- Backfill existing rows to the value the old rule implied: seven days after the
-- report was filed. A report already older than that lands with expires_at in
-- the past and is immediately treated as expired, exactly as before.
UPDATE missing_pets
   SET expires_at = created_at + INTERVAL '7 days'
 WHERE expires_at IS NULL;

ALTER TABLE missing_pets
    ALTER COLUMN expires_at SET NOT NULL;

-- The seven-day grant: the ONLY place the number lives now. Change it here
-- (ALTER COLUMN ... SET DEFAULT) to move every future report's lifetime.
ALTER TABLE missing_pets
    ALTER COLUMN expires_at SET DEFAULT (NOW() + INTERVAL '7 days');

-- The read paths only ever ask for still-live posts.
CREATE INDEX IF NOT EXISTS idx_missing_pets_expires_at
    ON missing_pets (expires_at);

-- ---------- 2. match_missing_pets — filter on expires_at ----------
-- Identical to the 2026-08-21 version apart from the age predicate: the
-- signature and the RETURNS TABLE are unchanged, so CREATE OR REPLACE is safe.
CREATE OR REPLACE FUNCTION match_missing_pets(
    p_sighting_id UUID,
    match_limit   INTEGER
) RETURNS TABLE (
    id                 UUID,
    pet_name           TEXT,
    species            TEXT,
    characteristics    JSONB,
    bounty_amount      NUMERIC,
    last_seen_location TEXT,
    last_seen_time     TEXT,
    image_url          TEXT,
    similarity         DOUBLE PRECISION,
    distance_meters    DOUBLE PRECISION,
    status             TEXT,
    primary_color_hex  TEXT
) LANGUAGE plpgsql AS $$
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
      AND mp.expires_at > NOW()
      AND ST_DWithin(mp.last_seen_location, v_location, 10000)
    ORDER BY mp.feature_vector <=> v_embedding ASC NULLS LAST
    LIMIT match_limit;
END;
$$;

-- ---------- 3. get_nearby_missing_pets — filter on expires_at ----------
-- Body-only change; signature and RETURNS TABLE unchanged.
CREATE OR REPLACE FUNCTION get_nearby_missing_pets(
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
    pattern_id        VARCHAR,
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
        mp.pattern_id,
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

-- ---------- 4. get_missing_pet_by_id — carry expires_at ----------
-- The single-pet read feeds the owner's "Expired" badge (pet_logic), so it must
-- return expires_at. Adding a column to a RETURNS TABLE is a return-type change,
-- which CREATE OR REPLACE cannot do — hence DROP + CREATE. The app is the sole
-- caller (SupabaseMissingPetRepository.get_missing_pet_by_id).
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
        ST_Y(mp.last_seen_location::geometry) AS latitude,
        ST_X(mp.last_seen_location::geometry) AS longitude,
        mp.last_seen_time,
        mp.image_url,
        mp.status::character varying,
        mp.created_at,
        mp.expires_at,
        mp.primary_color_hex,
        mp.pattern_id
    FROM public.missing_pets mp
    WHERE mp.id = p_pet_id;
END;
$$;

COMMIT;
