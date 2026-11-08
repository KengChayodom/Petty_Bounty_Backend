-- ============================================================
-- Feature: FCM Geolocation Push Alerts (SRS-FR-12)
-- Apply via Supabase SQL Editor or `supabase db push`.
--
-- Adds:
--   1. device_tokens          — one row per device FCM token
--   2. users.last_location    — hunter's current position (for geo-targeting)
--   3. get_nearby_hunters()   — RPC: hunters within radius, fresh + not the owner
--
-- Single transaction: no new enum is introduced (platform is TEXT + CHECK),
-- so the "enum value can't be used in the same tx it's created" constraint
-- that split the Feature #2 migration does not apply here.
-- ============================================================

BEGIN;

-- ---------- 1. device_tokens ----------
-- UNIQUE(fcm_token) lets POST /devices/register upsert on the token, so a
-- token that migrates to a different account reassigns cleanly instead of
-- duplicating. ON DELETE CASCADE drops tokens when a user is removed.
CREATE TABLE device_tokens (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    fcm_token   TEXT NOT NULL UNIQUE,
    platform    TEXT NOT NULL CHECK (platform IN ('android', 'ios')),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- Fan-out reads all of a user's tokens by user_id.
CREATE INDEX device_tokens_user_idx ON device_tokens (user_id);

-- ---------- 2. Hunter current location on users ----------
-- "Current" position (one point per user), used to answer "who is near this
-- newly-reported pet". Nullable: a user who has never shared location simply
-- isn't targetable. last_location_at drives the freshness filter below.
ALTER TABLE users
    ADD COLUMN last_location    GEOGRAPHY(POINT, 4326),
    ADD COLUMN last_location_at TIMESTAMPTZ;

-- GiST index so ST_DWithin hunter lookups use the spatial index (same pattern
-- as the missing_pets / sightings location indexes).
CREATE INDEX users_last_location_gix ON users USING GIST (last_location);

-- ---------- 3. get_nearby_hunters RPC ----------
-- Mirrors the get_nearby_missing_pets / match_missing_pets style: a GEOGRAPHY
-- center + radius_meters + ST_DWithin (GiST-backed). Additionally:
--   * max_age_hours (default 24): a hunter whose last_location_at is older than
--     this window is treated as stale and excluded — a 3-day-old location must
--     never receive a "nearby" alert.
--   * exclude_user_id: the pet owner must not be alerted about their own pet.
CREATE OR REPLACE FUNCTION get_nearby_hunters(
    center_location  GEOGRAPHY,        -- pet's last-seen point (WKT 'POINT(lng lat)')
    radius_meters    FLOAT,            -- search radius in metres
    max_age_hours    INT DEFAULT 24,   -- freshness window for last_location_at
    exclude_user_id  UUID DEFAULT NULL -- owner to skip (no self-alert)
) RETURNS TABLE (
    user_id          UUID,
    distance_meters  FLOAT,
    last_location_at TIMESTAMPTZ   -- returned for staleness debugging
) LANGUAGE plpgsql STABLE AS $$
BEGIN
    RETURN QUERY
    SELECT
        u.id,
        ST_Distance(u.last_location, center_location) AS distance_meters,
        u.last_location_at
    FROM users u
    WHERE u.last_location IS NOT NULL
      AND u.last_location_at IS NOT NULL
      -- freshness: drop stale locations
      AND u.last_location_at >= now() - make_interval(hours => max_age_hours)
      -- proximity: GiST-indexed radius filter
      AND ST_DWithin(u.last_location, center_location, radius_meters)
      -- never alert the owner about their own pet
      AND (exclude_user_id IS NULL OR u.id <> exclude_user_id)
    ORDER BY distance_meters;
END;
$$;

COMMIT;
