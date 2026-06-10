-- Integration-test schema PRELUDE — the pre-Feature#2 base state.
--
-- This is the deployed base schema (verbatim from Petty_Bounty_Backend/sql.txt:
-- extensions, enum types, tables) with TWO test-only shims and TWO deliberate
-- omissions, all explained here:
--
--   SHIM 1 — `auth` schema + `auth.users`: production runs on Supabase, where
--     `public.users.id` FKs to `auth.users(id)`. Plain Postgres has no `auth`
--     schema, so we create a minimal stub to keep the real FK intact rather
--     than editing the production DDL.
--   SHIM 2 — `auth.uid()`: a no-op stub so any auth-coupled object parses. (No
--     RLS is created here; see omission 1.)
--
--   OMISSION 1 — RLS policies (sql-update.txt:155-169): they call `auth.uid()`
--     and gate on the *anon/authenticated* roles. Production reads/writes go
--     through the Supabase SERVICE KEY, which BYPASSES RLS entirely, so RLS is
--     not part of the behaviour these RPCs exhibit in production. Including it
--     would only add Supabase-auth noise.
--   OMISSION 2 — the HNSW vector index (sql-update.txt:104): pgvector's HNSW is
--     an APPROXIMATE index; with it, `ORDER BY ... <=> ...` could perturb the
--     ordering of near-tie similarities and make assertions flaky. It is a
--     query accelerator, not a correctness constraint, so we omit it to keep
--     ranking exact and deterministic. (GIST/geography is exact; also omitted
--     as it changes no results on tiny test data.)
--
-- The real Feature#2 migration is applied ON TOP of this (see conftest), then
-- the live `match_missing_pets(uuid,int)` overload. This file MUST represent
-- the state BEFORE that migration (no verification_status, no 'Resolved', no
-- sighting_matches UNIQUE) so the migration's ALTERs apply cleanly.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

-- ---- SHIM: Supabase auth surface -----------------------------------------
CREATE SCHEMA IF NOT EXISTS auth;
CREATE TABLE IF NOT EXISTS auth.users (
    id    UUID PRIMARY KEY,
    email TEXT
);
CREATE OR REPLACE FUNCTION auth.uid() RETURNS UUID
    LANGUAGE sql STABLE AS $$ SELECT NULL::uuid $$;

-- ---- Enum types (pre-Feature#2: pet_status has NO 'Resolved') -------------
CREATE TYPE user_role AS ENUM ('user', 'admin');
CREATE TYPE pet_species AS ENUM ('Cat', 'Dog', 'Bird', 'Other');
CREATE TYPE pet_status AS ENUM ('Searching', 'Spotted', 'Found');
CREATE TYPE action_type AS ENUM ('Spotted', 'Caught');
CREATE TYPE sighting_status AS ENUM ('Pending_Analysis', 'Notified_Owner', 'Confirmed', 'Closed');
CREATE TYPE owner_decision AS ENUM ('Pending', 'Confirmed', 'Rejected');
CREATE TYPE transaction_status AS ENUM ('Pending_Verification', 'Verified', 'Rejected');
CREATE TYPE report_reason AS ENUM ('Spam', 'Not_a_pet', 'Inappropriate_image');
CREATE TYPE report_status AS ENUM ('Pending', 'Reviewed_Ban', 'Dismissed');

-- ---- Tables (pre-Feature#2 shape) ----------------------------------------
CREATE TABLE users (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    display_name VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    role user_role DEFAULT 'user',
    total_score INTEGER DEFAULT 0,
    profile_image_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE missing_pets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID REFERENCES users(id) ON DELETE CASCADE,
    pet_name VARCHAR(255) NOT NULL,
    species pet_species NOT NULL,
    characteristics JSONB NOT NULL,
    bounty_amount DECIMAL(12, 2) NOT NULL,
    last_seen_location GEOGRAPHY(POINT, 4326) NOT NULL,
    last_seen_time TIMESTAMP WITH TIME ZONE NOT NULL,
    image_url TEXT NOT NULL,
    feature_vector vector(512),
    status pet_status DEFAULT 'Searching',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE sightings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hunter_id UUID REFERENCES users(id) ON DELETE SET NULL,
    sighted_location GEOGRAPHY(POINT, 4326) NOT NULL,
    image_url TEXT NOT NULL,
    detected_species pet_species,
    detected_characteristics JSONB,
    feature_vector vector(512),
    initial_target_pet_id UUID REFERENCES missing_pets(id) ON DELETE SET NULL,
    action_type action_type NOT NULL,
    sighting_status sighting_status DEFAULT 'Pending_Analysis',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE sighting_matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sighting_id UUID REFERENCES sightings(id) ON DELETE CASCADE,
    missing_pet_id UUID REFERENCES missing_pets(id) ON DELETE CASCADE,
    similarity_score DECIMAL(5, 4),
    owner_status owner_decision DEFAULT 'Pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE bounty_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sighting_id UUID REFERENCES sightings(id) NOT NULL,
    missing_pet_id UUID REFERENCES missing_pets(id) NOT NULL,
    owner_id UUID REFERENCES users(id),
    amount DECIMAL(12, 2) NOT NULL,
    slip_image_url TEXT NOT NULL,
    reference_no VARCHAR(255),
    transfer_datetime TIMESTAMP WITH TIME ZONE,
    score_awarded INTEGER DEFAULT 0,
    status transaction_status DEFAULT 'Pending_Verification',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    verified_at TIMESTAMP WITH TIME ZONE,
    verified_by UUID REFERENCES users(id)
);

CREATE TABLE reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reporter_id UUID REFERENCES users(id),
    sighting_id UUID REFERENCES sightings(id),
    reason report_reason NOT NULL,
    status report_status DEFAULT 'Pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
