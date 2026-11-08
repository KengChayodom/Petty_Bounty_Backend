-- ============================================================
-- Feature #2 — Logging & Scoring
-- Apply via Supabase SQL Editor or `supabase db push`.
--
-- MUST be split into THREE transactions because PostgreSQL does
-- not allow a newly-added enum value to be used in the same
-- transaction it was added in.
-- ============================================================

-- ---------- Migration 1: verification_status on sightings ----------
BEGIN;

CREATE TYPE verification_status AS ENUM ('Pending', 'Verified', 'Dismissed');

ALTER TABLE sightings
  ADD COLUMN verification_status verification_status NOT NULL DEFAULT 'Pending';

-- Partial index — admin timeline + F1 ranking only ever read Verified rows.
CREATE INDEX sightings_pet_verified_created_idx
  ON sightings (initial_target_pet_id, verification_status, created_at)
  WHERE verification_status = 'Verified';

COMMIT;


-- ---------- Migration 2: add 'Resolved' to pet_status ----------
BEGIN;

ALTER TYPE pet_status ADD VALUE IF NOT EXISTS 'Resolved';

COMMIT;


-- ---------- Migration 3: score_awards table + RPCs ----------
BEGIN;

-- Audit trail. One row per (missing_pet_id, user_id) thanks to the UNIQUE
-- constraint — that is what enforces "each hunter awarded at most once per
-- resolution" (spam protection rule from the product brief).
CREATE TABLE score_awards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE NOT NULL,
    missing_pet_id UUID REFERENCES missing_pets(id) ON DELETE CASCADE NOT NULL,
    sighting_id UUID REFERENCES sightings(id) ON DELETE SET NULL,
    points INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    awarded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (missing_pet_id, user_id)
);

CREATE INDEX score_awards_user_idx ON score_awards (user_id, awarded_at DESC);

-- Atomic resolution. Single transaction so a failure mid-way leaves no
-- half-paid bounty or half-credited score.
CREATE OR REPLACE FUNCTION resolve_missing_pet(
    p_pet_id            UUID,
    p_final_sighting_id UUID,
    p_slip_image_url    TEXT,
    p_reference_no      VARCHAR,
    p_verified_by       UUID
) RETURNS JSONB
LANGUAGE plpgsql AS $$
DECLARE
    v_pet           missing_pets%ROWTYPE;
    v_final_hunter  UUID;
    v_award_points  INT[] := ARRAY[25, 15, 10];   -- 4th+ falls through to 5
    v_clue          RECORD;
    v_rank          INT := 0;
    v_pts           INT;
    v_awards        JSONB := '[]'::JSONB;
BEGIN
    SELECT * INTO v_pet FROM missing_pets WHERE id = p_pet_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Pet % not found', p_pet_id;
    END IF;
    IF v_pet.status = 'Resolved' THEN
        RAISE EXCEPTION 'Pet % already resolved', p_pet_id;
    END IF;

    -- "Associated with this pet" = explicitly targeted OR AI-matched.
    SELECT s.hunter_id INTO v_final_hunter
      FROM sightings s
     WHERE s.id = p_final_sighting_id
       AND s.action_type = 'Caught'
       AND s.verification_status = 'Verified'
       AND (s.initial_target_pet_id = p_pet_id
            OR EXISTS (SELECT 1 FROM sighting_matches sm
                        WHERE sm.sighting_id = s.id
                          AND sm.missing_pet_id = p_pet_id));
    IF v_final_hunter IS NULL THEN
        RAISE EXCEPTION
          'Final sighting % is not a verified Caught sighting for pet %',
          p_final_sighting_id, p_pet_id;
    END IF;

    -- 1. Bounty money → final hunter (one bounty_transactions row).
    INSERT INTO bounty_transactions
        (sighting_id, missing_pet_id, owner_id, amount, slip_image_url,
         reference_no, transfer_datetime, status, verified_at, verified_by)
    VALUES
        (p_final_sighting_id, p_pet_id, v_pet.owner_id, v_pet.bounty_amount,
         p_slip_image_url, p_reference_no, NOW(), 'Verified', NOW(),
         p_verified_by);

    -- 2. F1 clue scoring. DISTINCT ON dedupes by hunter (earliest verified
    --    sighting wins); the final hunter is excluded entirely.
    FOR v_clue IN
        SELECT DISTINCT ON (s.hunter_id)
               s.hunter_id, s.id AS sighting_id, s.created_at
          FROM sightings s
         WHERE s.verification_status = 'Verified'
           AND s.hunter_id IS NOT NULL
           AND s.hunter_id <> v_final_hunter
           AND (s.initial_target_pet_id = p_pet_id
                OR EXISTS (SELECT 1 FROM sighting_matches sm
                            WHERE sm.sighting_id = s.id
                              AND sm.missing_pet_id = p_pet_id))
         ORDER BY s.hunter_id, s.created_at ASC
    LOOP
        v_rank := v_rank + 1;
        v_pts  := COALESCE(v_award_points[v_rank], 5);

        INSERT INTO score_awards
            (user_id, missing_pet_id, sighting_id, points, rank)
        VALUES
            (v_clue.hunter_id, p_pet_id, v_clue.sighting_id, v_pts, v_rank);

        UPDATE users SET total_score = total_score + v_pts
         WHERE id = v_clue.hunter_id;

        v_awards := v_awards || jsonb_build_object(
            'user_id',     v_clue.hunter_id,
            'sighting_id', v_clue.sighting_id,
            'rank',        v_rank,
            'points',      v_pts
        );
    END LOOP;

    -- 3. Mark pet resolved.
    UPDATE missing_pets SET status = 'Resolved' WHERE id = p_pet_id;

    RETURN jsonb_build_object(
        'pet_id',          p_pet_id,
        'final_hunter_id', v_final_hunter,
        'bounty_amount',   v_pet.bounty_amount,
        'awards',          v_awards
    );
END;
$$;

-- Owner-facing union: every sighting "against" this pet = AI-matched OR
-- explicitly targeted. Hunter display_name + similarity_score joined in so
-- the owner UI can render a single list.
CREATE OR REPLACE FUNCTION sightings_for_pet(
    p_pet_id            UUID,
    p_limit             INT     DEFAULT 50,
    p_offset            INT     DEFAULT 0,
    p_include_dismissed BOOLEAN DEFAULT FALSE
) RETURNS TABLE (
    id                   UUID,
    hunter_id            UUID,
    hunter_display_name  VARCHAR,
    image_url            TEXT,
    detected_species     pet_species,
    action_type          action_type,
    sighting_status      sighting_status,
    verification_status  verification_status,
    sighted_location     TEXT,
    created_at           TIMESTAMP WITH TIME ZONE,
    similarity_score     DECIMAL,
    match_source         TEXT       -- 'matched' | 'targeted' | 'both'
) LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    WITH matched AS (
        SELECT sm.sighting_id, MAX(sm.similarity_score) AS sim
          FROM sighting_matches sm
         WHERE sm.missing_pet_id = p_pet_id
         GROUP BY sm.sighting_id
    ),
    targeted AS (
        SELECT s.id AS sighting_id
          FROM sightings s
         WHERE s.initial_target_pet_id = p_pet_id
    ),
    combined AS (
        SELECT sighting_id, sim, 'matched'::TEXT AS src FROM matched
        UNION ALL
        SELECT sighting_id, NULL::DECIMAL,         'targeted'      FROM targeted
    ),
    deduped AS (
        SELECT sighting_id,
               MAX(sim) AS sim,
               CASE WHEN COUNT(DISTINCT src) > 1 THEN 'both'
                    ELSE MAX(src) END AS src
          FROM combined
         GROUP BY sighting_id
    )
    SELECT s.id, s.hunter_id, u.display_name,
           s.image_url, s.detected_species, s.action_type,
           s.sighting_status, s.verification_status,
           ST_AsText(s.sighted_location::geometry),
           s.created_at,
           d.sim, d.src
      FROM deduped d
      JOIN sightings s ON s.id = d.sighting_id
      LEFT JOIN users u ON u.id = s.hunter_id
     WHERE p_include_dismissed OR s.verification_status <> 'Dismissed'
     ORDER BY s.created_at DESC
     LIMIT p_limit OFFSET p_offset;
END;
$$;

COMMIT;
