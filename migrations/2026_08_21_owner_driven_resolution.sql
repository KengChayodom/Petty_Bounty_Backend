-- ============================================================
-- Feature: owner-driven resolution — the owner decides, the admin moderates
-- Apply via Supabase SQL Editor or `supabase db push`.
--
-- WHAT CHANGES, AND WHY
--
-- Until now `verification_status = 'Verified'`, written only by an
-- administrator, was the gate on BOTH the bounty payout and the F1 clue
-- scores. That put a party with no knowledge of the animal in charge of the
-- one question only its owner can answer ("is that my pet?"), and it left the
-- owner's own verdict (`sighting_matches.owner_status`) decorative: the owner
-- could confirm a match and nothing whatsoever followed from it.
--
-- The split is now:
--   * OWNER  — decides every sighting (Confirmed / Rejected), and closes the
--              search by confirming a 'Caught' sighting. That single act ends
--              the search (pet_status -> 'Found') AND distributes every clue
--              score for the case. Scores are no longer an administrator's to
--              give.
--   * ADMIN  — moderation (spam flags -> 'Dismissed') and the bounty transfer
--              afterwards (pet_status 'Found' -> 'Resolved'). Nothing else.
--
-- SCORING RULES (decided 2026-08-21, all of them live in `owner_decide_sighting`)
--   * Only `owner_status = 'Confirmed'` earns. Pending and Rejected earn nothing.
--   * A sighting an administrator Dismissed never earns, even if the owner had
--     confirmed it before the flag was upheld.
--   * Ranked NEWEST first — the last clue is the one that led to the animal.
--   * One hunter, one seat, taken at their newest confirmed sighting.
--   * 25 / 15 / 10 / 5 / 5, capped at five hunters. There was previously no cap
--     at all: rank 4 and beyond each earned 5 points without limit.
--   * The hunter who caught the animal ranks FIRST and earns 25 on top of the
--     bounty. The old function excluded them from scoring entirely.
--
-- SEQUENTIAL DECISIONS
--   The owner rules on one card at a time, oldest first. A card with an older
--   undecided card beneath it is refused (the API answers 409). This is what
--   stops an owner from skipping straight to the rescue card and leaving every
--   hunter who helped along the way unpaid, and it settles two rival 'Caught'
--   claims without any tie-break logic: the older claim is offered first, and
--   rejecting it advances to the newer one.
--
--   A sighting an administrator has Dismissed is NOT part of the queue. One
--   piece of spam would otherwise block the owner's queue permanently.
--
-- POSTS EXPIRE AFTER 7 DAYS
--   Enforced by a predicate on the two read paths, not by a scheduled job:
--   this database has no pg_cron, and a stored "expired" state would have to be
--   recomputed everywhere a post can change. Past 7 days a post stops matching
--   new sightings and leaves the map; the row is untouched, so its owner can
--   still work the queue it already collected and still close it.
--
-- Sections:
--   1. backfill sighting_matches for targeted sightings (they had none)
--   2. sightings_for_pet      — expose owner_status; keep match_source honest
--   3. owner_decide_sighting  — NEW: the whole owner-side decision + payout
--   4. resolve_missing_pet    — bounty only; the clue loop moves to (3)
--   5. match_missing_pets     — 7-day expiry
--   6. get_nearby_missing_pets— 7-day expiry
-- ============================================================

BEGIN;

-- ---------- 1. backfill: give targeted sightings a queue row ----------
-- A targeted report (SRS-50: "I am looking at this pet's page and I see it")
-- carries `initial_target_pet_id` and no `sighting_matches` row, because it
-- skips the AI match entirely. That was harmless while the owner's verdict did
-- nothing. Now the verdict IS the currency: with no row there is nothing to
-- write a verdict to, so `decide_match` answers 404 and the hunter can never be
-- paid — and worse, the card sits in the queue undecidable, blocking every card
-- above it.
--
-- similarity_score stays NULL: no vector was ever computed for these. Section 2
-- relies on exactly that to keep calling them 'targeted'.
INSERT INTO sighting_matches (sighting_id, missing_pet_id, similarity_score, owner_status)
SELECT s.id, s.initial_target_pet_id, NULL, 'Pending'
  FROM sightings s
 WHERE s.initial_target_pet_id IS NOT NULL
ON CONFLICT (sighting_id, missing_pet_id) DO NOTHING;

-- ---------- 2. sightings_for_pet ----------
-- Two changes:
--   (a) returns `owner_status`. Without it no screen can draw the queue: which
--       card is decided, which is next, which is still locked. The owner's own
--       verdict was the one column the owner's own timeline did not return.
--   (b) the `matched` CTE now ignores NULL-similarity rows. Section 1 gave
--       targeted sightings a match row, which would otherwise flip their
--       `match_source` from 'targeted' to 'both' across the board. A NULL
--       similarity is precisely "this row is not an AI match".
--
-- DROP first: the return type gains a column, and CREATE OR REPLACE cannot
-- change a function's return type.
DROP FUNCTION IF EXISTS sightings_for_pet(UUID, INT, INT, BOOLEAN);

CREATE FUNCTION sightings_for_pet(
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
    owner_status         owner_decision,
    sighted_location     TEXT,
    created_at           TIMESTAMP WITH TIME ZONE,
    similarity_score     DECIMAL,
    match_source         TEXT
) LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    WITH matched AS (
        SELECT sm.sighting_id, MAX(sm.similarity_score) AS sim
          FROM sighting_matches sm
         WHERE sm.missing_pet_id = p_pet_id
           AND sm.similarity_score IS NOT NULL
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
               CASE
                 WHEN BOOL_OR(src = 'matched')
                  AND BOOL_OR(src = 'targeted') THEN 'both'
                 WHEN BOOL_OR(src = 'matched')  THEN 'matched'
                 ELSE 'targeted'
               END AS src
          FROM combined
         GROUP BY sighting_id
    )
    SELECT s.id, s.hunter_id, u.display_name,
           s.image_url, s.detected_species, s.action_type,
           s.sighting_status, s.verification_status,
           COALESCE(sm.owner_status, 'Pending'::owner_decision),
           ST_AsText(s.sighted_location::geometry),
           s.created_at,
           d.sim, d.src
      FROM deduped d
      JOIN sightings s ON s.id = d.sighting_id
      LEFT JOIN users u ON u.id = s.hunter_id
      LEFT JOIN sighting_matches sm
             ON sm.sighting_id = s.id AND sm.missing_pet_id = p_pet_id
     WHERE p_include_dismissed OR s.verification_status <> 'Dismissed'
     ORDER BY s.created_at DESC
     LIMIT p_limit OFFSET p_offset;
END;
$$;

-- ---------- 3. owner_decide_sighting ----------
-- The owner's verdict on one card, and — when that card is a confirmed
-- 'Caught' — the end of the search and the whole scoring payout, in ONE
-- transaction. Splitting it would allow a failure that closes a search without
-- paying anybody, or pays some hunters and not others, neither of which can be
-- repaired afterwards: the awards are not reconstructable once the search is
-- closed.
--
-- Every RAISE message here is load-bearing: `SightingService.decide_match`
-- matches on these prefixes to choose 404 / 409 / 400. Change one and change
-- it there too.
CREATE OR REPLACE FUNCTION owner_decide_sighting(
    p_pet_id      UUID,
    p_sighting_id UUID,
    p_owner_id    UUID,
    p_decision    TEXT
) RETURNS JSONB
LANGUAGE plpgsql AS $$
DECLARE
    v_pet          missing_pets%ROWTYPE;
    v_sighting     sightings%ROWTYPE;
    v_owner_status owner_decision;
    v_blockers     INT;
    v_closed       BOOLEAN := FALSE;
    v_award_points INT[] := ARRAY[25, 15, 10, 5, 5];
    v_clue         RECORD;
    v_rank         INT := 0;
    v_award_id     UUID;
    v_awards       JSONB := '[]'::JSONB;
BEGIN
    IF p_decision NOT IN ('Confirmed', 'Rejected') THEN
        RAISE EXCEPTION 'decision must be Confirmed or Rejected (got %)', p_decision;
    END IF;

    -- FOR UPDATE serialises two tabs (or two devices) deciding at once. Without
    -- it both could pass the "search still open" check below and both run the
    -- payout.
    SELECT * INTO v_pet FROM missing_pets WHERE id = p_pet_id FOR UPDATE;

    -- 404 for both "no such pet" and "not yours": a caller who does not own the
    -- report must not be able to tell which it was.
    IF NOT FOUND OR v_pet.owner_id IS DISTINCT FROM p_owner_id THEN
        RAISE EXCEPTION 'Missing pet % not found or not owned by you', p_pet_id;
    END IF;

    IF LOWER(v_pet.status::TEXT) IN ('found', 'resolved') THEN
        RAISE EXCEPTION 'Search for pet % is already closed', p_pet_id;
    END IF;

    SELECT * INTO v_sighting FROM sightings WHERE id = p_sighting_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Sighting % is not a match for pet %', p_sighting_id, p_pet_id;
    END IF;

    -- A Dismissed sighting has been withdrawn by moderation. It is not in the
    -- queue, so it can neither be decided nor block anything.
    IF v_sighting.verification_status = 'Dismissed' THEN
        RAISE EXCEPTION 'Sighting % is not a match for pet %', p_sighting_id, p_pet_id;
    END IF;

    SELECT sm.owner_status INTO v_owner_status
      FROM sighting_matches sm
     WHERE sm.sighting_id = p_sighting_id
       AND sm.missing_pet_id = p_pet_id
     FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Sighting % is not a match for pet %', p_sighting_id, p_pet_id;
    END IF;

    IF v_owner_status <> 'Pending' THEN
        RAISE EXCEPTION 'Sighting % has already been decided (%)',
              p_sighting_id, v_owner_status;
    END IF;

    -- Sequential guard. (created_at, id) rather than created_at alone so two
    -- sightings sharing a timestamp still have exactly one correct order and
    -- the queue cannot deadlock on a tie.
    SELECT COUNT(*) INTO v_blockers
      FROM sighting_matches sm
      JOIN sightings s ON s.id = sm.sighting_id
     WHERE sm.missing_pet_id = p_pet_id
       AND sm.owner_status = 'Pending'
       AND s.verification_status <> 'Dismissed'
       AND (s.created_at, s.id) < (v_sighting.created_at, v_sighting.id);

    IF v_blockers > 0 THEN
        RAISE EXCEPTION
          'Sighting % is out of order: % earlier sighting(s) are still undecided',
          p_sighting_id, v_blockers;
    END IF;

    UPDATE sighting_matches
       SET owner_status = p_decision::owner_decision
     WHERE sighting_id = p_sighting_id
       AND missing_pet_id = p_pet_id;

    IF p_decision = 'Confirmed' THEN
        -- Mirrors the pre-existing behaviour of decide_match: a confirmed match
        -- advances the sighting's own lifecycle column too.
        UPDATE sightings
           SET sighting_status = 'Confirmed'
         WHERE id = p_sighting_id
           AND sighting_status <> 'Closed';

        IF v_sighting.action_type = 'Caught' THEN
            v_closed := TRUE;

            UPDATE missing_pets SET status = 'Found' WHERE id = p_pet_id;

            FOR v_clue IN
                SELECT picks.hunter_id, picks.sighting_id
                  FROM (
                        SELECT DISTINCT ON (s.hunter_id)
                               s.hunter_id,
                               s.id AS sighting_id,
                               s.created_at
                          FROM sighting_matches sm
                          JOIN sightings s ON s.id = sm.sighting_id
                         WHERE sm.missing_pet_id = p_pet_id
                           AND sm.owner_status = 'Confirmed'
                           AND s.verification_status <> 'Dismissed'
                           AND s.hunter_id IS NOT NULL
                         -- one seat per hunter, taken at their NEWEST card
                         ORDER BY s.hunter_id, s.created_at DESC, s.id DESC
                       ) picks
                 ORDER BY picks.created_at DESC, picks.sighting_id DESC
                 LIMIT array_length(v_award_points, 1)
            LOOP
                v_rank := v_rank + 1;

                -- ON CONFLICT: score_awards is UNIQUE (missing_pet_id, user_id).
                -- Nothing should reach here twice (the closed-search check above
                -- is the real guard), but a payout that raises would roll back
                -- the owner's verdict as well, and losing the verdict is worse
                -- than skipping a duplicate award.
                INSERT INTO score_awards
                    (user_id, missing_pet_id, sighting_id, points, rank)
                VALUES
                    (v_clue.hunter_id, p_pet_id, v_clue.sighting_id,
                     v_award_points[v_rank], v_rank)
                ON CONFLICT (missing_pet_id, user_id) DO NOTHING
                RETURNING id INTO v_award_id;

                CONTINUE WHEN v_award_id IS NULL;

                UPDATE users
                   SET total_score = COALESCE(total_score, 0) + v_award_points[v_rank]
                 WHERE id = v_clue.hunter_id;

                v_awards := v_awards || jsonb_build_object(
                    'user_id',     v_clue.hunter_id,
                    'sighting_id', v_clue.sighting_id,
                    'rank',        v_rank,
                    'points',      v_award_points[v_rank]
                );
            END LOOP;

            -- The search is over, so every entry on this pet's timeline stops
            -- being a live lead. Same rule the owner's End Search button
            -- applies (close_sightings_for_pet), expressed here in one
            -- statement because both sources are reachable from SQL.
            UPDATE sightings s
               SET sighting_status = 'Closed'
             WHERE s.id <> p_sighting_id
               AND (s.initial_target_pet_id = p_pet_id
                    OR EXISTS (SELECT 1 FROM sighting_matches sm
                                WHERE sm.sighting_id = s.id
                                  AND sm.missing_pet_id = p_pet_id));
        END IF;
    END IF;

    RETURN jsonb_build_object(
        'pet_id',         p_pet_id,
        'sighting_id',    p_sighting_id,
        'owner_status',   p_decision,
        'search_closed',  v_closed,
        'pet_status',     CASE WHEN v_closed THEN 'Found'
                               ELSE v_pet.status::TEXT END,
        'awards',         v_awards
    );
END;
$$;

-- ---------- 4. resolve_missing_pet — bounty only ----------
-- The clue loop is GONE from this function; it now lives in
-- `owner_decide_sighting` (section 3) and runs when the owner confirms the
-- rescue, days before the money moves. Leaving a copy here would award every
-- hunter a second time when the administrator settles the payment.
--
-- The eligibility test changes with it. It used to require
-- `verification_status = 'Verified'`, which no administrator writes any more —
-- left as it was, no case could ever be paid. A sighting now qualifies to be
-- paid because its OWNER confirmed it as the catch, which is also what closed
-- the search, so the payment can only follow a resolution the owner already
-- made.
CREATE OR REPLACE FUNCTION resolve_missing_pet(
    p_pet_id            UUID,
    p_final_sighting_id UUID,
    p_slip_image_url    TEXT,
    p_reference_no      VARCHAR,
    p_verified_by       UUID
) RETURNS JSONB
LANGUAGE plpgsql AS $$
DECLARE
    v_pet          missing_pets%ROWTYPE;
    v_final_hunter UUID;
BEGIN
    SELECT * INTO v_pet FROM missing_pets WHERE id = p_pet_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Pet % not found', p_pet_id;
    END IF;
    IF v_pet.status = 'Resolved' THEN
        RAISE EXCEPTION 'Pet % already resolved', p_pet_id;
    END IF;
    IF v_pet.status <> 'Found' THEN
        RAISE EXCEPTION
          'Pet % has not been recovered yet (status %) — the owner closes the '
          'search before the bounty is paid', p_pet_id, v_pet.status;
    END IF;

    -- "Associated with this pet" = explicitly targeted OR AI-matched; either
    -- way the owner has to have confirmed THIS sighting as the catch.
    SELECT s.hunter_id INTO v_final_hunter
      FROM sightings s
      JOIN sighting_matches sm
        ON sm.sighting_id = s.id
       AND sm.missing_pet_id = p_pet_id
     WHERE s.id = p_final_sighting_id
       AND s.action_type = 'Caught'
       AND sm.owner_status = 'Confirmed';

    IF v_final_hunter IS NULL THEN
        RAISE EXCEPTION
          'Final sighting % is not a confirmed Caught sighting for pet %',
          p_final_sighting_id, p_pet_id;
    END IF;

    INSERT INTO bounty_transactions
        (sighting_id, missing_pet_id, owner_id, amount, slip_image_url,
         reference_no, transfer_datetime, status, verified_at, verified_by)
    VALUES
        (p_final_sighting_id, p_pet_id, v_pet.owner_id, v_pet.bounty_amount,
         p_slip_image_url, p_reference_no, NOW(), 'Verified', NOW(),
         p_verified_by);

    UPDATE missing_pets SET status = 'Resolved' WHERE id = p_pet_id;

    RETURN jsonb_build_object(
        'pet_id',          p_pet_id,
        'final_hunter_id', v_final_hunter,
        'bounty_amount',   v_pet.bounty_amount,
        -- Kept, always empty: scoring happened at rescue time. Callers and
        -- tests read this key, and an absent key would read as "unknown"
        -- rather than "none awarded here".
        'awards',          '[]'::JSONB
    );
END;
$$;

-- ---------- 5. match_missing_pets — 7-day expiry ----------
-- Identical to the deployed function apart from the age predicate. A post older
-- than a week stops collecting new sightings; the ones it already collected are
-- untouched and its owner can still work the queue and close it.
CREATE OR REPLACE FUNCTION match_missing_pets(
    p_sighting_id UUID,
    match_limit   INTEGER
) RETURNS TABLE (
    id                UUID,
    pet_name          TEXT,
    species           TEXT,
    characteristics   JSONB,
    bounty_amount     NUMERIC,
    last_seen_location TEXT,
    last_seen_time    TEXT,
    image_url         TEXT,
    similarity        DOUBLE PRECISION,
    distance_meters   DOUBLE PRECISION,
    status            TEXT,
    primary_color_hex TEXT
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
      AND mp.created_at > NOW() - INTERVAL '7 days'
      AND ST_DWithin(mp.last_seen_location, v_location, 10000)
    ORDER BY mp.feature_vector <=> v_embedding ASC NULLS LAST
    LIMIT match_limit;
END;
$$;

-- ---------- 6. get_nearby_missing_pets — 7-day expiry ----------
-- The map half of the same rule: an expired post stops being a pin.
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
      AND mp.created_at > NOW() - INTERVAL '7 days'
      AND ST_DWithin(
          mp.last_seen_location::geography,
          ST_GeomFromText(center_location, 4326)::geography,
          radius_meters
      )
    ORDER BY distance_meters
    LIMIT "limit";
END;
$$;

COMMIT;
