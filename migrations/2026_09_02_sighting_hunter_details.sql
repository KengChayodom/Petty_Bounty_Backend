-- ============================================================
-- 2026-09-02 · sightings_for_pet — return the hunter's contact details
--
-- The Status Tracker card shows "who reported this": an avatar, a name, and a
-- phone line. Only the name was ever real. The avatar was a hardcoded grey
-- person icon and the phone row never rendered at all, because the RPC that
-- feeds the card selects `u.display_name` and nothing else off `users`.
--
-- This is the same gap `2026_09_02_pet_owner_details.sql` closed on the other
-- side of the case: the hunter reading a pet's detail page gets the owner's
-- name, phone and photo so they can make contact. The owner reading their own
-- sighting timeline is the party who actually needs to ring somebody — the
-- person holding their pet — and had no number to ring.
--
-- Both new columns are LEFT JOIN reads of the hunter's own profile, so they
-- are NULL for a deleted account or a profile that never set them; the card
-- renders the avatar placeholder / omits the phone line rather than inventing
-- a value.
--
-- Scope note: this widens what the owner sees, not who may see it. The
-- endpoint (`GET /missing-pets/{id}/sightings`) is already authenticated and
-- already discloses who spotted the pet and where.
--
-- DROP first: the return type gains two columns, and CREATE OR REPLACE cannot
-- change a function's return type.
-- ============================================================

BEGIN;

DROP FUNCTION IF EXISTS sightings_for_pet(UUID, INT, INT, BOOLEAN);

CREATE FUNCTION sightings_for_pet(
    p_pet_id            UUID,
    p_limit             INT     DEFAULT 50,
    p_offset            INT     DEFAULT 0,
    p_include_dismissed BOOLEAN DEFAULT FALSE
) RETURNS TABLE (
    id                       UUID,
    hunter_id                UUID,
    hunter_display_name      VARCHAR,
    hunter_phone             VARCHAR,
    hunter_profile_image_url TEXT,
    image_url                TEXT,
    detected_species         pet_species,
    action_type              action_type,
    sighting_status          sighting_status,
    verification_status      verification_status,
    owner_status             owner_decision,
    sighted_location         TEXT,
    created_at               TIMESTAMP WITH TIME ZONE,
    similarity_score         DECIMAL,
    match_source             TEXT
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
    SELECT s.id, s.hunter_id, u.display_name, u.phone, u.profile_image_url,
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

COMMIT;
