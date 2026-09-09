-- 2026-09-09 — `users.display_name` becomes `users.username`.
--
-- The application has called this value the USERNAME since Feature 4 was
-- specified (SRS-73, and the sign-up validator's "Enter a username"). Only the
-- column kept the older name, which left one value carrying two names and every
-- document explaining the discrepancy instead of the value.
--
-- The column was renamed directly in Supabase. This file makes that change
-- reproducible and, more importantly, repairs the five functions the rename
-- broke: PostgreSQL stores a function body as text and does NOT rewrite it when
-- a column is renamed, so all five kept referring to a column that no longer
-- existed and failed at call time. `handle_new_user` is the worst of them — it
-- is the AFTER INSERT trigger on auth.users, so every sign-up was failing.
--
-- SCOPE: every layer. The column, the two OUT column names the RPCs expose
-- (`owner_display_name` -> `owner_username`, `hunter_display_name` ->
-- `hunter_username`), the two `json_build_object` keys, and the sign-up
-- metadata key. Nothing in the product answers to `display_name` afterwards.
--
-- Two functions therefore have to be DROPped rather than replaced: PostgreSQL
-- treats the OUT column names of a RETURNS TABLE as part of the return type, so
-- `CREATE OR REPLACE` refuses to rename one. They are dropped and recreated
-- inside this transaction, and no view or policy depends on either (checked
-- against pg_views, pg_indexes and pg_policies before writing this).
--
-- THIS IS A BREAKING API CHANGE. Every payload key that carried a name is now
-- `username`, `owner_username` or `hunter_username`, so the backend, the Flutter
-- app and the admin console must ship together with this migration.
--
-- The sign-up metadata key moves to `username` as well, but `handle_new_user`
-- still reads `display_name` as a fallback. A client build that has not shipped
-- yet keeps working instead of silently registering people under their email
-- prefix, and the fallback can be deleted once no such build is in the field.

BEGIN;

-- Idempotent: the rename was applied by hand before this file existed.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name   = 'users'
           AND column_name  = 'display_name'
    ) THEN
        ALTER TABLE public.users RENAME COLUMN display_name TO username;
    END IF;
END $$;

-- ---------- 1. handle_new_user — the sign-up trigger ----------
-- Prefers the `username` metadata key and falls back to `display_name` so a
-- client build from before the rename still registers people under their name.
CREATE OR REPLACE FUNCTION public.handle_new_user()
 RETURNS trigger
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public'
AS $function$
begin
  insert into public.users (id, username, phone)
  values (
    new.id,
    coalesce(
      nullif(new.raw_user_meta_data->>'username', ''),
      nullif(new.raw_user_meta_data->>'display_name', ''),   -- pre-rename clients
      split_part(new.email, '@', 1)
    ),
    nullif(new.raw_user_meta_data->>'phone', '')
  )
  on conflict (id) do nothing;
  return new;
end;
$function$;

-- ---------- 2. find_user_by_email (MD-57) ----------
-- The returned JSON key becomes 'username'; the admin console's AccountLookup
-- type changes with it.
CREATE OR REPLACE FUNCTION public.find_user_by_email(p_email text)
 RETURNS json
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'auth'
AS $function$
DECLARE
    v_row RECORD;
BEGIN
    SELECT u.id, u.username, u.role
      INTO v_row
      FROM auth.users AS au
      JOIN public.users AS u ON u.id = au.id
     WHERE LOWER(au.email) = LOWER(TRIM(p_email))
     LIMIT 1;

    IF NOT FOUND THEN
        RETURN NULL;
    END IF;

    RETURN json_build_object(
        'id',           v_row.id,
        'username',     v_row.username,
        'role',         v_row.role
    );
END;
$function$;

-- ---------- 3. assign_user_role (MD-58) ----------
CREATE OR REPLACE FUNCTION public.assign_user_role(p_target_user_id uuid, p_role user_role, p_changed_by uuid)
 RETURNS json
 LANGUAGE plpgsql
AS $function$
DECLARE
    v_before   user_role;
    v_name     TEXT;
    v_admins   INT;
    v_change_id UUID;
BEGIN
    SELECT role, username INTO v_before, v_name
      FROM users WHERE id = p_target_user_id FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'user % not found', p_target_user_id;
    END IF;

    IF v_before = p_role THEN
        RETURN json_build_object(
            'changed',      FALSE,
            'id',           p_target_user_id,
            'username',     v_name,
            'role_before',  v_before,
            'role_after',   v_before
        );
    END IF;

    -- Guard 1 — an administrator may not withdraw their own access. Asking a
    -- second administrator to do it keeps the action reviewable by someone
    -- other than the person taking it.
    IF p_target_user_id = p_changed_by AND p_role <> 'admin' THEN
        RAISE EXCEPTION 'cannot withdraw your own administrator access';
    END IF;

    -- Guard 2 — never leave the platform with no administrator. The subquery
    -- takes FOR UPDATE on every admin row, so a concurrent demotion waits here
    -- and then re-counts against the committed state.
    IF v_before = 'admin' THEN
        SELECT COUNT(*) INTO v_admins
          FROM (SELECT 1 FROM users WHERE role = 'admin' FOR UPDATE) AS locked;
        IF v_admins <= 1 THEN
            RAISE EXCEPTION 'cannot remove the last administrator';
        END IF;
    END IF;

    UPDATE users SET role = p_role WHERE id = p_target_user_id;

    INSERT INTO role_changes
        (target_user_id, changed_by, role_before, role_after)
    VALUES
        (p_target_user_id, p_changed_by, v_before, p_role)
    RETURNING id INTO v_change_id;

    RETURN json_build_object(
        'changed',        TRUE,
        'role_change_id', v_change_id,
        'id',             p_target_user_id,
        'username',       v_name,
        'role_before',    v_before,
        'role_after',     p_role
    );
END;
$function$;

-- ---------- 4. get_missing_pet_by_id ----------
-- The OUT column is renamed, which is a return-type change, so this one is
-- dropped and recreated rather than replaced.
DROP FUNCTION IF EXISTS public.get_missing_pet_by_id(uuid);
CREATE FUNCTION public.get_missing_pet_by_id(p_pet_id uuid)
 RETURNS TABLE(id uuid, owner_id uuid, pet_name character varying, species character varying, characteristics jsonb, bounty_amount numeric, latitude double precision, longitude double precision, last_seen_time timestamp with time zone, image_url text, status character varying, created_at timestamp with time zone, expires_at timestamp with time zone, primary_color_hex character varying, owner_username character varying, owner_phone character varying, owner_profile_image_url text)
 LANGUAGE plpgsql
 STABLE
AS $function$
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
        u.username,
        u.phone,
        u.profile_image_url
    FROM public.missing_pets mp
    LEFT JOIN public.users u ON u.id = mp.owner_id
    WHERE mp.id = p_pet_id;
END;
$function$;

-- ---------- 5. sightings_for_pet ----------
-- Same return-type change, same drop-and-recreate.
DROP FUNCTION IF EXISTS public.sightings_for_pet(uuid, integer, integer, boolean);
CREATE FUNCTION public.sightings_for_pet(p_pet_id uuid, p_limit integer DEFAULT 50, p_offset integer DEFAULT 0, p_include_dismissed boolean DEFAULT false)
 RETURNS TABLE(id uuid, hunter_id uuid, hunter_username character varying, hunter_phone character varying, hunter_profile_image_url text, image_url text, detected_species pet_species, action_type action_type, sighting_status sighting_status, verification_status verification_status, owner_status owner_decision, sighted_location text, created_at timestamp with time zone, similarity_score numeric, match_source text)
 LANGUAGE plpgsql
AS $function$
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
    SELECT s.id, s.hunter_id, u.username, u.phone, u.profile_image_url,
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
$function$;

COMMIT;
