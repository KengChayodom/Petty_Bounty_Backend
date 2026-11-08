-- 2026-09-30 — Close the direct-to-Supabase routes around FastAPI.
--
-- STATUS: PROPOSED. NOT APPLIED to any database. Read it, then apply it yourself.
--
-- WHY. The app carries the project's anon key, so anyone can call the Supabase REST
-- API directly and skip FastAPI, where every permission check lives. The backend uses
-- the service role, which bypasses RLS, so RLS is the only thing that stands between
-- the API roles (anon, authenticated) and the data. Read from the live catalog on
-- 2026-09-30:
--   * users has the policy "Users can update own profile" and the API roles hold UPDATE
--     on every column, so a signed-in user could set their OWN `role` to 'admin' and
--     their OWN `total_score` to anything. Nothing else (no trigger) stops it.
--   * missing_pets has INSERT and UPDATE policies for the owner, so a signed-in user
--     could create or edit a report directly, skipping validation and the rule that
--     only the administrator's settlement may write 'Resolved'.
--   * every RPC is executable by anon and authenticated. Most are neutralised by RLS,
--     but find_user_by_email is SECURITY DEFINER and answers anyone, signed in or not,
--     with the account's id, username and role for any email address.
--   * spatial_ref_sys has no RLS and the API roles can write it.
--
-- WHAT THIS ASSUMES. Neither the Flutter app nor the admin console writes a table or
-- calls an RPC directly. Both are true today: they use Supabase for Auth and Storage
-- only (the golden rule), and the backend, which uses the service role, does the rest.
-- If that ever changes, grant back exactly what is needed, never table-wide.
--
-- READS ARE NOT TOUCHED. RLS already limits them (users: own row; missing_pets: rows
-- with status 'Searching', which is what /nearby returns anyway).
--
-- ROLLBACK (if something turns out to need direct access):
--   GRANT INSERT, UPDATE, DELETE ON public.<table> TO authenticated;
--   GRANT EXECUTE ON FUNCTION public.<fn>(<args>) TO authenticated;
--
-- Storage is in its own file: 2026_09_30_lock_down_storage_uploads.sql.

BEGIN;

-- 1. The API roles write nothing directly. FastAPI writes with the service role.
DO $$
DECLARE t record;
BEGIN
    FOR t IN
        SELECT c.oid::regclass AS tbl
          FROM pg_class c
         WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p')
    LOOP
        BEGIN
            EXECUTE format(
                'REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON %s FROM anon, authenticated',
                t.tbl);
        EXCEPTION WHEN insufficient_privilege THEN
            -- e.g. spatial_ref_sys, owned by an extension role: revoke it as its owner.
            RAISE NOTICE 'skipped % (not owned by this role)', t.tbl;
        END;
    END LOOP;
END $$;

-- A table created later must not quietly hand the API roles write access again.
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
    REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON TABLES FROM anon, authenticated;

-- 2. The policies that allowed those writes are now unreachable. Dropping them makes the
--    intent explicit and stops a future GRANT from silently reviving them.
DROP POLICY IF EXISTS "Users can update own profile"           ON public.users;
DROP POLICY IF EXISTS "Authenticated users can insert pets"    ON public.missing_pets;
DROP POLICY IF EXISTS "Owners can update their own pets"       ON public.missing_pets;

-- 3. The RPCs are the backend's. Only the service role may call them.
--    handle_new_user is a trigger function on auth.users and is left alone.
DO $$
DECLARE f record;
BEGIN
    FOR f IN
        SELECT p.oid::regprocedure AS sig
          FROM pg_proc p
         WHERE p.pronamespace = 'public'::regnamespace
           AND p.prokind = 'f'
           AND p.proname IN (
               'match_missing_pets', 'get_nearby_missing_pets', 'get_missing_pet_by_id',
               'sightings_for_pet', 'get_nearby_hunters', 'owner_decide_sighting',
               'resolve_missing_pet', 'apply_score_penalty', 'assign_user_role',
               'find_user_by_email')
    LOOP
        EXECUTE format('REVOKE EXECUTE ON FUNCTION %s FROM PUBLIC, anon, authenticated', f.sig);
        -- Explicit, so the backend keeps working even if its access used to come from PUBLIC.
        EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO service_role', f.sig);
    END LOOP;
END $$;

COMMIT;
