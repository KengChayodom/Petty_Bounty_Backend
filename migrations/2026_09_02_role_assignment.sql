-- ============================================================
-- Feature 4 — administrator role assignment (URS-23, SRS-94..98, UD-23)
-- Apply via Supabase SQL Editor or `supabase db push`.
--
-- The console reads `users.role` on every request (require_admin) but nothing
-- could ever WRITE it: an administrator existed only by hand-editing the row.
-- This adds the supported way to grant and withdraw that access, and the audit
-- trail for it.
--
-- What this is NOT: the account list / search / suspend / deactivate subsystem
-- struck on 2026-08-21. Nothing here can suspend, deactivate, or delete an
-- account, and find_user_by_email resolves ONE account from a full address
-- rather than returning a page — see MD-58's note on why that line matters.
--
-- Adds:
--   1. role_changes        — append-only audit of every grant and withdrawal
--   2. find_user_by_email()— exact-address lookup across auth.users (MD-58)
--   3. assign_user_role()  — guards + role write + audit row, atomically (MD-59)
-- ============================================================

BEGIN;

-- ---------- 1. role_changes ----------
-- Append-only. Nothing in the design updates or deletes a row, so an account's
-- access history stays complete for as long as the table is kept.
--
-- `role_before` and `role_after` are stored rather than derived because the
-- point of the record is what the change WAS, and replaying the current role
-- backwards through the log is not the same thing once a row is missing.
--
-- ON DELETE SET NULL on changed_by, CASCADE on target_user_id: deleting the
-- administrator who acted must not erase the fact that the change happened,
-- but a deleted account has no access history worth keeping.
CREATE TABLE role_changes (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    changed_by     UUID REFERENCES users(id) ON DELETE SET NULL,
    role_before    user_role NOT NULL,
    role_after     user_role NOT NULL,
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT role_changes_actually_changed CHECK (role_before <> role_after)
);

CREATE INDEX role_changes_created_idx
    ON role_changes (created_at DESC);
CREATE INDEX role_changes_target_idx
    ON role_changes (target_user_id, created_at DESC);

-- ---------- 2. find_user_by_email ----------
-- SECURITY DEFINER because the email address lives in `auth.users`, which
-- PostgREST cannot reach and the API key has no grant on. The function is the
-- narrow window onto it: one exact address in, one row or none out. It exposes
-- no email address it was not already given, and it cannot enumerate.
--
-- Case-insensitive because GoTrue lower-cases addresses on sign-up but a person
-- typing one into the console will not.
CREATE OR REPLACE FUNCTION find_user_by_email(p_email TEXT)
RETURNS JSON
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, auth
AS $$
DECLARE
    v_row RECORD;
BEGIN
    SELECT u.id, u.display_name, u.role
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
        'display_name', v_row.display_name,
        'role',         v_row.role
    );
END;
$$;

-- ---------- 3. assign_user_role ----------
-- Atomic: the guards, the role write, and the audit row land together or not
-- at all.
--
-- Both guards live HERE rather than in the service on purpose. Evaluated in
-- Python ahead of the write, the administrator count is a read another
-- transaction can invalidate before the write lands, so two administrators
-- withdrawing each other's access at the same moment would each see the other
-- and leave the console unreachable. Locking every admin row serialises them.
-- The set is tiny, so the lock is cheap.
--
-- Assigning the role an account already holds is not an error: it reports
-- changed=false and writes no audit row, so a retried request cannot pad the
-- history. That is also why role_changes carries a CHECK that the two roles
-- differ — a no-op row is unrepresentable.
CREATE OR REPLACE FUNCTION assign_user_role(
    p_target_user_id UUID,
    p_role           user_role,
    p_changed_by     UUID
)
RETURNS JSON
LANGUAGE plpgsql
AS $$
DECLARE
    v_before   user_role;
    v_name     TEXT;
    v_admins   INT;
    v_change_id UUID;
BEGIN
    SELECT role, display_name INTO v_before, v_name
      FROM users WHERE id = p_target_user_id FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'user % not found', p_target_user_id;
    END IF;

    IF v_before = p_role THEN
        RETURN json_build_object(
            'changed',      FALSE,
            'id',           p_target_user_id,
            'display_name', v_name,
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
        'display_name',   v_name,
        'role_before',    v_before,
        'role_after',     p_role
    );
END;
$$;

COMMIT;
