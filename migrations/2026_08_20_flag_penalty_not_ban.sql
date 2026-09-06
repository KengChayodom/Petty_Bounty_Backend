-- ============================================================
-- Feature: uphold a flag by DEDUCTING SCORE, not by banning (SRS-73 revision)
-- Apply via Supabase SQL Editor or `supabase db push`.
--
-- Replaces the never-implemented "ban the reported user" outcome. Banning was
-- rejected because it has no cheap representation here: `users` carries no
-- account-state column, `user_role` is only {user, admin}, and an already-issued
-- JWT stays valid — so every request would have to re-check account state
-- against the DB. A score deduction touches only the scoring tables and leaves
-- the auth path alone.
--
-- Adds:
--   1. report_status 'Reviewed_Ban' renamed to 'Reviewed_Penalty'
--   2. score_penalties          — the deduction audit trail (mirror of score_awards)
--   3. apply_score_penalty()    — RPC: insert audit row + decrement total_score, atomically
--
-- Split into two transactions on purpose: PostgreSQL forbids using an enum
-- value in the same transaction that alters the type, and step 3's function
-- body references report_reason. RENAME VALUE does not rewrite the table — the
-- on-disk representation is the enum's OID, not its label — so step 1 is O(1)
-- and reversible by renaming back.
-- ============================================================

BEGIN;

-- ---------- 1. rename the decision that no longer bans anyone ----------
ALTER TYPE report_status RENAME VALUE 'Reviewed_Ban' TO 'Reviewed_Penalty';

COMMIT;

BEGIN;

-- ---------- 2. score_penalties ----------
-- Deliberately NOT a negative row in `score_awards`: that table has
-- `rank INTEGER NOT NULL` (a penalty has no rank), `missing_pet_id NOT NULL`
-- (a discovery sighting has no pet attached), and UNIQUE(missing_pet_id,
-- user_id) — which a hunter who was both awarded and later penalised on the
-- same case would violate.
--
-- UNIQUE(report_id) is the idempotency key. `review_report` writes the flag's
-- own status LAST so a mid-way failure is retryable; without this constraint a
-- retry would deduct the points a second time.
--
-- `points` records what the administrator RULED, not what was actually
-- subtracted. The two differ when the hunter had less score than the penalty:
-- total_score floors at 0 (see step 3) but the ruling stands as written.
--
-- report_id CASCADEs rather than SET NULL because NOT NULL is what makes the
-- UNIQUE usable as an idempotency key. Nothing deletes a flag today (there is
-- no delete-flag route), so the audit trail is not actually reachable by this
-- path; if one is ever added, it has to decide what happens to the deduction
-- and this constraint is where that decision goes.
CREATE TABLE score_penalties (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    sighting_id  UUID REFERENCES sightings(id) ON DELETE SET NULL,
    report_id    UUID NOT NULL UNIQUE REFERENCES reports(id) ON DELETE CASCADE,
    points       INTEGER NOT NULL CHECK (points >= 0),
    reason       report_reason,
    penalised_by UUID REFERENCES users(id),
    penalised_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX score_penalties_user_idx
    ON score_penalties (user_id, penalised_at DESC);

-- ---------- 3. apply_score_penalty ----------
-- Atomic: the audit row and the balance change land together or not at all.
-- Returning `already_applied` rather than raising lets the caller retry a
-- part-failed review without special-casing the second attempt.
CREATE OR REPLACE FUNCTION apply_score_penalty(
    p_user_id      UUID,
    p_sighting_id  UUID,
    p_report_id    UUID,
    p_points       INT,
    p_reason       report_reason,
    p_penalised_by UUID
)
RETURNS JSON
LANGUAGE plpgsql
AS $$
DECLARE
    v_penalty_id UUID;
    v_before     INT;
    v_applied    INT;
    v_after      INT;
BEGIN
    IF p_points < 0 THEN
        RAISE EXCEPTION 'penalty points must not be negative (got %)', p_points;
    END IF;

    INSERT INTO score_penalties
        (user_id, sighting_id, report_id, points, reason, penalised_by)
    VALUES
        (p_user_id, p_sighting_id, p_report_id, p_points, p_reason, p_penalised_by)
    ON CONFLICT (report_id) DO NOTHING
    RETURNING id INTO v_penalty_id;

    -- Conflict = this flag was already penalised by an earlier attempt.
    -- Report the standing balance and subtract nothing.
    IF v_penalty_id IS NULL THEN
        SELECT total_score INTO v_after FROM users WHERE id = p_user_id;
        RETURN json_build_object(
            'already_applied',   TRUE,
            'points',            p_points,
            'points_applied',    0,
            'total_score_after', COALESCE(v_after, 0)
        );
    END IF;

    -- FOR UPDATE: two admins upholding two flags against the same hunter at
    -- once must not both read the same pre-deduction balance.
    SELECT COALESCE(total_score, 0) INTO v_before
    FROM users WHERE id = p_user_id FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'user % not found', p_user_id;
    END IF;

    -- Floor at zero: a negative total_score would sort below hunters who have
    -- never contributed at all, which reads as a broken leaderboard.
    v_applied := LEAST(p_points, GREATEST(v_before, 0));

    UPDATE users
       SET total_score = v_before - v_applied
     WHERE id = p_user_id
    RETURNING total_score INTO v_after;

    RETURN json_build_object(
        'already_applied',   FALSE,
        'penalty_id',        v_penalty_id,
        'points',            p_points,
        'points_applied',    v_applied,
        'total_score_after', v_after
    );
END;
$$;

COMMIT;
