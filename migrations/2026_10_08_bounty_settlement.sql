-- 2026-10-08 — Feature 6: owner-paid bounty settlement.
--
-- STATUS: APPLIED (found applied to the database on 2026-10-10). Verified
-- against the integration container (tests/integration/test_bounty_settlement.py).
--
-- WHAT. The owner pays the returning hunter directly, by PromptPay, and submits
-- the slip. The hunter confirms or disputes it. The Administrator rules on
-- disputes, on payments left unconfirmed past the three-day dispute window, and
-- on reports the hunter referred as unpaid. The platform never holds the money.
-- Specified in petty_bounty_brain/wiki/final_progress/use_case_descriptions_f6.md.
--
-- REUSES. bounty_transactions and its transaction_status values
-- (Pending_Verification = awaiting confirmation, Verified = confirmed,
-- Rejected = rejected), verified_at / verified_by for who confirmed and when.
-- resolve_missing_pet is left in place untouched; the new functions supersede
-- it for the owner-paid flow.
--
-- Every RAISE message below is load-bearing: the service layer maps the
-- prefixes "not found", "not allowed", "already" / "is not" (state conflicts)
-- and "window" to 404 / 403 / 409 / 409. Change one and change it there too.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Columns
-- ---------------------------------------------------------------------------

-- When the system closed the search on a confirmed Caught sighting. Starts the
-- hunter's three-day window to refer the report as unpaid.
ALTER TABLE missing_pets ADD COLUMN IF NOT EXISTS found_at TIMESTAMPTZ;
-- When the returning hunter referred the report as unpaid. Cleared when the
-- owner submits a payment, which closes the referral.
ALTER TABLE missing_pets ADD COLUMN IF NOT EXISTS unpaid_referred_at TIMESTAMPTZ;
-- An Administrator closed the report with the bounty unpaid, and who did it.
ALTER TABLE missing_pets ADD COLUMN IF NOT EXISTS bounty_unpaid BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE missing_pets ADD COLUMN IF NOT EXISTS unpaid_closed_by UUID REFERENCES users(id);
ALTER TABLE missing_pets ADD COLUMN IF NOT EXISTS unpaid_closed_at TIMESTAMPTZ;

-- When the returning hunter disputed the payment. transaction_status has no
-- "disputed" value; a timestamp keeps the dispute on an awaiting payment
-- without rewriting the enumeration.
ALTER TABLE bounty_transactions ADD COLUMN IF NOT EXISTS disputed_at TIMESTAMPTZ;

-- At most one live payment per report: awaiting confirmation or confirmed.
-- Rejected payments are kept as history and do not count. The constraint is the
-- real guard; the check in submit_bounty_payment only gives a clearer message.
CREATE UNIQUE INDEX IF NOT EXISTS bounty_transactions_one_live_per_pet
    ON bounty_transactions (missing_pet_id)
 WHERE status IN ('Pending_Verification', 'Verified');

-- ---------------------------------------------------------------------------
-- 2. owner_decide_sighting — the same function as 2026-08-21, with two
--    changes when a Caught sighting is confirmed: the report goes straight to
--    'Resolved' when the bounty is zero (nobody to pay), and found_at is set.
-- ---------------------------------------------------------------------------
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
    v_new_status   TEXT;
    v_award_points INT[] := ARRAY[25, 15, 10, 5, 5];
    v_clue         RECORD;
    v_rank         INT := 0;
    v_award_id     UUID;
    v_awards       JSONB := '[]'::JSONB;
BEGIN
    IF p_decision NOT IN ('Confirmed', 'Rejected') THEN
        RAISE EXCEPTION 'decision must be Confirmed or Rejected (got %)', p_decision;
    END IF;

    SELECT * INTO v_pet FROM missing_pets WHERE id = p_pet_id FOR UPDATE;

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
        UPDATE sightings
           SET sighting_status = 'Confirmed'
         WHERE id = p_sighting_id
           AND sighting_status <> 'Closed';

        IF v_sighting.action_type = 'Caught' THEN
            v_closed := TRUE;

            -- Nobody to pay on a zero bounty, so there is no payment step and
            -- the case ends here. Otherwise it waits at 'Found' for the payment.
            v_new_status := CASE WHEN COALESCE(v_pet.bounty_amount, 0) > 0
                                 THEN 'Found' ELSE 'Resolved' END;

            UPDATE missing_pets
               SET status = v_new_status::pet_status,
                   found_at = NOW()
             WHERE id = p_pet_id;

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
                         ORDER BY s.hunter_id, s.created_at DESC, s.id DESC
                       ) picks
                 ORDER BY picks.created_at DESC, picks.sighting_id DESC
                 LIMIT array_length(v_award_points, 1)
            LOOP
                v_rank := v_rank + 1;

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
        'pet_status',     CASE WHEN v_closed THEN v_new_status
                               ELSE v_pet.status::TEXT END,
        'awards',         v_awards
    );
END;
$$;

-- ---------------------------------------------------------------------------
-- 3. Helper: the final sighting of a closed report — the Caught sighting whose
--    confirmation closed it — and so the returning hunter.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION bounty_final_sighting(p_pet_id UUID)
RETURNS TABLE (sighting_id UUID, hunter_id UUID)
LANGUAGE sql STABLE AS $$
    SELECT s.id, s.hunter_id
      FROM sighting_matches sm
      JOIN sightings s ON s.id = sm.sighting_id
     WHERE sm.missing_pet_id = p_pet_id
       AND sm.owner_status = 'Confirmed'
       AND s.action_type = 'Caught'
       AND s.verification_status <> 'Dismissed'
     ORDER BY s.created_at DESC, s.id DESC
     LIMIT 1;
$$;

-- ---------------------------------------------------------------------------
-- 4. The owner submits a payment for an unpaid report.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION submit_bounty_payment(
    p_pet_id       UUID,
    p_owner_id     UUID,
    p_slip_url     TEXT,
    p_reference_no VARCHAR
) RETURNS JSONB
LANGUAGE plpgsql AS $$
DECLARE
    v_pet   missing_pets%ROWTYPE;
    v_final RECORD;
    v_id    UUID;
BEGIN
    IF p_slip_url IS NULL OR btrim(p_slip_url) = '' THEN
        RAISE EXCEPTION 'A payment is not accepted without a slip image';
    END IF;

    SELECT * INTO v_pet FROM missing_pets WHERE id = p_pet_id FOR UPDATE;
    IF NOT FOUND OR v_pet.owner_id IS DISTINCT FROM p_owner_id THEN
        RAISE EXCEPTION 'Missing pet % not found or not owned by you', p_pet_id;
    END IF;
    IF v_pet.status <> 'Found' OR COALESCE(v_pet.bounty_amount, 0) <= 0 THEN
        RAISE EXCEPTION 'Missing pet % is not awaiting a bounty payment', p_pet_id;
    END IF;
    IF EXISTS (SELECT 1 FROM bounty_transactions
                WHERE missing_pet_id = p_pet_id
                  AND status IN ('Pending_Verification', 'Verified')) THEN
        RAISE EXCEPTION 'Missing pet % already has a payment awaiting confirmation or confirmed', p_pet_id;
    END IF;

    SELECT * INTO v_final FROM bounty_final_sighting(p_pet_id);
    IF v_final.sighting_id IS NULL THEN
        RAISE EXCEPTION 'Missing pet % is not awaiting a bounty payment', p_pet_id;
    END IF;

    INSERT INTO bounty_transactions
        (sighting_id, missing_pet_id, owner_id, amount, slip_image_url,
         reference_no, transfer_datetime, status)
    VALUES
        (v_final.sighting_id, p_pet_id, p_owner_id, v_pet.bounty_amount,
         btrim(p_slip_url), NULLIF(btrim(COALESCE(p_reference_no, '')), ''),
         NOW(), 'Pending_Verification')
    RETURNING id INTO v_id;

    -- Paying closes any referral as unpaid.
    UPDATE missing_pets SET unpaid_referred_at = NULL WHERE id = p_pet_id;

    RETURN jsonb_build_object('payment_id', v_id, 'pet_id', p_pet_id,
                              'hunter_id', v_final.hunter_id,
                              'amount', v_pet.bounty_amount,
                              'status', 'Pending_Verification');
END;
$$;

-- ---------------------------------------------------------------------------
-- 5. The returning hunter confirms a payment at any time, or disputes it
--    within its dispute window (three days after it was submitted).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION hunter_answer_bounty_payment(
    p_payment_id UUID,
    p_hunter_id  UUID,
    p_answer     TEXT
) RETURNS JSONB
LANGUAGE plpgsql AS $$
DECLARE
    v_tx    bounty_transactions%ROWTYPE;
    v_final RECORD;
BEGIN
    IF p_answer NOT IN ('Confirm', 'Dispute') THEN
        RAISE EXCEPTION 'answer must be Confirm or Dispute (got %)', p_answer;
    END IF;

    SELECT * INTO v_tx FROM bounty_transactions WHERE id = p_payment_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Payment % not found', p_payment_id;
    END IF;

    SELECT * INTO v_final FROM bounty_final_sighting(v_tx.missing_pet_id);
    IF v_final.hunter_id IS DISTINCT FROM p_hunter_id THEN
        RAISE EXCEPTION 'Payment % not found', p_payment_id;
    END IF;

    IF v_tx.status <> 'Pending_Verification' THEN
        RAISE EXCEPTION 'Payment % is already %', p_payment_id, v_tx.status;
    END IF;

    IF p_answer = 'Confirm' THEN
        UPDATE bounty_transactions
           SET status = 'Verified', verified_at = NOW(), verified_by = p_hunter_id
         WHERE id = p_payment_id;
        UPDATE missing_pets SET status = 'Resolved' WHERE id = v_tx.missing_pet_id;
        RETURN jsonb_build_object('payment_id', p_payment_id, 'status', 'Verified',
                                  'pet_status', 'Resolved');
    END IF;

    IF v_tx.disputed_at IS NOT NULL THEN
        RAISE EXCEPTION 'Payment % is already disputed', p_payment_id;
    END IF;
    IF NOW() > v_tx.created_at + INTERVAL '72 hours' THEN
        RAISE EXCEPTION 'The dispute window for payment % has closed', p_payment_id;
    END IF;

    UPDATE bounty_transactions SET disputed_at = NOW() WHERE id = p_payment_id;
    RETURN jsonb_build_object('payment_id', p_payment_id,
                              'status', 'Pending_Verification', 'disputed', TRUE);
END;
$$;

-- ---------------------------------------------------------------------------
-- 6. The returning hunter refers an unpaid report to the Administrator, within
--    three days of the report becoming 'Found'.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION refer_bounty_unpaid(
    p_pet_id    UUID,
    p_hunter_id UUID
) RETURNS JSONB
LANGUAGE plpgsql AS $$
DECLARE
    v_pet   missing_pets%ROWTYPE;
    v_final RECORD;
BEGIN
    SELECT * INTO v_pet FROM missing_pets WHERE id = p_pet_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Missing pet % not found', p_pet_id;
    END IF;

    SELECT * INTO v_final FROM bounty_final_sighting(p_pet_id);
    IF v_final.hunter_id IS DISTINCT FROM p_hunter_id THEN
        RAISE EXCEPTION 'Missing pet % not found', p_pet_id;
    END IF;

    IF v_pet.status <> 'Found' OR COALESCE(v_pet.bounty_amount, 0) <= 0
       OR EXISTS (SELECT 1 FROM bounty_transactions
                   WHERE missing_pet_id = p_pet_id
                     AND status IN ('Pending_Verification', 'Verified')) THEN
        RAISE EXCEPTION 'Missing pet % is not unpaid', p_pet_id;
    END IF;
    IF v_pet.unpaid_referred_at IS NOT NULL THEN
        RAISE EXCEPTION 'Missing pet % is already referred as unpaid', p_pet_id;
    END IF;
    IF v_pet.found_at IS NULL OR NOW() > v_pet.found_at + INTERVAL '72 hours' THEN
        RAISE EXCEPTION 'The window to refer missing pet % as unpaid has closed', p_pet_id;
    END IF;

    UPDATE missing_pets SET unpaid_referred_at = NOW() WHERE id = p_pet_id;
    RETURN jsonb_build_object('pet_id', p_pet_id, 'referred', TRUE);
END;
$$;

-- ---------------------------------------------------------------------------
-- 7. The Administrator confirms or rejects a payment in the queue.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION admin_rule_bounty_payment(
    p_payment_id UUID,
    p_admin_id   UUID,
    p_decision   TEXT
) RETURNS JSONB
LANGUAGE plpgsql AS $$
DECLARE
    v_tx bounty_transactions%ROWTYPE;
BEGIN
    IF p_decision NOT IN ('Confirm', 'Reject') THEN
        RAISE EXCEPTION 'decision must be Confirm or Reject (got %)', p_decision;
    END IF;

    SELECT * INTO v_tx FROM bounty_transactions WHERE id = p_payment_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Payment % not found', p_payment_id;
    END IF;
    IF v_tx.status <> 'Pending_Verification' THEN
        RAISE EXCEPTION 'Payment % is already %', p_payment_id, v_tx.status;
    END IF;

    IF p_decision = 'Confirm' THEN
        UPDATE bounty_transactions
           SET status = 'Verified', verified_at = NOW(), verified_by = p_admin_id
         WHERE id = p_payment_id;
        UPDATE missing_pets SET status = 'Resolved' WHERE id = v_tx.missing_pet_id;
        RETURN jsonb_build_object('payment_id', p_payment_id, 'status', 'Verified',
                                  'pet_status', 'Resolved');
    END IF;

    -- Rejected: the report stays 'Found' and the owner pays again.
    UPDATE bounty_transactions
       SET status = 'Rejected', verified_at = NOW(), verified_by = p_admin_id
     WHERE id = p_payment_id;
    RETURN jsonb_build_object('payment_id', p_payment_id, 'status', 'Rejected',
                              'pet_status', 'Found', 'owner_id', v_tx.owner_id);
END;
$$;

-- ---------------------------------------------------------------------------
-- 8. The Administrator closes an unpaid report as 'Resolved', recording that
--    the bounty was not paid.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION admin_close_bounty_unpaid(
    p_pet_id   UUID,
    p_admin_id UUID
) RETURNS JSONB
LANGUAGE plpgsql AS $$
DECLARE
    v_pet missing_pets%ROWTYPE;
BEGIN
    SELECT * INTO v_pet FROM missing_pets WHERE id = p_pet_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Missing pet % not found', p_pet_id;
    END IF;
    IF v_pet.status <> 'Found' OR COALESCE(v_pet.bounty_amount, 0) <= 0
       OR EXISTS (SELECT 1 FROM bounty_transactions
                   WHERE missing_pet_id = p_pet_id
                     AND status IN ('Pending_Verification', 'Verified')) THEN
        RAISE EXCEPTION 'Missing pet % is not unpaid', p_pet_id;
    END IF;

    UPDATE missing_pets
       SET status = 'Resolved', bounty_unpaid = TRUE,
           unpaid_closed_by = p_admin_id, unpaid_closed_at = NOW()
     WHERE id = p_pet_id;
    RETURN jsonb_build_object('pet_id', p_pet_id, 'pet_status', 'Resolved',
                              'bounty_unpaid', TRUE);
END;
$$;

-- ---------------------------------------------------------------------------
-- 9. The Administrator's queue: disputed payments, payments left unconfirmed
--    past their dispute window, reports referred as unpaid, and rejected
--    payments not replaced within three days. Derived on
--    read, so no scheduled job is needed.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION bounty_ruling_queue()
RETURNS TABLE (
    kind           TEXT,
    payment_id     UUID,
    pet_id         UUID,
    pet_name       TEXT,
    owner_id       UUID,
    hunter_id      UUID,
    amount         NUMERIC,
    slip_image_url TEXT,
    reference_no   TEXT,
    since          TIMESTAMPTZ
)
LANGUAGE sql STABLE AS $$
    SELECT CASE WHEN t.disputed_at IS NOT NULL THEN 'disputed' ELSE 'unconfirmed' END,
           t.id, mp.id, mp.pet_name::TEXT, mp.owner_id, s.hunter_id, t.amount,
           t.slip_image_url, t.reference_no::TEXT,
           COALESCE(t.disputed_at, t.created_at + INTERVAL '72 hours')
      FROM bounty_transactions t
      JOIN missing_pets mp ON mp.id = t.missing_pet_id
      JOIN sightings s ON s.id = t.sighting_id
     WHERE t.status = 'Pending_Verification'
       AND (t.disputed_at IS NOT NULL OR NOW() > t.created_at + INTERVAL '72 hours')
    UNION ALL
    SELECT 'unpaid', NULL, mp.id, mp.pet_name::TEXT, mp.owner_id, f.hunter_id,
           mp.bounty_amount, NULL, NULL, mp.unpaid_referred_at
      FROM missing_pets mp
      CROSS JOIN LATERAL bounty_final_sighting(mp.id) f
     WHERE mp.status = 'Found'
       AND mp.unpaid_referred_at IS NOT NULL
    UNION ALL
    -- A payment the Administrator rejected more than three days ago, with no
    -- new payment since. Without this, a report whose referral was closed by
    -- the (rejected) payment would never reach anyone again. The action on it
    -- is to close the report as unpaid, so no payment id is given.
    SELECT 'rejected_unpaid', NULL, mp.id, mp.pet_name::TEXT, mp.owner_id, f.hunter_id,
           mp.bounty_amount, r.slip_image_url, r.reference_no::TEXT,
           r.verified_at + INTERVAL '72 hours'
      FROM missing_pets mp
      CROSS JOIN LATERAL bounty_final_sighting(mp.id) f
      CROSS JOIN LATERAL (
            SELECT bt.status, bt.slip_image_url, bt.reference_no, bt.verified_at
              FROM bounty_transactions bt
             WHERE bt.missing_pet_id = mp.id
             ORDER BY bt.created_at DESC
             LIMIT 1
      ) r
     WHERE mp.status = 'Found'
       AND mp.unpaid_referred_at IS NULL
       AND r.status = 'Rejected'
       AND NOW() > r.verified_at + INTERVAL '72 hours'
       AND NOT EXISTS (SELECT 1 FROM bounty_transactions live
                        WHERE live.missing_pet_id = mp.id
                          AND live.status IN ('Pending_Verification', 'Verified'))
    ORDER BY 10;
$$;

-- ---------------------------------------------------------------------------
-- 9b. The reports a hunter returned the pet for, with their payment state, so
--     the hunter can find what they are owed. Newest closure first.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION bounties_for_hunter(p_hunter_id UUID)
RETURNS TABLE (
    pet_id             UUID,
    pet_name           TEXT,
    image_url          TEXT,
    pet_status         TEXT,
    bounty_amount      NUMERIC,
    found_at           TIMESTAMPTZ,
    unpaid_referred_at TIMESTAMPTZ,
    bounty_unpaid      BOOLEAN,
    payment_id         UUID,
    payment_status     TEXT,
    disputed_at        TIMESTAMPTZ
)
LANGUAGE sql STABLE AS $$
    SELECT mp.id, mp.pet_name::TEXT, mp.image_url::TEXT, mp.status::TEXT,
           mp.bounty_amount, mp.found_at, mp.unpaid_referred_at, mp.bounty_unpaid,
           t.id, t.status::TEXT, t.disputed_at
      FROM missing_pets mp
      CROSS JOIN LATERAL bounty_final_sighting(mp.id) f
      LEFT JOIN LATERAL (
            SELECT bt.id, bt.status, bt.disputed_at
              FROM bounty_transactions bt
             WHERE bt.missing_pet_id = mp.id
             ORDER BY bt.created_at DESC
             LIMIT 1
      ) t ON TRUE
     WHERE f.hunter_id = p_hunter_id
       AND mp.status IN ('Found', 'Resolved')
       AND COALESCE(mp.bounty_amount, 0) > 0
     ORDER BY mp.found_at DESC NULLS LAST;
$$;

-- ---------------------------------------------------------------------------
-- 9c. Backfill the reports that were already "Found" before this migration.
--     Before it, the owner's hand close wrote "Found" too, so a "Found" report
--     with no returning hunter (or no bounty) has nobody to pay: it is closed
--     to "Resolved", as a hand close now is. The rest get found_at stamped
--     with the migration time, which opens their hunters' three-day window to
--     refer them as unpaid from today. A function so the integration suite can
--     run it against seeded rows; it is idempotent.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION bounty_backfill_existing()
RETURNS JSONB
LANGUAGE plpgsql AS $$
DECLARE
    v_closed  INT;
    v_stamped INT;
BEGIN
    UPDATE missing_pets mp
       SET status = 'Resolved'
     WHERE mp.status = 'Found'
       AND (COALESCE(mp.bounty_amount, 0) <= 0
            OR NOT EXISTS (SELECT 1 FROM bounty_final_sighting(mp.id)));
    GET DIAGNOSTICS v_closed = ROW_COUNT;

    UPDATE missing_pets
       SET found_at = NOW()
     WHERE status = 'Found'
       AND found_at IS NULL;
    GET DIAGNOSTICS v_stamped = ROW_COUNT;

    RETURN jsonb_build_object('closed_to_resolved', v_closed,
                              'found_at_stamped', v_stamped);
END;
$$;

SELECT bounty_backfill_existing();

-- ---------------------------------------------------------------------------
-- 10. The functions are the backend's. Only the service role may call them,
--     matching 2026_09_30_lock_down_direct_access.sql.
-- ---------------------------------------------------------------------------
DO $$
DECLARE f record;
BEGIN
    FOR f IN
        SELECT p.oid::regprocedure AS sig
          FROM pg_proc p
         WHERE p.pronamespace = 'public'::regnamespace
           AND p.prokind = 'f'
           AND p.proname IN (
               'owner_decide_sighting', 'bounty_final_sighting',
               'submit_bounty_payment', 'hunter_answer_bounty_payment',
               'refer_bounty_unpaid', 'admin_rule_bounty_payment',
               'admin_close_bounty_unpaid', 'bounty_ruling_queue',
               'bounties_for_hunter', 'bounty_backfill_existing')
    LOOP
        EXECUTE format('REVOKE EXECUTE ON FUNCTION %s FROM PUBLIC', f.sig);
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
            EXECUTE format('REVOKE EXECUTE ON FUNCTION %s FROM anon, authenticated', f.sig);
        END IF;
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
            EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO service_role', f.sig);
        END IF;
    END LOOP;
END $$;

COMMIT;
