"""
debug_notify_owner.py — isolate why a sighting never advanced to
`Notified_Owner`, i.e. which gate in `notify_pet_owners` swallowed the push.

`sighting_status` only leaves 'Pending_Analysis' when FCM actually delivered
something (notification_service.py). A stuck status therefore has FIVE possible
causes and "FCM is broken" is only one of them. This script walks all five, in
the order the code hits them:

    1. pet_ids empty      — the sighting matched nothing, and a targeted
                            sighting names no pet. Nobody to tell.
    2. Firebase not ready — FIREBASE_CREDENTIALS unset/invalid, so every send
                            is a logged no-op.
    3. owner_ids empty    — every matched pet is owned by the reporting hunter.
                            You do not push a person about their own pet.
    4. no device tokens   — the owners exist but have never registered a token
                            (or theirs were pruned as UNREGISTERED).
    5. FCM send failed    — the only cause that actually means FCM is broken.

It is READ-ONLY: FCM is exercised with dry_run=True, which validates
credentials and each token without delivering anything, and no token is pruned.

Run from the backend root with .env loaded:
    python debug_notify_owner.py                 # last 20 sightings
    python debug_notify_owner.py <sighting_id>   # one sighting
    python debug_notify_owner.py --tokens        # token validation only
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from app.core.database import get_supabase_client  # noqa: E402
from app.core.firebase import init_firebase, is_firebase_ready  # noqa: E402

BAR = "=" * 68


def _db():
    client = get_supabase_client()
    return next(client) if hasattr(client, "__next__") else client


def check_firebase() -> bool:
    """Gate 2 — credentials. Without this every send is a silent no-op."""
    print(BAR)
    print("GATE 2 — Firebase credentials")
    ready = init_firebase() and is_firebase_ready()
    path = os.getenv("FIREBASE_CREDENTIALS") or "(unset)"
    print(f"  FIREBASE_CREDENTIALS = {path}")
    print(f"  is_firebase_ready()  = {ready}")
    if not ready:
        print("  -> Every push is a no-op. Nothing else matters until this is True.")
    return ready


def check_tokens(db, ready: bool) -> set:
    """Gate 4 — device tokens, validated against FCM without delivering."""
    print(BAR)
    print("GATE 4 — device tokens (dry-run validated, nothing delivered)")
    rows = db.table("device_tokens").select("user_id, fcm_token, updated_at").execute().data
    if not rows:
        print("  No device tokens at all. No user can receive a push.")
        return set()

    print(f"  {len(rows)} token(s) across {len({r['user_id'] for r in rows})} user(s)")
    if not ready:
        print("  Firebase not ready — cannot validate. Listing only.")
        return {r["user_id"] for r in rows}

    from firebase_admin import messaging

    tokens = [r["fcm_token"] for r in rows if r.get("fcm_token")]
    msg = messaging.MulticastMessage(
        tokens=tokens,
        notification=messaging.Notification(title="dry run", body="dry run"),
    )
    resp = messaging.send_each_for_multicast(msg, dry_run=True)
    print(f"  would deliver {resp.success_count}/{len(tokens)}, would fail {resp.failure_count}")

    live = set()
    for row, res in zip(rows, resp.responses):
        tok, user = (row.get("fcm_token") or "")[:20], row["user_id"]
        stamp = str(row.get("updated_at"))[:10]
        if res.success:
            live.add(user)
            print(f"    VALID    {tok}... user={user[:8]} updated={stamp}")
        else:
            why = type(res.exception).__name__
            print(f"    DEAD     {tok}... user={user[:8]} {why} "
                  f"({'send_to_users would prune this' if why == 'UnregisteredError' else 'kept'})")
    if not live:
        print("  -> Every token is dead. Delivery is 0 and the status stays put.")
    return live


def walk_sightings(db, live_token_users: set, sighting_id: str | None):
    """Gates 1 and 3 — per sighting, the reason it never reached FCM."""
    print(BAR)
    print("GATES 1 & 3 — per sighting: is there anyone to notify?")

    q = db.table("sightings").select(
        "id, hunter_id, sighting_status, initial_target_pet_id, created_at"
    )
    if sighting_id:
        q = q.eq("id", sighting_id)
    else:
        q = q.order("created_at", desc=True).limit(20)
    sightings = q.execute().data
    if not sightings:
        print("  No sightings found.")
        return

    verdicts = {}
    for s in sightings:
        sid, hunter = s["id"], s["hunter_id"]
        pet_ids = []
        if s.get("initial_target_pet_id"):
            pet_ids = [s["initial_target_pet_id"]]
        else:
            rows = (db.table("sighting_matches").select("missing_pet_id")
                    .eq("sighting_id", sid).execute().data)
            pet_ids = [r["missing_pet_id"] for r in rows]

        if not pet_ids:
            verdict = "GATE 1: matched nothing — nobody to tell"
        else:
            owners = (db.table("missing_pets").select("owner_id")
                      .in_("id", pet_ids).execute().data)
            owner_ids = {o["owner_id"] for o in owners if o.get("owner_id")}
            others = owner_ids - {hunter}
            if not others:
                verdict = "GATE 3: hunter owns every matched pet (self-match)"
            elif not (others & live_token_users):
                verdict = f"GATE 4: {len(others)} owner(s), none with a live token"
            else:
                verdict = "REACHES FCM — should have advanced"

        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        flag = "OK " if s["sighting_status"] != "Pending_Analysis" else "   "
        print(f"  {flag}{sid[:8]}  {str(s['created_at'])[:10]}  "
              f"{s['sighting_status']:<17} {verdict}")

    print(f"\n  Summary over {len(sightings)} sighting(s):")
    for verdict, n in sorted(verdicts.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>3}  {verdict}")


def readiness(db, live_token_users: set):
    """Can an end-to-end test even succeed with the current data?"""
    print(BAR)
    print("END-TO-END READINESS — can a real push happen at all today?")
    pets = (db.table("missing_pets").select("id, pet_name, owner_id, status")
            .eq("status", "Searching").execute().data)
    eligible = [p for p in pets if p.get("owner_id") in live_token_users]
    print(f"  {len(pets)} pet(s) in 'Searching'; "
          f"{len(eligible)} owned by a user with a LIVE token")
    if not eligible:
        print("  -> No cross-account push can succeed yet. To make one possible:")
        print("     1. log in as a SECOND account and register its device token")
        print("     2. post a missing pet from that account (status 'Searching')")
        print("     3. report a sighting of it from a DIFFERENT account")
        return
    for p in eligible[:5]:
        print(f"     pet={p['id'][:8]} '{p.get('pet_name')}' owner={p['owner_id'][:8]}")
    print("  -> Report a sighting of one of these from ANY OTHER account;")
    print("     sighting_status should become 'Notified_Owner'.")


def main():
    args = [a for a in sys.argv[1:]]
    tokens_only = "--tokens" in args
    sighting_id = next((a for a in args if not a.startswith("--")), None)

    db = _db()
    ready = check_firebase()
    live = check_tokens(db, ready)
    if not tokens_only:
        walk_sightings(db, live, sighting_id)
        readiness(db, live)
    print(BAR)


if __name__ == "__main__":
    main()
