"""Supabase adapter for BountyRepository."""


class SupabaseBountyRepository:
    def __init__(self, db):
        self._db = db

    def get_pet(self, pet_id: str) -> dict | None:
        res = (self._db.table("missing_pets")
                       .select("id, owner_id, pet_name, status, bounty_amount, "
                               "found_at, unpaid_referred_at, bounty_unpaid")
                       .eq("id", pet_id)
                       .execute())
        return res.data[0] if res.data else None

    def get_final_sighting(self, pet_id: str) -> dict | None:
        res = self._db.rpc("bounty_final_sighting", {"p_pet_id": pet_id}).execute()
        return res.data[0] if res.data else None

    def get_latest_payment(self, pet_id: str) -> dict | None:
        res = (self._db.table("bounty_transactions")
                       .select("id, status, amount, slip_image_url, reference_no, "
                               "created_at, disputed_at, verified_at")
                       .eq("missing_pet_id", pet_id)
                       .order("created_at", desc=True)
                       .limit(1)
                       .execute())
        return res.data[0] if res.data else None

    def get_user_phone(self, user_id: str) -> str | None:
        res = (self._db.table("users").select("phone").eq("id", user_id).execute())
        return res.data[0].get("phone") if res.data else None

    def get_username(self, user_id: str) -> str | None:
        res = (self._db.table("users").select("username").eq("id", user_id).execute())
        return res.data[0].get("username") if res.data else None

    def submit_payment(
        self, pet_id: str, owner_id: str, slip_url: str, reference_no: str | None
    ) -> dict:
        res = self._db.rpc("submit_bounty_payment", {
            "p_pet_id": pet_id, "p_owner_id": owner_id,
            "p_slip_url": slip_url, "p_reference_no": reference_no,
        }).execute()
        return res.data or {}

    def hunter_answer(self, payment_id: str, hunter_id: str, answer: str) -> dict:
        res = self._db.rpc("hunter_answer_bounty_payment", {
            "p_payment_id": payment_id, "p_hunter_id": hunter_id, "p_answer": answer,
        }).execute()
        return res.data or {}

    def refer_unpaid(self, pet_id: str, hunter_id: str) -> dict:
        res = self._db.rpc("refer_bounty_unpaid", {
            "p_pet_id": pet_id, "p_hunter_id": hunter_id,
        }).execute()
        return res.data or {}

    def admin_rule(self, payment_id: str, admin_id: str, decision: str) -> dict:
        res = self._db.rpc("admin_rule_bounty_payment", {
            "p_payment_id": payment_id, "p_admin_id": admin_id, "p_decision": decision,
        }).execute()
        return res.data or {}

    def admin_close_unpaid(self, pet_id: str, admin_id: str) -> dict:
        res = self._db.rpc("admin_close_bounty_unpaid", {
            "p_pet_id": pet_id, "p_admin_id": admin_id,
        }).execute()
        return res.data or {}

    def ruling_queue(self) -> list[dict]:
        res = self._db.rpc("bounty_ruling_queue", {}).execute()
        return res.data or []

    def bounties_for_hunter(self, hunter_id: str) -> list[dict]:
        res = self._db.rpc("bounties_for_hunter", {"p_hunter_id": hunter_id}).execute()
        return res.data or []
