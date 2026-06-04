"""
Provision a pre-set Administrator account (Feature #6).

Admins are not self-registered — they are created out-of-band with this
script, which uses the Supabase service key to:
  1. Create a confirmed auth user (no email verification step), and
  2. Flip the mirrored public.users row's role to 'admin'.

Usage:
    python seed_admin.py <email> <password> [display_name]

Requires SUPABASE_URL and SUPABASE_SERVICE_KEY in the environment / .env.
The service key is admin-scoped — never ship it to the client.
"""
import sys

from supabase import create_client

from app.core.config import settings


def seed_admin(email: str, password: str, display_name: str) -> None:
    client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

    # 1. Create a confirmed auth user. The handle_new_user trigger mirrors a
    #    public.users row with role='user'.
    created = client.auth.admin.create_user(
        {
            "email": email,
            "password": password,
            "email_confirm": True,
            "user_metadata": {"display_name": display_name},
        }
    )
    user_id = created.user.id
    print(f"Created auth user {email} -> {user_id}")

    # 2. Promote the mirrored profile to admin.
    client.table("users").update({"role": "admin"}).eq("id", user_id).execute()
    print(f"Granted role='admin' to {user_id}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python seed_admin.py <email> <password> [display_name]")
        raise SystemExit(1)

    email = sys.argv[1]
    password = sys.argv[2]
    display_name = sys.argv[3] if len(sys.argv) > 3 else email.split("@")[0]
    seed_admin(email, password, display_name)
