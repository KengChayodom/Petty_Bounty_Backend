-- =====================================================================
-- Feature #6: Authentication
-- Run this in the Supabase SQL editor (or `supabase db push`).
-- Idempotent: safe to re-run.
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. Auto-create a public.users profile row whenever a new auth user
--    signs up. This is the canonical Supabase pattern: the profile is
--    created atomically with the auth.users insert and cannot be skipped
--    by any client. username / phone come from the signUp metadata
--    (raw_user_meta_data); email stays the single source of truth in
--    auth.users and is intentionally NOT duplicated here.
--    The metadata key was `display_name` until 2026-09-09 and is still read
--    as a fallback so a client build from before the rename keeps working;
--    see migrations/2026_09_09_rename_display_name_to_username.sql.
-- ---------------------------------------------------------------------
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
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
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ---------------------------------------------------------------------
-- 2. Row Level Security on public.users (defense-in-depth).
--    The FastAPI backend uses the service-role key and bypasses RLS;
--    these policies protect the direct Flutter <-> Supabase auth channel
--    so a logged-in user can only ever read/update their own row.
-- ---------------------------------------------------------------------
alter table public.users enable row level security;

drop policy if exists "Users can view own profile" on public.users;
create policy "Users can view own profile"
  on public.users for select
  using (auth.uid() = id);

drop policy if exists "Users can update own profile" on public.users;
create policy "Users can update own profile"
  on public.users for update
  using (auth.uid() = id)
  with check (auth.uid() = id);
