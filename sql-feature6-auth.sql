-- =====================================================================
-- Feature #6: Authentication
-- Run this in the Supabase SQL editor (or `supabase db push`).
-- Idempotent: safe to re-run.
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. Auto-create a public.users profile row whenever a new auth user
--    signs up. This is the canonical Supabase pattern: the profile is
--    created atomically with the auth.users insert and cannot be skipped
--    by any client. display_name / phone come from the signUp metadata
--    (raw_user_meta_data); email stays the single source of truth in
--    auth.users and is intentionally NOT duplicated here.
-- ---------------------------------------------------------------------
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.users (id, display_name, phone)
  values (
    new.id,
    coalesce(nullif(new.raw_user_meta_data->>'display_name', ''), split_part(new.email, '@', 1)),
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
