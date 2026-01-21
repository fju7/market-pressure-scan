-- Market Pressure Scan: Supabase Schema
-- Run this in your Supabase SQL editor

-- 1) Who can access (invite-only until Stripe)
create table if not exists public.reviewers (
  user_id uuid primary key references auth.users(id) on delete cascade,
  email text unique,
  role text not null default 'reviewer', -- 'admin' | 'reviewer'
  created_at timestamptz not null default now()
);

-- 2) Strategy versions (V1, V2, etc.)
create table if not exists public.strategy_versions (
  id bigint generated always as identity primary key,
  experiment_name text not null,              -- e.g. 'news-novelty-v1'
  baseline_version text not null,             -- e.g. 'V1'
  locked_date date,
  benchmark_symbol text not null default 'SPY',
  unique (experiment_name, baseline_version)
);

-- 3) Weekly runs (shadow/live)
do $$ begin
  if not exists (select 1 from pg_type where typname = 'run_type') then
    create type public.run_type as enum ('shadow', 'real');
  end if;
  if not exists (select 1 from pg_type where typname = 'week_type') then
    create type public.week_type as enum ('CLEAN_TRADE', 'CLEAN_SKIP', 'DATA_IMPAIRED');
  end if;
end $$;

create table if not exists public.weeks (
  id bigint generated always as identity primary key,
  strategy_version_id bigint not null references public.strategy_versions(id) on delete cascade,
  run_type public.run_type not null default 'shadow',
  week_ending date not null,                  -- Friday date
  universe_type text not null default 'test',  -- 'test' | 'full'
  num_symbols_covered int,
  week_type public.week_type not null,
  completeness_pass boolean not null default false,
  missing_artifacts jsonb not null default '[]'::jsonb,
  skip_reasons jsonb not null default '{}'::jsonb,
  artifact_root text,                         -- path or URL prefix
  created_at timestamptz not null default now(),
  unique (strategy_version_id, run_type, week_ending)
);

-- 4) Weekly performance numbers
create table if not exists public.weekly_performance (
  week_id bigint primary key references public.weeks(id) on delete cascade,
  strategy_return numeric,
  benchmark_return numeric,
  active_return numeric,
  transaction_cost_bps numeric,
  basket_size int,
  computed_at timestamptz not null default now()
);

-- 5) Holdings for trade weeks (20 equal weight)
create table if not exists public.basket_holdings (
  id bigint generated always as identity primary key,
  week_id bigint not null references public.weeks(id) on delete cascade,
  symbol text not null,
  weight numeric not null,
  unique (week_id, symbol)
);

-- 6) Artifact links (optional but useful)
create table if not exists public.artifact_links (
  week_id bigint primary key references public.weeks(id) on delete cascade,
  report_meta_url text,
  basket_url text,
  scores_url text,
  ops_log_url text,
  bt_ledger_url text
);

-- RLS: default deny; we'll allow only reviewers.
alter table public.reviewers enable row level security;
alter table public.strategy_versions enable row level security;
alter table public.weeks enable row level security;
alter table public.weekly_performance enable row level security;
alter table public.basket_holdings enable row level security;
alter table public.artifact_links enable row level security;

-- helper: is reviewer?
create or replace function public.is_reviewer()
returns boolean
language sql stable
as $$
  select exists (select 1 from public.reviewers r where r.user_id = auth.uid());
$$;

-- policies: reviewers can read everything
create policy "reviewers read reviewers"
on public.reviewers for select
using (public.is_reviewer());

create policy "reviewers read strategy_versions"
on public.strategy_versions for select
using (public.is_reviewer());

create policy "reviewers read weeks"
on public.weeks for select
using (public.is_reviewer());

create policy "reviewers read weekly_performance"
on public.weekly_performance for select
using (public.is_reviewer());

create policy "reviewers read basket_holdings"
on public.basket_holdings for select
using (public.is_reviewer());

create policy "reviewers read artifact_links"
on public.artifact_links for select
using (public.is_reviewer());

-- For ingestion: you'll use service role key, so no insert policy needed.
