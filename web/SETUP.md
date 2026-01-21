# Market Pressure Scan — Next.js + Supabase MVP

## Quick Start

### 1) Supabase Setup

Create a free project at https://supabase.com

In your Supabase SQL editor, run the schema from [SUPABASE_SCHEMA.sql](./SUPABASE_SCHEMA.sql).

### 2) Environment Variables

Create `web/.env.local`:

```env
NEXT_PUBLIC_SUPABASE_URL=https://YOUR_PROJECT.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=YOUR_ANON_KEY
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY
```

Get these from Supabase **Settings → API**.

### 3) Add Your User to Reviewers

1. Sign up in the login page (creates an auth user)
2. In Supabase table editor, go to `reviewers` table
3. Insert your `user_id` (from `auth.users`) with role `admin`

```sql
INSERT INTO public.reviewers (user_id, email, role) VALUES
  ('YOUR_USER_ID', 'you@example.com', 'admin');
```

### 4) Run Dev Server

```bash
cd web
npm run dev
```

Visit http://localhost:3000/login

### 5) Ingest Data

Install Python client:

```bash
python -m pip install supabase
```

Run the ingestion script:

```bash
cd web
python scripts/ingest_week.py \
  --artifact-dir ../artifacts/news-novelty-v1/V1/2026-01-02 \
  --run-type shadow
```

The script will:
- Read `report_meta.json`, `basket.csv`, `scores_weekly.parquet`, `ops_compact_friday.log`
- Classify the week (CLEAN_TRADE / CLEAN_SKIP / DATA_IMPAIRED)
- Write to Postgres
- Charts will populate once `weekly_performance` has `strategy_return` / `benchmark_return`

---

## Next Steps: Performance Numbers

To see equity curves on the dashboard, we need to populate `weekly_performance`:

1. Where is `bt_weekly.parquet` in your artifact structure?
2. What columns does it have?

Paste your artifact tree for one week, and I'll add the performance parsing.

---

## Architecture

### Routes

- `/login` — Email/password auth (Supabase)
- `/dashboard` — Protected; only reviewers can access
- `/not-authorized` — Shown if user is not in allowlist

### Tables

- `strategy_versions` — V1, V2, etc.
- `weeks` — Weekly runs (shadow/real)
- `weekly_performance` — Returns + metrics
- `basket_holdings` — 20 equal-weight holdings per week
- `reviewers` — Allowlist (invite-only until Stripe)

### RLS

All tables use Row-Level Security:
- Only rows where `is_reviewer() = true` are readable
- Ingestion uses service role key (no RLS)

---

## Later: Stripe Integration

When ready:

1. Add `entitlements` table (or `subscriptions`)
2. Stripe webhooks update entitlements
3. Change middleware: `is_reviewer()` → `has_entitlement()`
4. No dashboard refactor needed

---

## Troubleshooting

**"Not authorized" after login?**
- Check that your `user_id` is in the `reviewers` table

**Charts don't render?**
- Ingestion ran, but `weekly_performance` is empty
- Add the `strategy_return` / `benchmark_return` parsing

**API errors?**
- Verify `.env.local` has correct Supabase URL and keys
- Check RLS policies in Supabase SQL editor
