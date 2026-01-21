import { supabaseServer } from "@/lib/supabase/server";
import DashboardClient from "./ui";

export default async function DashboardPage() {
  const supabase = supabaseServer();

  // Pull latest 200 weeks across shadow+real for your primary version
  const { data: versions } = await supabase
    .from("strategy_versions")
    .select("id, experiment_name, baseline_version, locked_date, benchmark_symbol")
    .order("id", { ascending: false })
    .limit(1);

  const version = versions?.[0];
  if (!version)
    return (
      <div className="p-6 max-w-2xl">
        <h1 className="text-2xl font-semibold">Ready to ingest data</h1>
        <p className="text-gray-600 mt-4 space-y-2">
          <span className="block">
            Once you run a backtest, the dashboard will populate automatically. Expected artifact paths:
          </span>
          <code className="block bg-gray-100 p-3 rounded mt-2 font-mono text-sm">
            data/derived/backtest/bt_weekly.parquet
          </code>
          <code className="block bg-gray-100 p-3 rounded mt-2 font-mono text-sm">
            data/derived/scores_weekly/week_ending=YYYY-MM-DD/scores_weekly.parquet
          </code>
          <span className="block mt-4">
            Once generated, ingest with:
          </span>
          <code className="block bg-gray-100 p-3 rounded mt-2 font-mono text-sm">
            python web/scripts/ingest_week.py --week-ending YYYY-MM-DD --run-type shadow
          </code>
        </p>
      </div>
    );

  const { data: weeks } = await supabase
    .from("weeks")
    .select(
      `
      id, week_ending, week_type, run_type, completeness_pass, universe_type,
      weekly_performance(strategy_return, benchmark_return, active_return, basket_size)
    `
    )
    .eq("strategy_version_id", version.id)
    .order("week_ending", { ascending: true })
    .limit(400);

  return <DashboardClient version={version} weeks={weeks ?? []} />;
}
