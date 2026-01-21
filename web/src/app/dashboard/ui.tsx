"use client";

import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { format } from "date-fns";

type WeekRow = {
  id: number;
  week_ending: string;
  week_type: "CLEAN_TRADE" | "CLEAN_SKIP" | "DATA_IMPAIRED";
  run_type: "shadow" | "real";
  completeness_pass: boolean;
  universe_type: string;
  weekly_performance: {
    strategy_return: number | null;
    benchmark_return: number | null;
    active_return: number | null;
    basket_size: number | null;
  } | null;
};

export default function DashboardClient({
  version,
  weeks,
}: {
  version: any;
  weeks: WeekRow[];
}) {
  const [runType, setRunType] = useState<"shadow" | "real">("shadow");

  const series = useMemo(() => {
    const filtered = weeks
      .filter((w) => w.run_type === runType)
      .filter((w) => w.week_type !== "DATA_IMPAIRED")
      .filter(
        (w) =>
          w.weekly_performance?.strategy_return != null &&
          w.weekly_performance?.benchmark_return != null
      );

    let eqStrategy = 1;
    let eqBench = 1;
    let peakStrategy = 1;

    return filtered.map((w) => {
      const rS = Number(w.weekly_performance!.strategy_return);
      const rB = Number(w.weekly_performance!.benchmark_return);

      eqStrategy *= 1 + rS;
      eqBench *= 1 + rB;
      peakStrategy = Math.max(peakStrategy, eqStrategy);
      const dd = eqStrategy / peakStrategy - 1;

      return {
        date: w.week_ending,
        dateLabel: format(new Date(w.week_ending), "yyyy-MM-dd"),
        equity_strategy: eqStrategy,
        equity_benchmark: eqBench,
        drawdown: dd,
        week_type: w.week_type,
      };
    });
  }, [weeks, runType]);

  const stats = useMemo(() => {
    const total = weeks.filter(
      (w) => w.run_type === runType && w.week_type !== "DATA_IMPAIRED"
    );
    const complete = total.filter((w) => w.completeness_pass);
    const trade = complete.filter((w) => w.week_type === "CLEAN_TRADE");
    const skip = complete.filter((w) => w.week_type === "CLEAN_SKIP");

    const hit = trade.filter(
      (w) => (w.weekly_performance?.strategy_return ?? 0) > 0
    ).length;
    const skipRate = complete.length ? skip.length / complete.length : 0;

    return {
      totalWeeks: total.length,
      completeWeeks: complete.length,
      tradeWeeks: trade.length,
      skipWeeks: skip.length,
      skipRate,
      hitRate: trade.length ? hit / trade.length : 0,
    };
  }, [weeks, runType]);

  return (
    <main className="p-6 space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Market Pressure Scan</h1>
          <p className="text-sm text-gray-600">
            {version.experiment_name} • {version.baseline_version} • locked{" "}
            {version.locked_date ?? "—"} • benchmark {version.benchmark_symbol}
          </p>
        </div>

        <div className="flex gap-2">
          <button
            className={`rounded-md border px-3 py-2 text-sm transition ${
              runType === "shadow"
                ? "font-semibold bg-blue-100 border-blue-600"
                : ""
            }`}
            onClick={() => setRunType("shadow")}
          >
            Shadow
          </button>
          <button
            className={`rounded-md border px-3 py-2 text-sm transition ${
              runType === "real"
                ? "font-semibold bg-blue-100 border-blue-600"
                : ""
            }`}
            onClick={() => setRunType("real")}
          >
            Real
          </button>
        </div>
      </header>

      <section className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <Stat label="Weeks (non-impaired)" value={stats.totalWeeks} />
        <Stat label="Complete weeks" value={stats.completeWeeks} />
        <Stat label="Trade weeks" value={stats.tradeWeeks} />
        <Stat label="Skip weeks" value={stats.skipWeeks} />
        <Stat label="Skip rate" value={`${(stats.skipRate * 100).toFixed(0)}%`} />
      </section>

      <section className="rounded-2xl border p-4">
        <h2 className="font-semibold mb-3">Equity curve vs Benchmark</h2>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series}>
              <XAxis dataKey="dateLabel" hide />
              <YAxis domain={["auto", "auto"]} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="equity_strategy"
                dot={false}
                name="Strategy"
                stroke="#3b82f6"
              />
              <Line
                type="monotone"
                dataKey="equity_benchmark"
                dot={false}
                name="SPY"
                stroke="#10b981"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="rounded-2xl border p-4">
        <h2 className="font-semibold mb-3">Drawdown</h2>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series}>
              <XAxis dataKey="dateLabel" hide />
              <YAxis domain={["auto", 0]} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="drawdown"
                dot={false}
                name="Drawdown"
                stroke="#ef4444"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="rounded-2xl border p-4">
        <h2 className="font-semibold mb-3">Weeks</h2>
        <div className="overflow-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="text-left border-b">
                <th className="p-2 font-medium">Week ending</th>
                <th className="p-2 font-medium">Type</th>
                <th className="p-2 font-medium">Complete</th>
                <th className="p-2 font-medium">Universe</th>
              </tr>
            </thead>
            <tbody>
              {weeks
                .filter((w) => w.run_type === runType)
                .slice()
                .sort((a, b) => a.week_ending.localeCompare(b.week_ending))
                .map((w) => (
                  <tr key={w.id} className="border-t">
                    <td className="p-2">{w.week_ending}</td>
                    <td className="p-2 text-gray-600">{w.week_type}</td>
                    <td className="p-2">
                      {w.completeness_pass ? (
                        <span className="text-green-600 font-medium">YES</span>
                      ) : (
                        <span className="text-red-600 font-medium">NO</span>
                      )}
                    </td>
                    <td className="p-2 text-gray-600">{w.universe_type}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: any }) {
  return (
    <div className="rounded-2xl border p-3">
      <div className="text-xs text-gray-600 font-medium">{label}</div>
      <div className="text-lg font-semibold">{value}</div>
    </div>
  );
}
