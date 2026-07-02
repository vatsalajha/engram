import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { fetchEvalRuns } from "../api.ts";
import type { EvalRun, Arm } from "../types.ts";

const ARM_COLOR: Record<Arm, string> = {
  no_memory:    "#6e7681",
  full_history: "#58a6ff",
  engram:       "#3fb950",
};

const ARM_LABEL: Record<Arm, string> = {
  no_memory:    "No Memory",
  full_history: "Full History",
  engram:       "Engram",
};

interface ChartPoint {
  session: number;
  no_memory?: number;
  full_history?: number;
  engram?: number;
}

function buildSeries(runs: EvalRun[], field: keyof EvalRun): ChartPoint[] {
  const map = new Map<number, ChartPoint>();
  for (const r of runs) {
    const s = r.session_index;
    if (!map.has(s)) map.set(s, { session: s });
    const pt = map.get(s)!;
    const val = r[field];
    if (typeof val === "number") {
      (pt as unknown as Record<string, number>)[r.arm] = val;
    }
  }
  return Array.from(map.values()).sort((a, b) => a.session - b.session);
}

function MetricCard({
  label,
  value,
  unit,
  color = "text-text",
}: {
  label: string;
  value: string | number | null;
  unit?: string;
  color?: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <p className="text-xs text-muted mb-1">{label}</p>
      <p className={`text-2xl font-mono font-medium ${color}`}>
        {value == null ? "—" : value}
        {unit && <span className="text-sm text-muted ml-1">{unit}</span>}
      </p>
    </div>
  );
}

const tooltipStyle = {
  backgroundColor: "#161b22",
  border: "1px solid #30363d",
  borderRadius: "8px",
  color: "#e6edf3",
  fontSize: 11,
};

/** Sweep rows (accuracy-vs-budget) carry metadata { sweep: true, budget: N }. */
function sweepMeta(r: EvalRun): { budget: number } | null {
  const m = r.metadata as { sweep?: boolean; budget?: number } | null;
  return m && m.sweep === true && typeof m.budget === "number" ? { budget: m.budget } : null;
}

export function DashboardView() {
  const [allRuns, setAllRuns] = useState<EvalRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

  useEffect(() => {
    fetchEvalRuns()
      .then(setAllRuns)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);

  // Budget-sweep rows are a separate experiment — keep them out of the
  // per-session charts, average accuracy per budget for the sweep line.
  const sweepRuns = allRuns.filter((r) => sweepMeta(r) !== null);
  const runs      = allRuns.filter((r) => sweepMeta(r) === null);

  const sweepByBudget = new Map<number, number[]>();
  for (const r of sweepRuns) {
    if (r.task_accuracy == null) continue;
    const b = sweepMeta(r)!.budget;
    if (!sweepByBudget.has(b)) sweepByBudget.set(b, []);
    sweepByBudget.get(b)!.push(r.task_accuracy);
  }
  const sweepSeries = Array.from(sweepByBudget.entries())
    .map(([budget, accs]) => ({
      budget,
      accuracy: accs.reduce((s, a) => s + a, 0) / accs.length,
    }))
    .sort((a, b) => a.budget - b.budget);

  const accuracySeries  = buildSeries(runs, "task_accuracy");
  const tokensSeries    = buildSeries(runs, "tokens_in_context");

  const engramRuns = runs.filter((r) => r.arm === "engram");
  // Only sessions where forgetting precision was actually measured (post-change).
  const fpRuns = engramRuns.filter((r) => r.forgetting_precision != null);
  const avgFP = fpRuns.length
    ? fpRuns.reduce((s, r) => s + r.forgetting_precision!, 0) / fpRuns.length
    : null;
  const avgLatency = engramRuns.length
    ? engramRuns.reduce((s, r) => s + (r.latency_p50_ms ?? 0), 0) / engramRuns.length
    : null;
  const totalCost = runs.reduce((s, r) => s + (r.cost_usd ?? 0), 0);
  const lastEngram = engramRuns.at(-1);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted animate-pulse">
        Loading eval runs…
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-2">
        <p className="text-danger text-sm">Failed to load eval data</p>
        <p className="text-muted text-xs font-mono">{error}</p>
        <p className="text-muted/60 text-xs mt-2">
          Run <code className="text-accent">npm run eval</code> to seed benchmark data.
        </p>
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-2 text-center">
        <p className="text-text text-sm">No eval data yet</p>
        <p className="text-muted text-xs">
          Run <code className="font-mono text-accent">npm run eval</code> to seed benchmark results,
          then reload.
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-6 py-6 space-y-8">
      <div>
        <h2 className="text-base font-medium text-text mb-1">Benchmark Results</h2>
        <p className="text-xs text-muted">
          {runs.length} runs · {engramRuns.length} Engram sessions ·{" "}
          {new Set(runs.map((r) => r.session_index)).size} sessions per arm
        </p>
      </div>

      {/* metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          label="Engram accuracy (last)"
          value={lastEngram?.task_accuracy != null ? (lastEngram.task_accuracy * 100).toFixed(1) : null}
          unit="%"
          color="text-success"
        />
        <MetricCard
          label="Forgetting precision (avg)"
          value={avgFP != null ? (avgFP * 100).toFixed(1) : null}
          unit="%"
          color="text-accent"
        />
        <MetricCard
          label="Latency p50 (Engram avg)"
          value={avgLatency != null ? Math.round(avgLatency) : null}
          unit="ms"
        />
        <MetricCard
          label="Total eval cost"
          value={totalCost > 0 ? `$${totalCost.toFixed(4)}` : null}
          color="text-warn"
        />
      </div>

      {/* accuracy chart */}
      <div className="rounded-xl border border-border bg-surface p-5">
        <h3 className="text-sm font-medium text-text mb-1">Task Accuracy vs Session</h3>
        <p className="text-xs text-muted mb-4">
          Engram learns across sessions. No-memory plateaus; full-history grows slowly but costs more.
        </p>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={accuracySeries} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
            <XAxis
              dataKey="session"
              label={{ value: "Session", position: "insideBottom", offset: -2, fill: "#8b949e", fontSize: 11 }}
              tick={{ fill: "#8b949e", fontSize: 11 }}
            />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              tick={{ fill: "#8b949e", fontSize: 11 }}
              width={40}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(v: number, name: string) => [
                `${(v * 100).toFixed(1)}%`,
                ARM_LABEL[name as Arm] ?? name,
              ]}
              labelFormatter={(l: number) => `Session ${l}`}
            />
            <Legend
              formatter={(v: string) => (
                <span style={{ color: "#8b949e", fontSize: 11 }}>
                  {ARM_LABEL[v as Arm] ?? v}
                </span>
              )}
            />
            {(["no_memory", "full_history", "engram"] as Arm[]).map((arm) => (
              <Line
                key={arm}
                type="monotone"
                dataKey={arm}
                stroke={ARM_COLOR[arm]}
                strokeWidth={arm === "engram" ? 2.5 : 1.5}
                dot={{ r: 3, fill: ARM_COLOR[arm] }}
                activeDot={{ r: 5 }}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* tokens chart */}
      <div className="rounded-xl border border-border bg-surface p-5">
        <h3 className="text-sm font-medium text-text mb-1">Tokens in Context vs Session</h3>
        <p className="text-xs text-muted mb-4">
          Full-history balloons to thousands of tokens. Engram stays bounded.
        </p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart
            data={tokensSeries}
            margin={{ top: 4, right: 16, bottom: 4, left: 0 }}
            barCategoryGap="20%"
            barGap={2}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
            <XAxis
              dataKey="session"
              tick={{ fill: "#8b949e", fontSize: 11 }}
              label={{ value: "Session", position: "insideBottom", offset: -2, fill: "#8b949e", fontSize: 11 }}
            />
            <YAxis tick={{ fill: "#8b949e", fontSize: 11 }} width={48} />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(v: number, name: string) => [
                v.toLocaleString(),
                ARM_LABEL[name as Arm] ?? name,
              ]}
              labelFormatter={(l: number) => `Session ${l}`}
            />
            <Legend
              formatter={(v: string) => (
                <span style={{ color: "#8b949e", fontSize: 11 }}>
                  {ARM_LABEL[v as Arm] ?? v}
                </span>
              )}
            />
            {(["no_memory", "full_history", "engram"] as Arm[]).map((arm) => (
              <Bar key={arm} dataKey={arm} fill={ARM_COLOR[arm]} radius={[3, 3, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* accuracy vs budget sweep — the graceful-degradation chart */}
      <div className="rounded-xl border border-border bg-surface p-5">
        <h3 className="text-sm font-medium text-text mb-1">Accuracy vs Context Budget</h3>
        <p className="text-xs text-muted mb-4">
          Engram degrades gracefully as the context window shrinks — quality bends,
          it doesn't break.
        </p>
        {sweepSeries.length === 0 ? (
          <p className="text-xs text-muted/70 italic py-8 text-center">
            No sweep data yet — run{" "}
            <code className="font-mono text-accent">npm run eval -- --sweep</code>{" "}
            to chart accuracy across budgets 300 → 2500.
          </p>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={sweepSeries} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
              <XAxis
                dataKey="budget"
                type="number"
                domain={["dataMin", "dataMax"]}
                tick={{ fill: "#8b949e", fontSize: 11 }}
                label={{ value: "Context budget (tokens)", position: "insideBottom", offset: -2, fill: "#8b949e", fontSize: 11 }}
              />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                tick={{ fill: "#8b949e", fontSize: 11 }}
                width={40}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, "accuracy"]}
                labelFormatter={(l: number) => `budget ${l} tok`}
              />
              <Line
                type="monotone"
                dataKey="accuracy"
                stroke="#3fb950"
                strokeWidth={2.5}
                dot={{ r: 4, fill: "#3fb950" }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* raw data table */}
      <div className="rounded-xl border border-border bg-surface overflow-hidden">
        <div className="px-4 py-3 border-b border-border">
          <h3 className="text-sm font-medium text-text">Raw Eval Runs</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-muted font-mono">
            <thead className="bg-canvas/50">
              <tr>
                {["Arm", "Session", "Accuracy", "Tokens", "Recall@k", "FP", "Latency p50", "Cost"].map((h) => (
                  <th key={h} className="px-3 py-2 text-left text-muted/70 font-medium">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/30">
              {runs.map((r) => (
                <tr key={r.id} className="hover:bg-surface/60">
                  <td className="px-3 py-1.5" style={{ color: ARM_COLOR[r.arm] }}>
                    {ARM_LABEL[r.arm]}
                  </td>
                  <td className="px-3 py-1.5">{r.session_index}</td>
                  <td className="px-3 py-1.5">
                    {r.task_accuracy != null ? `${(r.task_accuracy * 100).toFixed(1)}%` : "—"}
                  </td>
                  <td className="px-3 py-1.5">
                    {r.tokens_in_context?.toLocaleString() ?? "—"}
                  </td>
                  <td className="px-3 py-1.5">
                    {r.recall_at_k != null ? r.recall_at_k.toFixed(2) : "—"}
                  </td>
                  <td className="px-3 py-1.5">
                    {r.forgetting_precision != null ? r.forgetting_precision.toFixed(2) : "—"}
                  </td>
                  <td className="px-3 py-1.5">
                    {r.latency_p50_ms != null ? `${Math.round(r.latency_p50_ms)}ms` : "—"}
                  </td>
                  <td className="px-3 py-1.5">
                    {r.cost_usd != null ? `$${r.cost_usd.toFixed(5)}` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
