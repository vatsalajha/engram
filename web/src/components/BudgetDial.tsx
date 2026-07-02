import { useEffect, useState } from "react";
import { fetchStats } from "../api.ts";

interface Props {
  budget: number;
  onChange: (budget: number) => void;
  disabled?: boolean;
}

/**
 * The live budget dial (SPEC: "drag 2000 → 300 and watch quality degrade
 * gracefully") + always-visible cumulative Qwen cost from /admin/stats.
 */
export function BudgetDial({ budget, onChange, disabled }: Props) {
  const [costUSD, setCostUSD] = useState<number | null>(null);
  const [totalTokens, setTotalTokens] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const stats = await fetchStats();
        if (!cancelled) {
          setCostUSD(stats.estimatedCostUSD ?? 0);
          setTotalTokens(stats.tokenUsage?.totalTokens ?? 0);
        }
      } catch { /* stats are decorative — never block the demo */ }
    };
    void poll();
    const t = setInterval(() => { void poll(); }, 5000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  // Colour shifts as the budget tightens — visual cue for the squeeze.
  const tone = budget <= 400 ? "text-danger" : budget <= 900 ? "text-warn" : "text-accent";

  return (
    <div className="flex items-center gap-3 min-w-0">
      <span className="text-[10px] uppercase tracking-widest text-muted/70 flex-shrink-0">
        budget
      </span>
      <input
        type="range"
        min={150}
        max={2500}
        step={50}
        value={budget}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.currentTarget.value))}
        className="w-36 accent-current cursor-pointer disabled:opacity-40"
        aria-label="Context token budget"
      />
      <span className={`font-mono text-xs ${tone} w-14 flex-shrink-0`}>
        {budget} tok
      </span>
      <span className="text-border flex-shrink-0">·</span>
      <span className="font-mono text-[11px] text-muted flex-shrink-0" title="cumulative Qwen spend (estimated)">
        {costUSD === null ? "—" : `$${costUSD.toFixed(4)}`}
        {totalTokens !== null && (
          <span className="text-muted/50"> / {totalTokens.toLocaleString()} tok</span>
        )}
      </span>
    </div>
  );
}
