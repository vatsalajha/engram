import { useState } from "react";
import type { ManifestEntry, UsageInfo } from "../types.ts";

interface Props {
  manifest: ManifestEntry[];
  usage?:   UsageInfo;
  /** The budget in force when this turn ran (for the tokens-vs-budget bar). */
  budget?:  number;
}

const TYPE_DOT: Record<string, string> = {
  fact:       "bg-blue-400",
  preference: "bg-purple-400",
  decision:   "bg-orange-400",
  event:      "bg-teal-400",
  skill:      "bg-green-400",
  hypothesis: "bg-fuchsia-400",
  note:       "bg-zinc-500",
};

/** Tiny horizontal confidence bar, 0–1. */
function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const hue = value >= 0.7 ? "bg-success" : value >= 0.4 ? "bg-warn" : "bg-danger";
  return (
    <span className="flex items-center gap-1 w-20 flex-shrink-0" title={`confidence ${pct}%`}>
      <span className="h-1 flex-1 rounded-full bg-border/60 overflow-hidden">
        <span className={`block h-full ${hue}`} style={{ width: `${pct}%` }} />
      </span>
      <span className="font-mono text-[9px] text-muted/70 w-6 text-right">{pct}</span>
    </span>
  );
}

/** "session-3" → "s3"; anything else passes through shortened. */
function bornLabel(session: string | null): string {
  if (!session) return "—";
  const m = /(\d+)$/.exec(session);
  return m ? `s${m[1]}` : session.slice(0, 8);
}

export function ContextManifest({ manifest, usage, budget }: Props) {
  const [open, setOpen] = useState(false);

  const totalTokens = manifest.reduce((s, m) => s + m.tokens, 0);
  const budgetPct = budget ? Math.min(100, Math.round((totalTokens / budget) * 100)) : null;

  return (
    <div className="mt-2 rounded-lg border border-border overflow-hidden text-xs">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-3 py-2 bg-surface
                   hover:bg-border/30 transition-colors text-left"
      >
        <span className="flex items-center gap-2 text-muted">
          <span className="text-accent">⊕</span>
          Provenance —{" "}
          <span className="text-text font-medium">{manifest.length} memories</span>
          <span className="text-border">·</span>
          <span className="font-mono">
            {totalTokens}{budget ? ` / ${budget}` : ""} tok
          </span>
          {budgetPct !== null && (
            <span className="hidden sm:flex items-center w-16 h-1 rounded-full bg-border/60 overflow-hidden">
              <span className="block h-full bg-accent" style={{ width: `${budgetPct}%` }} />
            </span>
          )}
          {usage && (
            <>
              <span className="text-border">·</span>
              <span className="font-mono text-muted">{usage.model}</span>
            </>
          )}
        </span>
        <span className="text-muted">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="divide-y divide-border/50 bg-canvas/50">
          {manifest.length === 0 ? (
            <p className="px-3 py-2 text-muted italic">
              No memories in context — this answer used none of what Engram knows.
            </p>
          ) : (
            manifest.map((m) => (
              <div
                key={m.id}
                className="flex items-center gap-2 px-3 py-1.5 hover:bg-surface/50"
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${TYPE_DOT[m.type] ?? "bg-zinc-500"}`}
                />
                <span className="text-muted w-20 flex-shrink-0">{m.type}</span>
                <ConfidenceBar value={m.confidence ?? 0} />
                <span
                  className="font-mono text-muted/80 flex-shrink-0 w-14 text-center"
                  title={`born in ${m.session_born ?? "unknown session"}`}
                >
                  born {bornLabel(m.session_born)}
                </span>
                <span className="font-mono text-muted flex-shrink-0 w-14 text-right">
                  {m.tokens} tok
                </span>
                {m.compressed && <span className="text-warn">compressed</span>}
                <span className="text-muted/40 font-mono text-[9px] truncate ml-auto">
                  {m.id.slice(0, 8)}
                </span>
              </div>
            ))
          )}
          {usage && (
            <div className="px-3 py-1.5 flex gap-4 text-muted/70 font-mono bg-canvas/30">
              <span>context {totalTokens}</span>
              <span>prompt {usage.promptTokens}</span>
              <span>completion {usage.completionTokens}</span>
              <span>total {usage.totalTokens}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
