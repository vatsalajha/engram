import { useState } from "react";
import { useMemories } from "../hooks/useMemories.ts";
import { MemoryBadge } from "./MemoryBadge.tsx";
import type { Memory, MemoryEdge, MemoryType } from "../types.ts";

interface Props {
  userId: string;
  /**
   * Epoch ms of the last sleep run (or null). Memories created after this
   * get a brief highlight — the on-camera "it dreamed something" moment.
   */
  highlightSince?: number | null;
}

/** Display order: what the packer prioritises first, hypotheses near the end. */
const TYPE_ORDER: MemoryType[] = [
  "preference", "decision", "fact", "skill", "hypothesis", "event", "note",
];

const EDGE_LABEL: Record<string, string> = {
  supersedes:   "supersedes",
  derived_from: "derived from",
  supports:     "supports",
  contradicts:  "contradicts",
};

function EdgePopover({
  memory, edges, all, onClose,
}: {
  memory: Memory;
  edges: MemoryEdge[];
  all: Map<string, Memory>;
  onClose: () => void;
}) {
  const related = edges.filter((e) => e.src_id === memory.id || e.dst_id === memory.id);
  return (
    <div
      className="absolute right-2 z-20 mt-1 w-64 rounded-lg border border-border bg-surface
                 shadow-xl shadow-black/40 p-2 text-[11px]"
      onMouseLeave={onClose}
    >
      <p className="text-muted/70 uppercase tracking-widest text-[9px] mb-1.5">memory graph</p>
      {related.length === 0 ? (
        <p className="text-muted italic">No edges yet — this memory stands alone.</p>
      ) : (
        <ul className="space-y-1.5">
          {related.map((e, i) => {
            const outgoing = e.src_id === memory.id;
            const otherId = outgoing ? e.dst_id : e.src_id;
            const other = all.get(otherId);
            return (
              <li key={i} className="flex items-start gap-1.5">
                <span className={`font-mono flex-shrink-0 ${outgoing ? "text-accent" : "text-warn"}`}>
                  {outgoing ? "→" : "←"}
                </span>
                <span>
                  <span className="text-muted">{EDGE_LABEL[e.kind] ?? e.kind}</span>{" "}
                  <span className="text-text/90">
                    {other
                      ? (other.content.length > 70 ? other.content.slice(0, 70) + "…" : other.content)
                      : otherId.slice(0, 10) + "…"}
                  </span>
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function MemoryRow({
  memory, superseded = false, highlighted = false, edgeCount, onGraphClick, popover,
}: {
  memory: Memory;
  superseded?: boolean;
  highlighted?: boolean;
  edgeCount: number;
  onGraphClick: () => void;
  popover: React.ReactNode;
}) {
  const isHypothesis = memory.type === "hypothesis";
  return (
    <div
      className={`relative px-3 py-2 border-b border-border/40 text-xs animate-slide-in
        ${superseded ? "opacity-60" : "hover:bg-surface/60"}
        ${isHypothesis && !superseded ? "border-l-2 border-l-fuchsia-500/50" : ""}
        ${highlighted ? "bg-fuchsia-500/10 ring-1 ring-inset ring-fuchsia-500/40" : ""}`}
    >
      <div className="flex items-start gap-2">
        <MemoryBadge type={memory.type} />
        <p
          className={`flex-1 leading-snug ${superseded ? "line-through text-muted" : "text-text"}`}
        >
          {memory.content.length > 120
            ? memory.content.slice(0, 120) + "…"
            : memory.content}
        </p>
      </div>
      <div className="mt-1 flex items-center gap-3 text-muted/60 font-mono pl-0.5">
        <span title="salience">sal {memory.salience.toFixed(2)}</span>
        <span
          title="confidence"
          className={memory.confidence < 0.4 ? "text-danger/80" : memory.confidence >= 0.7 ? "text-success/80" : ""}
        >
          conf {memory.confidence.toFixed(2)}
        </span>
        {memory.session_id && (
          <span title={`born in ${memory.session_id}`}>
            born {/(\d+)$/.exec(memory.session_id)?.[1] ? `s${/(\d+)$/.exec(memory.session_id)![1]}` : memory.session_id.slice(0, 6)}
          </span>
        )}
        {memory.tags.includes("inferred") && (
          <span className="text-fuchsia-400/90 not-italic" title="proposed by the sleep cycle">inferred</span>
        )}
        {memory.needs_review && <span className="text-warn" title="belief floor hit">review</span>}
        {superseded && <span className="text-danger/70">superseded</span>}
        {memory.pinned && <span className="text-warn">📌</span>}
        <button
          onClick={onGraphClick}
          className={`ml-auto hover:text-accent transition-colors ${edgeCount > 0 ? "text-accent/70" : "text-muted/40"}`}
          title={`${edgeCount} graph edge${edgeCount === 1 ? "" : "s"}`}
        >
          ⌬ {edgeCount > 0 ? edgeCount : ""}
        </button>
      </div>
      {popover}
    </div>
  );
}

export function MemoryPanel({ userId, highlightSince }: Props) {
  const { active, superseded, edges, loading, lastUpdated } = useMemories(userId);
  const [popoverId, setPopoverId] = useState<string | null>(null);

  const allById = new Map<string, Memory>([...active, ...superseded].map((m) => [m.id, m]));
  const edgeCount = (id: string) =>
    edges.filter((e) => e.src_id === id || e.dst_id === id).length;

  const isHighlighted = (m: Memory) =>
    !!highlightSince && new Date(m.created_at).getTime() >= highlightSince;

  const groups = TYPE_ORDER
    .map((t) => ({ type: t, items: active.filter((m) => m.type === t) }))
    .filter((g) => g.items.length > 0);

  const row = (m: Memory, sup = false) => (
    <MemoryRow
      key={m.id}
      memory={m}
      superseded={sup}
      highlighted={!sup && isHighlighted(m)}
      edgeCount={edgeCount(m.id)}
      onGraphClick={() => setPopoverId((p) => (p === m.id ? null : m.id))}
      popover={
        popoverId === m.id ? (
          <EdgePopover
            memory={m}
            edges={edges}
            all={allById}
            onClose={() => setPopoverId(null)}
          />
        ) : null
      }
    />
  );

  return (
    <aside className="w-80 flex-shrink-0 flex flex-col border-l border-border bg-canvas">
      {/* header */}
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <h2 className="text-sm font-medium text-text">Memory</h2>
        <div className="flex items-center gap-2">
          <span
            className={`w-1.5 h-1.5 rounded-full ${loading ? "bg-warn animate-pulse" : "bg-success"}`}
          />
          <span className="text-[10px] text-muted font-mono">
            {active.length} active
          </span>
        </div>
      </div>

      {/* memory list */}
      <div className="flex-1 overflow-y-auto">
        {loading && active.length === 0 ? (
          <p className="px-4 py-6 text-xs text-muted italic text-center">
            Polling memories…
          </p>
        ) : active.length === 0 && superseded.length === 0 ? (
          <p className="px-4 py-6 text-xs text-muted italic text-center">
            No memories yet. Start chatting to build context.
          </p>
        ) : (
          <>
            {groups.map((g) => (
              <div key={g.type}>
                <p className="px-3 py-1.5 text-[10px] uppercase tracking-widest text-muted/60 bg-surface/30">
                  {g.type} · {g.items.length}
                </p>
                {g.items.map((m) => row(m))}
              </div>
            ))}

            {superseded.length > 0 && (
              <div>
                <p className="px-3 py-1.5 text-[10px] uppercase tracking-widest text-danger/50 bg-surface/30">
                  forgetting…
                </p>
                {superseded.map((m) => row(m, true))}
              </div>
            )}
          </>
        )}
      </div>

      {/* footer */}
      {lastUpdated && (
        <div className="px-3 py-2 border-t border-border">
          <p className="text-[10px] text-muted/50 font-mono">
            updated {lastUpdated.toLocaleTimeString()}
          </p>
        </div>
      )}
    </aside>
  );
}
