import { useState, useEffect, useRef } from "react";
import { fetchMemories } from "../api.ts";
import type { Memory, MemoryEdge } from "../types.ts";

/** How long a superseded memory stays visible (struck-through) before dropping. */
const SUPERSEDED_LINGER_MS = 10_000;

export interface UseMemoriesReturn {
  active:     Memory[];
  /** Recently-superseded memories, each visible for ~10s (the forgetting moment). */
  superseded: Memory[];
  edges:      MemoryEdge[];
  loading:    boolean;
  lastUpdated: Date | null;
}

export function useMemories(
  userId: string,
  pollMs = 2000,
): UseMemoriesReturn {
  const [active, setActive]         = useState<Memory[]>([]);
  const [superseded, setSuperseded] = useState<Memory[]>([]);
  const [edges, setEdges]           = useState<MemoryEdge[]>([]);
  const [loading, setLoading]       = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // id → epoch ms when we FIRST saw it superseded; drives the 10s linger.
  const supersededSeenRef = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    if (!userId) return;

    let cancelled = false;

    const poll = async () => {
      try {
        const { memories, edges: fetchedEdges } = await fetchMemories(userId);
        if (cancelled) return;

        const now = Date.now();
        const seen = supersededSeenRef.current;

        const supersededNow = memories.filter((m) => m.status === "superseded");
        for (const m of supersededNow) {
          if (!seen.has(m.id)) seen.set(m.id, now);
        }
        // Show only those still inside the linger window.
        const lingering = supersededNow.filter(
          (m) => now - (seen.get(m.id) ?? 0) < SUPERSEDED_LINGER_MS,
        );

        setActive(memories.filter((m) => m.status === "active"));
        setSuperseded(lingering);
        setEdges(fetchedEdges ?? []);
        setLastUpdated(new Date());
        setLoading(false);
      } catch {
        if (!cancelled) setLoading(false);
      }
    };

    void poll();
    timerRef.current = setInterval(() => { void poll(); }, pollMs);

    return () => {
      cancelled = true;
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [userId, pollMs]);

  return { active, superseded, edges, loading, lastUpdated };
}
