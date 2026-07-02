import type {
  Memory, MemoryEdge, EvalRun, ManifestEntry, UsageInfo, DonePayload,
  SleepResult, AdminStats,
} from "./types.ts";

export interface MemoriesResponse {
  memories: Memory[];
  edges: MemoryEdge[];
}

export async function fetchMemories(
  userId: string,
  status?: string,
): Promise<MemoriesResponse> {
  const params = new URLSearchParams({ userId });
  if (status) params.set("status", status);
  const resp = await fetch(`/api/memories?${params}`);
  if (!resp.ok) throw new Error(`fetchMemories: HTTP ${resp.status}`);
  const data = (await resp.json()) as MemoriesResponse;
  return data;
}

export async function fetchEvalRuns(): Promise<EvalRun[]> {
  const resp = await fetch("/api/eval-runs");
  if (!resp.ok) throw new Error(`fetchEvalRuns: HTTP ${resp.status}`);
  const data = (await resp.json()) as { runs: EvalRun[] };
  return data.runs;
}

export async function postSleep(userId: string): Promise<SleepResult> {
  const resp = await fetch("/sleep", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ userId }),
  });
  if (!resp.ok && resp.status !== 409) throw new Error(`postSleep: HTTP ${resp.status}`);
  return (await resp.json()) as SleepResult;
}

export async function fetchStats(userId?: string): Promise<AdminStats> {
  const params = userId ? `?userId=${encodeURIComponent(userId)}` : "";
  const resp = await fetch(`/admin/stats${params}`);
  if (!resp.ok) throw new Error(`fetchStats: HTTP ${resp.status}`);
  return (await resp.json()) as AdminStats;
}

export type SSEEvent =
  | { type: "token";    data: string }
  | { type: "manifest"; data: ManifestEntry[] }
  | { type: "usage";    data: UsageInfo }
  | { type: "done";     data: DonePayload }
  | { type: "error";    data: { error: string } };

async function* parseSSE(
  response: Response,
): AsyncGenerator<{ event: string; raw: string }> {
  const reader = response.body!.getReader();
  const dec = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const blocks = buf.split("\n\n");
    buf = blocks.pop() ?? "";
    for (const block of blocks) {
      if (!block.trim()) continue;
      let event = "message";
      let raw = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        if (line.startsWith("data: "))  raw   = line.slice(6);
      }
      if (raw) yield { event, raw };
    }
  }
}

export async function actStream(
  userId: string,
  sessionId: string,
  input: string,
  onEvent: (e: SSEEvent) => void,
  budget?: number,
): Promise<void> {
  const resp = await fetch("/act", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ userId, sessionId, input, ...(budget ? { budget } : {}) }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }

  for await (const { event, raw } of parseSSE(resp)) {
    if (event === "token") {
      onEvent({ type: "token", data: raw });
    } else if (event === "manifest") {
      onEvent({ type: "manifest", data: JSON.parse(raw) as ManifestEntry[] });
    } else if (event === "usage") {
      onEvent({ type: "usage", data: JSON.parse(raw) as UsageInfo });
    } else if (event === "done") {
      onEvent({ type: "done", data: JSON.parse(raw) as DonePayload });
    } else if (event === "error") {
      onEvent({ type: "error", data: JSON.parse(raw) as { error: string } });
    }
  }
}
