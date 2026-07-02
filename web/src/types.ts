export type MemoryType =
  | "fact" | "preference" | "decision" | "event" | "skill" | "hypothesis" | "note";
export type MemoryStatus = "active" | "archived" | "superseded" | "expired";
export type EdgeKind = "supersedes" | "derived_from" | "supports" | "contradicts";

export interface Memory {
  id: string;
  type: MemoryType;
  status: MemoryStatus;
  content: string;
  salience: number;
  confidence: number;
  pinned: boolean;
  needs_review: boolean;
  tags: string[];
  created_at: string;
  access_count: number;
  session_id: string | null;
  superseded_by: string | null;
}

export interface MemoryEdge {
  src_id: string;
  dst_id: string;
  kind: EdgeKind;
}

export interface ManifestEntry {
  id: string;
  type: string;
  tokens: number;
  utility: number;
  confidence: number | null;
  session_born: string | null;
  compressed?: boolean;
}

export interface UsageInfo {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  contextTokens: number;
  model: string;
}

export interface DonePayload {
  turnId: string;
  manifest: ManifestEntry[];
  usage: UsageInfo;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  turnId?: string;
  manifest?: ManifestEntry[];
  usage?: UsageInfo;
}

export interface SleepResult {
  id: string | null;
  user_id: string;
  consolidated_n: number;
  decayed_n: number;
  inferred_n: number;
  skipped: boolean;
}

export interface AdminStats {
  estimatedCostUSD: number;
  tokenUsage: { totalTokens: number; calls: number };
}

export type Arm = "no_memory" | "full_history" | "engram";

export interface EvalRun {
  id: string;
  run_at: string;
  arm: Arm;
  session_index: number;
  task_accuracy: number | null;
  tokens_in_context: number | null;
  recall_at_k: number | null;
  forgetting_precision: number | null;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  cost_usd: number | null;
  metadata: unknown;
}
