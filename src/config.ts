import "dotenv/config"; // loads .env in development; no-op when vars already set
import { z } from "zod";

const ConfigSchema = z.object({
  DASHSCOPE_API_KEY: z.string().min(1),
  ENGRAM_QWEN_BASE_URL: z
    .string()
    .url()
    .default("https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
  // Optional: set to upgrade rate-limit/idempotency/locks from in-process to Redis
  REDIS_URL: z.string().optional(),
  ENGRAM_DB_PATH: z.string().default("./data/engram.db"),
  ENGRAM_CONTEXT_BUDGET: z.coerce.number().int().positive().default(1500),
  PORT: z.coerce.number().int().positive().default(3000),
  NODE_ENV: z
    .enum(["development", "production", "test"])
    .default("development"),
  ENGRAM_RERANK_ENABLED: z
    .string()
    .transform((v) => v === "true")
    .default("true"),
  ENGRAM_SLEEP_INTERVAL_MIN: z.coerce.number().int().positive().default(10),
  // "on" enables the background scheduler; anything else disables it
  ENGRAM_SCHEDULER: z.string().default("off"),
  // Write path
  ENGRAM_EXTRACT_MODEL: z.string().default("qwen-plus"),
  // Cosine similarity threshold for semantic dedup (1 - distance).
  // Candidates with similarity >= this value to an existing active memory are skipped.
  ENGRAM_DEDUPE_THRESHOLD: z.coerce.number().min(0).max(1).default(0.95),
  // Read / retrieval path
  // k constant in RRF: score = Σ 1/(RRF_K + rank_i). Classic default: 60.
  ENGRAM_RRF_K: z.coerce.number().int().positive().default(60),
  // How many candidates from FTS and vec individually before fusion (each leg).
  ENGRAM_RETRIEVE_LEG_K: z.coerce.number().int().positive().default(20),
  // Max candidates sent to the reranker (trim after fusion, before rerank).
  ENGRAM_RERANK_SHORTLIST: z.coerce.number().int().positive().default(16),
  ENGRAM_RERANK_MODEL: z.string().default("qwen-flash"),
  // ── Scoring / utility function (SPEC §4.2 — confidence is first-class in v2)
  // utility = w_rel·relevance + w_sal·salience + w_con·confidence + w_rec·recency + w_typ·typePriority
  ENGRAM_SCORE_W_REL: z.coerce.number().min(0).max(1).default(0.45),
  ENGRAM_SCORE_W_SAL: z.coerce.number().min(0).max(1).default(0.20),
  ENGRAM_SCORE_W_CON: z.coerce.number().min(0).max(1).default(0.15),
  ENGRAM_SCORE_W_REC: z.coerce.number().min(0).max(1).default(0.10),
  ENGRAM_SCORE_W_TYP: z.coerce.number().min(0).max(1).default(0.10),
  // Packer: fraction of the budget reserved for pinned + preference/decision
  ENGRAM_PACK_RESERVED_FRACTION: z.coerce.number().min(0).max(1).default(0.25),
  // λ in recency(ageDays) = 1 / (1 + ageDays * λ)
  ENGRAM_RECENCY_LAMBDA: z.coerce.number().positive().default(0.1),
  // ── Belief engine ─────────────────────────────────────────────────────────
  // η: confidence step per confirmed/contradicted signal (SPEC §4.4)
  ENGRAM_BELIEF_ETA: z.coerce.number().min(0).max(1).default(0.15),
  // confidence floor: below this, hypotheses auto-archive, facts flag supersession
  ENGRAM_CONFIDENCE_FLOOR: z.coerce.number().min(0).max(1).default(0.25),
  // ── Retention / forgetting weights (SPEC §4.5 — 4 terms in v2) ────────────
  // retention = w_sal·sal + w_con·conf + w_rec·recency + w_acc·norm(log(1+access))
  ENGRAM_RETENTION_W_SAL: z.coerce.number().min(0).max(1).default(0.35),
  ENGRAM_RETENTION_W_CON: z.coerce.number().min(0).max(1).default(0.25),
  ENGRAM_RETENTION_W_REC: z.coerce.number().min(0).max(1).default(0.25),
  ENGRAM_RETENTION_W_ACC: z.coerce.number().min(0).max(1).default(0.15),
  // ── Sleep cycle: forgetting ───────────────────────────────────────────────
  // retention score below this → status='archived'
  ENGRAM_ARCHIVE_THRESHOLD: z.coerce.number().min(0).max(1).default(0.3),
  // archived and untouched for this many hours → status='expired'
  ENGRAM_EXPIRE_ARCHIVED_HOURS: z.coerce.number().positive().default(48),
  // ── Sleep cycle: consolidation ────────────────────────────────────────────
  // minimum cluster size to trigger episodic consolidation
  ENGRAM_CONSOLIDATE_MIN: z.coerce.number().int().positive().default(3),
  // k for the ANN search used during clustering
  ENGRAM_CONSOLIDATE_K: z.coerce.number().int().positive().default(10),
  // pairwise cosine similarity required for cluster membership
  ENGRAM_CONSOLIDATE_SIM: z.coerce.number().min(0).max(1).default(0.82),
  // ── Sleep cycle: inference ("the agent dreams") ───────────────────────────
  // max hypotheses proposed per sleep run
  ENGRAM_INFER_MAX: z.coerce.number().int().positive().default(3),
  // skip a proposed hypothesis if an existing memory is this similar
  ENGRAM_INFER_DEDUPE_SIM: z.coerce.number().min(0).max(1).default(0.9),
  // confidence assigned to newborn hypotheses (they must earn the rest)
  ENGRAM_HYPOTHESIS_CONFIDENCE: z.coerce.number().min(0).max(1).default(0.4),
  // how many recent episodic turns feed the inference prompt
  ENGRAM_INFER_EPISODIC_N: z.coerce.number().int().positive().default(40),
  // ── Rate limiting ─────────────────────────────────────────────────────────
  // Max requests per window per user (sliding window counter in Redis)
  ENGRAM_RATE_LIMIT_MAX: z.coerce.number().int().positive().default(60),
  // Window size for rate limiting (seconds)
  ENGRAM_RATE_LIMIT_WINDOW_SECONDS: z.coerce.number().int().positive().default(60),
  // ── Idempotency ───────────────────────────────────────────────────────────
  // How long to remember an Idempotency-Key before accepting it again (seconds)
  ENGRAM_IDEMPOTENCY_TTL_SECONDS: z.coerce.number().int().positive().default(86400),
});

export type Config = z.infer<typeof ConfigSchema>;

function loadConfig(): Config {
  const result = ConfigSchema.safeParse(process.env);
  if (!result.success) {
    console.error("Invalid configuration:", result.error.format());
    process.exit(1);
  }
  return result.data;
}

export const config = loadConfig();
