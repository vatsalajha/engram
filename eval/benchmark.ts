/**
 * eval/benchmark.ts
 *
 * Deterministic synthetic benchmark for the 3-arm Engram memory evaluation
 * (SPEC §6). Seedable: generateBenchmark(seed, nSessions) reproduces identically.
 *
 * Shape: 8 sessions · 12 stable facts/preferences · 3 preferences that FLIP in
 * sessions 4–5 · 5 decision tasks per session (40 total), each with gold
 * keywords; changed-preference tasks are tagged testsChangedPreference.
 *
 * User profile: Jordan Lee, backend engineer at VertexCore AI.
 *
 * Stable facts (sessions 1–3):
 *   identity/employer · MLOps product · Go backend · Alibaba cn-beijing ·
 *   CockroachDB · project Andromeda · 300k executions/day · React+TS frontend ·
 *   GitHub Actions CI · Grafana+Prometheus · team of 6 · Keycloak OIDC
 *
 * Flipped preferences:
 *   PREF_A  editor:     Helix (s1)        → Zed (s4)
 *   PREF_B  messaging:  NATS JetStream (s2) → Apache Pulsar (s5)
 *   PREF_C  API style:  REST/OpenAPI (s3) → gRPC (s5)
 *
 * Scoring helpers (used offline and as the fallback when flash grading fails):
 *   accuracy(task, answer)            = fraction of goldKeywords present (0–1)
 *   forgettingPrecision(task, answer) = 1 clean · 0 stale leak · 0.5 unknown · null non-FP
 */

// ─── Simple seeded PRNG (mulberry32) ─────────────────────────────────────────

function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ─── Types ────────────────────────────────────────────────────────────────────

export type Arm = "no_memory" | "full_history" | "engram";

export interface Task {
  id: string;
  /** Question posed to the agent. */
  question: string;
  /** Keywords that must appear in the answer for full accuracy credit. */
  goldKeywords: string[];
  /** True if this task measures forgetting precision (changed preference test). */
  testsChangedPreference: boolean;
  /** Which changed-preference scenario this belongs to. */
  changedPrefId?: "PREF_A" | "PREF_B" | "PREF_C";
  /** For FP tasks: keywords from the OLD, superseded preference (must NOT appear). */
  oldKeywords?: string[];
  /** For FP tasks: keywords the NEW preference introduces (≥1 must appear). */
  newKeywords?: string[];
}

export interface BenchmarkSession {
  index: number; // 1-based
  /**
   * Statements Jordan makes during this session.
   * → Engram arm: extractMemories + writeMemories.
   * → full_history arm: appended to the transcript verbatim.
   * → no_memory arm: ignored.
   */
  prologues: string[];
  /** Decision tasks Jordan poses at the end of this session (5 each). */
  tasks: Task[];
}

// ─── Task helpers ─────────────────────────────────────────────────────────────

const stable = (id: string, question: string, goldKeywords: string[]): Task => ({
  id, question, goldKeywords, testsChangedPreference: false,
});

const fp = (
  id: string,
  question: string,
  pref: "PREF_A" | "PREF_B" | "PREF_C",
): Task => {
  const table = {
    PREF_A: { gold: ["zed"], old: ["helix"], new: ["zed"] },
    PREF_B: { gold: ["pulsar"], old: ["nats", "jetstream"], new: ["pulsar"] },
    PREF_C: { gold: ["grpc"], old: ["rest", "openapi"], new: ["grpc"] },
  }[pref];
  return {
    id, question,
    goldKeywords: table.gold,
    testsChangedPreference: true,
    changedPrefId: pref,
    oldKeywords: table.old,
    newKeywords: table.new,
  };
};

// ─── The 8 sessions ───────────────────────────────────────────────────────────

const ALL_SESSIONS: BenchmarkSession[] = [
  // ── Session 1: identity + core stack; PREF_A initial ──────────────────────
  {
    index: 1,
    prologues: [
      "I'm Jordan, a backend engineer at VertexCore AI. We build enterprise MLOps tooling.",
      "Our backend services are written primarily in Go.",
      "Our services run on Alibaba Cloud ECS in the cn-beijing region.",
      "I use the Helix editor for all my development work — it's my daily driver.",
    ],
    tasks: [
      stable("s1t1", "Who am I and where do I work?", ["jordan", "vertexcore"]),
      stable("s1t2", "What kind of product does VertexCore build?", ["mlops"]),
      stable("s1t3", "What programming language should a new VertexCore backend service use?", ["go"]),
      stable("s1t4", "What cloud provider and region do we deploy to?", ["alibaba", "cn-beijing"]),
      stable("s1t5", "Which editor should be installed on my new workstation?", ["helix"]), // pre-flip: helix IS correct
    ],
  },

  // ── Session 2: data layer + project; PREF_B initial ───────────────────────
  {
    index: 2,
    prologues: [
      "We use CockroachDB as our primary transactional database.",
      "The project I'm leading is called Andromeda — a distributed workflow orchestration engine.",
      "Andromeda handles over 300,000 workflow executions per day.",
      "For messaging between services we use NATS JetStream.",
    ],
    tasks: [
      stable("s2t1", "What transactional database does VertexCore rely on?", ["cockroachdb"]),
      stable("s2t2", "What project am I leading and what does it do?", ["andromeda", "workflow"]),
      stable("s2t3", "Roughly how many workflow executions does Andromeda handle daily?", ["300,000"]),
      stable("s2t4", "What message broker should a new internal service use?", ["nats"]), // pre-flip
      stable("s2t5", "Remind me which language our backend is written in.", ["go"]),
    ],
  },

  // ── Session 3: tooling + team; PREF_C initial ──────────────────────────────
  {
    index: 3,
    prologues: [
      "Our CI runs on GitHub Actions — every PR gets typecheck and tests.",
      "For observability we use Grafana dashboards backed by Prometheus.",
      "My team has 6 engineers including me.",
      "Our public APIs are REST with OpenAPI specs — that's our standard.",
      "The Andromeda dashboard frontend is React with TypeScript.",
    ],
    tasks: [
      stable("s3t1", "Where does our CI run?", ["github actions"]),
      stable("s3t2", "What's our observability stack?", ["grafana", "prometheus"]),
      stable("s3t3", "How many engineers are on my team?", ["6"]),
      stable("s3t4", "What API style should a new public endpoint follow?", ["rest"]), // pre-flip
      stable("s3t5", "What's the Andromeda dashboard frontend built with?", ["react", "typescript"]),
    ],
  },

  // ── Session 4: PREF_A FLIPS (Helix → Zed) ──────────────────────────────────
  {
    index: 4,
    prologues: [
      "I've moved away from Helix. My new editor is Zed — I switched to Zed and it's now my primary development environment.",
      "We're integrating single sign-on using Keycloak with OIDC.",
    ],
    tasks: [
      fp("s4t1", "What editor should I configure for my development setup?", "PREF_A"),
      stable("s4t2", "What auth system are we integrating for SSO?", ["keycloak"]),
      stable("s4t3", "What database sits under Andromeda?", ["cockroachdb"]),
      stable("s4t4", "Which cloud region are we in?", ["cn-beijing"]),
      stable("s4t5", "What messaging system do our services use?", ["nats"]), // PREF_B not flipped yet
    ],
  },

  // ── Session 5: PREF_B and PREF_C FLIP ──────────────────────────────────────
  {
    index: 5,
    prologues: [
      "We have migrated our messaging layer from NATS JetStream to Apache Pulsar. Pulsar is now the standard for all service-to-service communication.",
      "Also, new architectural decision: internal APIs are now gRPC instead of REST. gRPC is the standard going forward.",
    ],
    tasks: [
      fp("s5t1", "What message broker should I use for a new VertexCore service?", "PREF_B"),
      fp("s5t2", "What API style should new internal services expose?", "PREF_C"),
      fp("s5t3", "Which code editor is my daily driver these days?", "PREF_A"),
      stable("s5t4", "What scale does Andromeda operate at daily?", ["300,000"]),
      stable("s5t5", "Where does our CI pipeline run?", ["github actions"]),
    ],
  },

  // ── Session 6: recall under interference ───────────────────────────────────
  {
    index: 6,
    prologues: [
      "Busy week — mostly reviews and Andromeda v2 planning.",
    ],
    tasks: [
      fp("s6t1", "What editor does Jordan code in day-to-day?", "PREF_A"),
      fp("s6t2", "What event-streaming platform does the team use now?", "PREF_B"),
      fp("s6t3", "A teammate asks: REST or something else for the new internal API?", "PREF_C"),
      stable("s6t4", "How big is my team?", ["6"]),
      stable("s6t5", "What do we use for dashboards and metrics?", ["grafana", "prometheus"]),
    ],
  },

  // ── Session 7: synthesis tasks ──────────────────────────────────────────────
  {
    index: 7,
    prologues: [],
    tasks: [
      stable("s7t1", "Summarize VertexCore's core backend stack in one line.", ["go", "cockroachdb", "alibaba"]),
      fp("s7t2", "Set up a new laptop for me — which editor gets installed?", "PREF_A"),
      fp("s7t3", "Which messaging technology is in production right now?", "PREF_B"),
      fp("s7t4", "Write the tech-radar entry: what's our API standard?", "PREF_C"),
      stable("s7t5", "What project am I leading?", ["andromeda"]),
    ],
  },

  // ── Session 8: final recall across everything ──────────────────────────────
  {
    index: 8,
    prologues: [],
    tasks: [
      stable("s8t1", "What cloud region does VertexCore operate in?", ["cn-beijing"]),
      fp("s8t2", "What is my preferred editor?", "PREF_A"),
      fp("s8t3", "What streaming platform underpins inter-service messaging?", "PREF_B"),
      fp("s8t4", "New endpoint spec: which API style do we standardize on?", "PREF_C"),
      stable("s8t5", "What's the frontend stack for the Andromeda dashboard?", ["react", "typescript"]),
    ],
  },
];

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Return a reproducible benchmark for the given seed and session count.
 * seed=0, nSessions=8 is the canonical run. Core facts, flips, and tasks are
 * invariant; the seed only perturbs neutral phrasing (reserved for future use)
 * so results remain comparable across seeds.
 */
export function generateBenchmark(
  seed: number = 0,
  nSessions: number = 8,
): BenchmarkSession[] {
  const rng = mulberry32(seed);
  for (let i = 0; i < 10; i++) rng(); // burn-in so seeds diverge if ever used

  return ALL_SESSIONS.slice(0, Math.min(nSessions, ALL_SESSIONS.length));
}

// ─── Scoring helpers ──────────────────────────────────────────────────────────

/** Lowercase + strip commas so "300,000" matches "300000". */
const norm = (s: string) => s.toLowerCase().replace(/,/g, "");

/** Fraction of goldKeywords present in the answer (case-insensitive), 0–1. */
export function scoreAccuracy(task: Task, answer: string): number {
  const a = norm(answer);
  const hits = task.goldKeywords.filter((kw) => a.includes(norm(kw)));
  return hits.length / task.goldKeywords.length;
}

/**
 * Forgetting precision for a changed-preference task.
 *   1    – answer contains ≥1 newKeyword AND zero oldKeywords (clean forget)
 *   0    – answer contains ≥1 oldKeyword (stale memory leaked)
 *   0.5  – answer contains neither (agent said "I don't know")
 *   null – not a changed-preference task
 */
export function scoreForgettingPrecision(task: Task, answer: string): number | null {
  if (!task.testsChangedPreference || !task.oldKeywords || !task.newKeywords) return null;

  const a = norm(answer);
  const hasOld = task.oldKeywords.some((kw) => a.includes(norm(kw)));
  const hasNew = task.newKeywords.some((kw) => a.includes(norm(kw)));

  if (hasOld) return 0;       // stale preference leaked
  if (hasNew) return 1;       // clean — only new preference present
  return 0.5;                 // neither mentioned (unknown response)
}

export type { BenchmarkSession as Session };
