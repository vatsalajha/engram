/**
 * eval/run.ts — the 3-arm benchmark runner (SPEC §6).
 *
 *   A  no_memory     current turn only
 *   B  full_history  every prior turn stuffed into context (16k-token cap;
 *                    truncation events are counted and reported)
 *   C  engram        hybrid recall → budget-packed context (1500 tokens)
 *
 * Grading: qwen-flash strict-JSON verdict against the gold keywords, with the
 * deterministic keyword score as fallback (and as the sole grader in --stub).
 * Forgetting precision stays mechanical (old/new keyword check) — objective
 * and free.
 *
 * After the main run: budget sweep for arm C at {300, 600, 1000, 1500, 2500}
 * against the end-state memory store → eval_runs rows tagged
 * metadata {sweep: true, budget} (the dashboard's graceful-degradation chart).
 *
 * Prints a summary table and evaluates the three headline assertions:
 *   1. C accuracy TREND is increasing (late-session avg > early-session avg)
 *   2. C tokens ≪ B tokens (≤ 40 % at the final session)
 *   3. C forgetting-precision > 0.8 while A and B fail (< 0.8)
 * Failures print a diagnosis block for the architect — no silent tuning.
 *
 * Usage:  npm run eval [-- --sessions 8 --seed 0 --stub --no-sweep --db path]
 */

import { parseArgs } from "node:util";
import { randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { z } from "zod";
import { openDb } from "../src/db/client.js";
import { extractMemories, writeMemories } from "../src/memory/write.js";
import { recallDetailed } from "../src/memory/read.js";
import { chat, chatJSON, Models, getUsageStats, resetUsage } from "../src/llm/qwen.js";
import { buildSystemPrompt } from "../src/agent/loop.js";
import {
  generateBenchmark, scoreAccuracy, scoreForgettingPrecision,
} from "./benchmark.js";
import type { Task, BenchmarkSession } from "./benchmark.js";
import type { Db } from "../src/db/client.js";

// ─── CLI args ─────────────────────────────────────────────────────────────────

const { values: args } = parseArgs({
  args:    process.argv.slice(2),
  options: {
    sessions:   { type: "string",  default: "8"   },
    seed:       { type: "string",  default: "0"   },
    db:         { type: "string",  default: ""    },
    stub:       { type: "boolean", default: false },
    "no-sweep": { type: "boolean", default: false },
  },
  strict: false,
});

const STUB_MODE  = Boolean(args.stub);
const RUN_SWEEP  = !args["no-sweep"];
const N_SESSIONS = Math.max(1, Math.min(8, parseInt(args.sessions as string, 10) || 8));
const SEED       = parseInt(args.seed as string, 10) || 0;

/** Arm C context budget for the main run (SPEC §6). */
const ENGRAM_BUDGET = 1500;
/** Arm B transcript cap: beyond this the oldest turns are truncated. */
const FULL_HISTORY_CAP_TOKENS = 16_000;
/** Budget sweep points for arm C. */
const SWEEP_BUDGETS = [300, 600, 1000, 1500, 2500];

// ─── Guard ────────────────────────────────────────────────────────────────────

const apiKey = process.env.DASHSCOPE_API_KEY ?? "";
if (!STUB_MODE && (!apiKey || apiKey.length < 20 || apiKey.includes("..."))) {
  console.error("ERROR: DASHSCOPE_API_KEY must be set to a valid key.");
  console.error("  export DASHSCOPE_API_KEY=sk-...");
  console.error("  Or run with --stub for a deterministic demo without API calls.");
  process.exit(1);
}

// ─── Database setup ───────────────────────────────────────────────────────────

// Fresh in-memory DB for Engram arm memories (isolated per run)
const engDb: Db = openDb(":memory:");

// On-disk DB for writing eval_runs (visible in the web dashboard)
const rawDbPath = (args.db as string).trim();
const dbPath    = rawDbPath
  ? path.resolve(rawDbPath)
  : path.resolve("data/engram.db");

mkdirSync(path.dirname(dbPath), { recursive: true });
const resultsDb: Db = openDb(dbPath);

// Clear previous eval_runs from this seed to avoid double-counting
resultsDb
  .prepare("DELETE FROM eval_runs WHERE metadata LIKE '%\"seed\":' || ? || '%'")
  .run(String(SEED));

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Estimate tokens from character count (4 chars ≈ 1 token). */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/** DashScope pricing estimate (USD per token, qwen-plus rates). */
const PRICE_IN  = 0.0004 / 1000;
const PRICE_OUT = 0.0012 / 1000;

function estimateCost(promptTokens: number, completionTokens: number): number {
  return promptTokens * PRICE_IN + completionTokens * PRICE_OUT;
}

function percentile(arr: number[], p: number): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.floor((p / 100) * (sorted.length - 1));
  return sorted[idx]!;
}

// ─── qwen-flash grader (strict JSON verdict) ──────────────────────────────────

const VerdictSchema = z.object({
  correct: z.boolean(),
  reasoning: z.string(),
});

let _gradeFallbacks = 0;

/**
 * Grade an answer against the task's gold keywords with qwen-flash.
 * Returns 1|0 from the verdict; falls back to the deterministic keyword
 * fraction if the call fails. Stub mode always uses the keyword score.
 */
async function gradeAnswer(task: Task, answer: string): Promise<number> {
  if (STUB_MODE) return scoreAccuracy(task, answer);
  try {
    const verdict = await chatJSON(
      {
        system: `You are a strict exam grader. Given a QUESTION, the REQUIRED FACTS, and an ANSWER, decide whether the answer correctly conveys ALL required facts (synonyms and paraphrases count; missing or wrong facts do not).
Return JSON only: {"correct": true|false, "reasoning": "..."}`,
        messages: [
          {
            role: "user",
            content: `QUESTION: ${task.question}\nREQUIRED FACTS: ${task.goldKeywords.join(", ")}\nANSWER: ${answer}`,
          },
        ],
        model: Models.flash,
        temperature: 0,
      },
      VerdictSchema,
    );
    return verdict.correct ? 1 : 0;
  } catch {
    _gradeFallbacks++;
    return scoreAccuracy(task, answer); // deterministic fallback
  }
}

// ─── Per-arm, per-task result ─────────────────────────────────────────────────

type ArmName = "no_memory" | "full_history" | "engram";

interface TaskResult {
  taskId:              string;
  arm:                 ArmName;
  answer:              string;
  accuracy:            number;
  forgettingPrecision: number | null;
  tokensInContext:     number;
  recallHit:           boolean | null;  // Engram only: gold in top-5 retrieved
  latencyMs:           number;
  promptTokens:        number;
  completionTokens:    number;
}

// ─── System prompts ───────────────────────────────────────────────────────────

const NO_MEMORY_SYSTEM = `\
You are a helpful assistant. Answer ONLY based on information that has been explicitly provided to you in this conversation. Do not draw on outside world knowledge. If you do not have the required information, respond with: "I don't have that information."`;

const HISTORY_SYSTEM_HEADER = `\
You are a helpful assistant with access to a transcript of a user's past statements. Use only the information in [PAST CONTEXT] to answer. Do not add outside knowledge.`;

let _truncationEvents = 0;

/**
 * Build arm B's system prompt from the running transcript.
 * Enforces the 16k-token cap by dropping the OLDEST turns first; every
 * truncated build increments the global counter (reported in metadata).
 */
function buildFullHistorySystem(history: string[]): string {
  if (history.length === 0) return NO_MEMORY_SYSTEM;

  let kept = [...history];
  let text = kept.join("\n");
  if (estimateTokens(text) > FULL_HISTORY_CAP_TOKENS) {
    _truncationEvents++;
    while (kept.length > 1 && estimateTokens(kept.join("\n")) > FULL_HISTORY_CAP_TOKENS) {
      kept.shift(); // drop oldest
    }
    text = kept.join("\n");
  }

  return `${HISTORY_SYSTEM_HEADER}\n\n[PAST CONTEXT]\n${text}\n[END CONTEXT]`;
}

// ─── Stub LLM (deterministic, no API) ────────────────────────────────────────
// Simulates each arm's characteristic behaviour from the task's own keyword
// metadata, so the story holds for ANY benchmark content without hardcoding.

let _stubTokens = { prompt: 0, completion: 0, calls: 0 };

function stubAnswer(
  arm: ArmName,
  task: Task,
  history: string[],
  sessionIndex: number,
): { text: string; latencyMs: number; promptTokens: number; completionTokens: number } {
  const jitter = (base: number) => base + ((task.id.charCodeAt(1) * 7 + sessionIndex * 13) % 60);
  const histText = history.join(" ").toLowerCase();
  const knowsGold = task.goldKeywords.every((k) =>
    histText.replace(/,/g, "").includes(k.toLowerCase().replace(/,/g, "")),
  );

  if (arm === "no_memory") {
    _stubTokens.calls++; _stubTokens.prompt += 200; _stubTokens.completion += 40;
    return {
      text: "I don't have that information.",
      latencyMs: jitter(380), promptTokens: 200, completionTokens: 40,
    };
  }

  if (arm === "full_history") {
    // Has everything ever said — answers gold facts, but on flipped preferences
    // it surfaces the STALE value too (both are in the transcript).
    const parts: string[] = [];
    if (knowsGold) parts.push(`Based on your history: ${task.goldKeywords.join(", ")}.`);
    if (task.testsChangedPreference && task.oldKeywords) {
      const oldInHist = task.oldKeywords.some((k) => histText.includes(k.toLowerCase()));
      if (oldInHist) parts.push(`Previously you used ${task.oldKeywords[0]}, now ${task.newKeywords?.[0]}.`);
    }
    if (parts.length === 0) parts.push("I don't have that information.");
    const promptToks = 300 + estimateTokens(history.join(" "));
    _stubTokens.calls++; _stubTokens.prompt += promptToks; _stubTokens.completion += 80;
    return {
      text: parts.join(" "),
      latencyMs: jitter(480 + history.length * 12),
      promptTokens: promptToks, completionTokens: 80,
    };
  }

  // engram: supersession means only the CURRENT value is in context.
  const known = knowsGold || task.testsChangedPreference;
  _stubTokens.calls++; _stubTokens.prompt += 450; _stubTokens.completion += 60;
  return {
    text: known
      ? `From memory: ${task.goldKeywords.join(", ")}.`
      : "I don't have that information.",
    latencyMs: jitter(520), promptTokens: 450, completionTokens: 60,
  };
}

// ─── Arm runners ─────────────────────────────────────────────────────────────

async function runNoMemory(task: Task, history: string[], sessionIndex: number): Promise<TaskResult> {
  if (STUB_MODE) {
    const s = stubAnswer("no_memory", task, [], sessionIndex);
    return {
      taskId: task.id, arm: "no_memory", answer: s.text,
      accuracy: scoreAccuracy(task, s.text),
      forgettingPrecision: scoreForgettingPrecision(task, s.text),
      tokensInContext: 0, recallHit: null, latencyMs: s.latencyMs,
      promptTokens: s.promptTokens, completionTokens: s.completionTokens,
    };
  }

  const t0 = Date.now();
  const before = getUsageStats();
  const result = await chat({
    system: NO_MEMORY_SYSTEM,
    messages: [{ role: "user", content: task.question }],
    model: Models.plus,
    temperature: 0.1,
  });
  const after = getUsageStats();

  return {
    taskId: task.id, arm: "no_memory", answer: result.text,
    accuracy: await gradeAnswer(task, result.text),
    forgettingPrecision: scoreForgettingPrecision(task, result.text),
    tokensInContext: 0, recallHit: null,
    latencyMs: Date.now() - t0,
    promptTokens: after.promptTokens - before.promptTokens,
    completionTokens: after.completionTokens - before.completionTokens,
  };
}

async function runFullHistory(task: Task, history: string[], sessionIndex: number): Promise<TaskResult> {
  const system  = buildFullHistorySystem(history);
  const ctxToks = estimateTokens(system);

  if (STUB_MODE) {
    const s = stubAnswer("full_history", task, history, sessionIndex);
    return {
      taskId: task.id, arm: "full_history", answer: s.text,
      accuracy: scoreAccuracy(task, s.text),
      forgettingPrecision: scoreForgettingPrecision(task, s.text),
      tokensInContext: ctxToks, recallHit: null, latencyMs: s.latencyMs,
      promptTokens: s.promptTokens, completionTokens: s.completionTokens,
    };
  }

  const t0 = Date.now();
  const before = getUsageStats();
  const result = await chat({
    system,
    messages: [{ role: "user", content: task.question }],
    model: Models.plus,
    temperature: 0.1,
  });
  const after = getUsageStats();

  return {
    taskId: task.id, arm: "full_history", answer: result.text,
    accuracy: await gradeAnswer(task, result.text),
    forgettingPrecision: scoreForgettingPrecision(task, result.text),
    tokensInContext: ctxToks, recallHit: null,
    latencyMs: Date.now() - t0,
    promptTokens: after.promptTokens - before.promptTokens,
    completionTokens: after.completionTokens - before.completionTokens,
  };
}

async function runEngram(
  task: Task,
  userId: string,
  db: Db,
  history: string[],
  sessionIndex: number,
  budget: number = ENGRAM_BUDGET,
): Promise<TaskResult> {
  if (STUB_MODE) {
    // Deterministic per-task gate (0–5) drives a learning curve: stable facts
    // are missed early and reliably recalled by later sessions; flipped
    // preferences always resolve to the NEW value (supersession).
    const gate = [...task.id].reduce((s, c) => s + c.charCodeAt(0), 0) % 6;
    const learned = task.testsChangedPreference ? true : sessionIndex + 1 > gate;
    // Budget starvation: tight budgets drop some stable answers.
    const starved = !task.testsChangedPreference &&
      (budget <= 300 ? gate % 2 === 0 : budget <= 600 ? gate === 0 : false);

    const know = learned && !starved;
    const text = know
      ? `From memory: ${task.goldKeywords.join(", ")}.`
      : "I don't have that information.";
    const ctxToks = Math.min(380 + sessionIndex * 15, budget);
    _stubTokens.calls++; _stubTokens.prompt += 450; _stubTokens.completion += 60;
    return {
      taskId: task.id, arm: "engram", answer: text,
      accuracy: scoreAccuracy(task, text),
      forgettingPrecision: scoreForgettingPrecision(task, text),
      tokensInContext: ctxToks,
      recallHit: know,
      latencyMs: 520 + ((gate * 17 + sessionIndex * 13) % 60),
      promptTokens: 450, completionTokens: 60,
    };
  }

  const t0 = Date.now();
  const before = getUsageStats();

  // Hybrid recall → pack; `retrieved` (pre-packing) drives recall@5.
  const { packed, retrieved } = await recallDetailed(userId, task.question, db, budget);
  const top5 = retrieved.slice(0, 5).map((r) => r.content.toLowerCase().replace(/,/g, "")).join(" ");
  const recallHit = task.goldKeywords.some((kw) =>
    top5.includes(kw.toLowerCase().replace(/,/g, "")),
  );

  const result = await chat({
    system: buildSystemPrompt(packed.text),
    messages: [{ role: "user", content: task.question }],
    model: Models.plus,
    temperature: 0.1,
  });
  const after = getUsageStats();

  return {
    taskId: task.id, arm: "engram", answer: result.text,
    accuracy: await gradeAnswer(task, result.text),
    forgettingPrecision: scoreForgettingPrecision(task, result.text),
    tokensInContext: packed.totalTokens, recallHit,
    latencyMs: Date.now() - t0,
    promptTokens: after.promptTokens - before.promptTokens,
    completionTokens: after.completionTokens - before.completionTokens,
  };
}

// ─── Engram prologue writer ───────────────────────────────────────────────────

async function ingestPrologue(
  text: string,
  userId: string,
  sessionId: string,
  db: Db,
): Promise<void> {
  if (STUB_MODE) return; // stub mode: memories are simulated in stubAnswer
  try {
    const candidates = await extractMemories(text);
    if (candidates.length > 0) {
      await writeMemories(userId, sessionId, candidates, db);
    }
  } catch (err) {
    console.warn(`  [engram-prologue] extraction failed: ${(err as Error).message}`);
  }
}

// ─── Per-session aggregation ──────────────────────────────────────────────────

interface SessionMetrics {
  sessionIndex: number;
  arm:          ArmName;
  taskAccuracy: number;
  tokensInContext: number;
  recallAtK:    number | null;
  forgettingPrecision: number | null;
  latencyP50Ms: number;
  latencyP95Ms: number;
  costUsd:      number;
}

function aggregateArm(results: TaskResult[], arm: ArmName, sessionIndex: number): SessionMetrics {
  const armResults = results.filter((r) => r.arm === arm);
  if (armResults.length === 0) {
    return {
      sessionIndex, arm,
      taskAccuracy: 0, tokensInContext: 0,
      recallAtK: null, forgettingPrecision: null,
      latencyP50Ms: 0, latencyP95Ms: 0, costUsd: 0,
    };
  }

  const avgAccuracy = armResults.reduce((s, r) => s + r.accuracy, 0) / armResults.length;
  const avgTokens   = armResults.reduce((s, r) => s + r.tokensInContext, 0) / armResults.length;

  const engRecalls = armResults.filter((r) => r.recallHit !== null);
  const recallAtK = engRecalls.length > 0
    ? engRecalls.filter((r) => r.recallHit).length / engRecalls.length
    : null;

  const fpResults = armResults.filter((r) => r.forgettingPrecision !== null);
  const forgettingPrecision = fpResults.length > 0
    ? fpResults.reduce((s, r) => s + r.forgettingPrecision!, 0) / fpResults.length
    : null;

  const latencies = armResults.map((r) => r.latencyMs);
  const totalCost = armResults.reduce(
    (s, r) => s + estimateCost(r.promptTokens, r.completionTokens), 0,
  );

  return {
    sessionIndex, arm,
    taskAccuracy: avgAccuracy,
    tokensInContext: Math.round(avgTokens),
    recallAtK, forgettingPrecision,
    latencyP50Ms: percentile(latencies, 50),
    latencyP95Ms: percentile(latencies, 95),
    costUsd: totalCost,
  };
}

// ─── eval_runs writer ─────────────────────────────────────────────────────────

const insertRun = resultsDb.prepare(`
  INSERT OR REPLACE INTO eval_runs
    (id, run_at, arm, session_index, task_accuracy, tokens_in_context,
     recall_at_k, forgetting_precision, latency_p50_ms, latency_p95_ms,
     cost_usd, metadata)
  VALUES
    (@id, @run_at, @arm, @session_index, @task_accuracy, @tokens_in_context,
     @recall_at_k, @forgetting_precision, @latency_p50_ms, @latency_p95_ms,
     @cost_usd, @metadata)
`);

function persistMetrics(m: SessionMetrics, extraMeta: Record<string, unknown> = {}): void {
  insertRun.run({
    id: randomUUID(),
    run_at: Date.now(),
    arm: m.arm,
    session_index: m.sessionIndex,
    task_accuracy: m.taskAccuracy,
    tokens_in_context: m.tokensInContext,
    recall_at_k: m.recallAtK,
    forgetting_precision: m.forgettingPrecision,
    latency_p50_ms: m.latencyP50Ms,
    latency_p95_ms: m.latencyP95Ms,
    cost_usd: m.costUsd,
    metadata: JSON.stringify({ seed: SEED, stub: STUB_MODE, ...extraMeta }),
  });
}

// ─── Table printer ────────────────────────────────────────────────────────────

function fmt(v: number | null, decimals = 2): string {
  if (v == null) return "  —   ";
  return v.toFixed(decimals).padStart(6);
}

function printTable(allMetrics: SessionMetrics[]): void {
  const ARMS: ArmName[] = ["no_memory", "full_history", "engram"];
  const ARM_LABEL: Record<string, string> = {
    no_memory: "No-Mem  ", full_history: "FullHist", engram: "Engram  ",
  };

  console.log("\n" + "═".repeat(108));
  console.log("  Engram 3-arm benchmark — summary");
  console.log("═".repeat(108));
  console.log("  " + ["Sess", "Arm     ", "  Acc ", "  Tok ", " R@5 ", "  FP  ", "P50ms", "P95ms", "Cost($)"].join("  "));
  console.log("  " + "─".repeat(104));

  const sessions = [...new Set(allMetrics.map((m) => m.sessionIndex))].sort((a, b) => a - b);
  for (const s of sessions) {
    for (const arm of ARMS) {
      const m = allMetrics.find((x) => x.sessionIndex === s && x.arm === arm);
      if (!m) continue;
      console.log("  " + [
        String(s).padStart(4),
        ARM_LABEL[arm]!,
        fmt(m.taskAccuracy),
        String(m.tokensInContext).padStart(6),
        fmt(m.recallAtK),
        fmt(m.forgettingPrecision),
        String(Math.round(m.latencyP50Ms)).padStart(5),
        String(Math.round(m.latencyP95Ms)).padStart(5),
        m.costUsd.toFixed(5).padStart(7),
      ].join("  "));
    }
    console.log("  " + "·".repeat(104));
  }
  console.log("═".repeat(108));
}

// ─── Headline assertions (SPEC §6) ────────────────────────────────────────────

interface AssertionOutcome { name: string; pass: boolean; diagnosis: string }

function assertHeadlines(allMetrics: SessionMetrics[]): AssertionOutcome[] {
  const outcomes: AssertionOutcome[] = [];
  const eng = (s: number) => allMetrics.find((m) => m.sessionIndex === s && m.arm === "engram");
  const lastSession = Math.max(...allMetrics.map((m) => m.sessionIndex));
  const sessions = [...new Set(allMetrics.map((m) => m.sessionIndex))].sort((a, b) => a - b);

  // 1. Engram accuracy TREND is increasing: late-third avg > early-third avg
  const third = Math.max(1, Math.floor(sessions.length / 3));
  const early = sessions.slice(0, third).map((s) => eng(s)?.taskAccuracy ?? 0);
  const late  = sessions.slice(-third).map((s) => eng(s)?.taskAccuracy ?? 0);
  const earlyAvg = early.reduce((a, b) => a + b, 0) / early.length;
  const lateAvg  = late.reduce((a, b) => a + b, 0) / late.length;
  outcomes.push({
    name: "C accuracy trend increasing",
    pass: lateAvg > earlyAvg,
    diagnosis: `early-third avg ${fmt(earlyAvg)} vs late-third avg ${fmt(lateAvg)}. ` +
      `If flat/decreasing: check extraction quality (memories written per session), ` +
      `recall@5 (retrieval misses), and whether supersession is over-firing on stable facts.`,
  });

  // 2. C tokens ≪ B tokens at the final session (≤ 40 %)
  const engLast  = allMetrics.find((m) => m.sessionIndex === lastSession && m.arm === "engram")!;
  const fullLast = allMetrics.find((m) => m.sessionIndex === lastSession && m.arm === "full_history")!;
  const ratio = fullLast.tokensInContext > 0 ? engLast.tokensInContext / fullLast.tokensInContext : 0;
  outcomes.push({
    name: "C tokens ≪ B tokens",
    pass: ratio <= 0.4,
    diagnosis: `Engram ${engLast.tokensInContext} tok vs full-history ${fullLast.tokensInContext} tok ` +
      `(ratio ${(ratio * 100).toFixed(1)}%). If failing: packer budget leak or full-history cap set too low.`,
  });

  // 3. C forgetting-precision > 0.8 while A and B fail (< 0.8)
  const avgFP = (arm: ArmName) => {
    const rows = allMetrics.filter((m) => m.arm === arm && m.forgettingPrecision !== null);
    return rows.length ? rows.reduce((s, m) => s + m.forgettingPrecision!, 0) / rows.length : null;
  };
  const engFP = avgFP("engram"), fullFP = avgFP("full_history"), noneFP = avgFP("no_memory");
  outcomes.push({
    name: "C forgetting-precision > 0.8, A/B fail",
    pass: engFP !== null && engFP > 0.8 && (fullFP === null || fullFP < 0.8) && (noneFP === null || noneFP < 0.8),
    diagnosis: `FP — Engram ${fmt(engFP)} · full-history ${fmt(fullFP)} · no-memory ${fmt(noneFP)}. ` +
      `If Engram < 0.8: supersession verdicts missing the flip (inspect memory_edges + belief_events for the flipped prefs). ` +
      `If B ≥ 0.8: the model is resolving staleness from transcript order — tighten FP tasks.`,
  });

  console.log("\n── Headline assertions ─────────────────────────────────────");
  for (const o of outcomes) {
    console.log(`  [${o.pass ? "PASS" : "FAIL"}] ${o.name}`);
    if (!o.pass) console.log(`         diagnosis: ${o.diagnosis}`);
  }
  const allPass = outcomes.every((o) => o.pass);
  console.log(`\n  Overall: ${allPass ? "✓ ALL ASSERTIONS PASS" : "✗ ASSERTION FAILURES — file an issue per failure with the diagnosis above; the architect decides the fix."}`);
  console.log("─".repeat(60));
  if (!allPass) process.exitCode = 1;
  return outcomes;
}

// ─── Budget sweep (arm C, end-state store) ────────────────────────────────────

async function runSweep(benchmark: BenchmarkSession[], userId: string): Promise<void> {
  console.log(`\nBudget sweep (arm C, end-state store): ${SWEEP_BUDGETS.join(" / ")} tokens`);
  const allTasks = benchmark.flatMap((s) => s.tasks);
  // The store is END-STATE — every task is evaluated as of the final session.
  const finalSession = Math.max(...benchmark.map((s) => s.index));

  for (const budget of SWEEP_BUDGETS) {
    const results: TaskResult[] = [];
    process.stdout.write(`  budget ${String(budget).padStart(4)}: `);
    for (const task of allTasks) {
      results.push(await runEngram(task, userId, engDb, [], finalSession, budget));
    }
    const m = aggregateArm(results, "engram", finalSession);
    persistMetrics(m, { sweep: true, budget });
    console.log(
      `acc ${fmt(m.taskAccuracy)}  tok ${String(m.tokensInContext).padStart(5)}  ` +
      `R@5 ${fmt(m.recallAtK)}  FP ${fmt(m.forgettingPrecision)}`,
    );
  }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const benchmark = generateBenchmark(SEED, N_SESSIONS);
  const userId = `eval-jordan-seed${SEED}`;
  const totalTasks = benchmark.reduce((s, b) => s + b.tasks.length, 0);

  console.log(`\nEngram benchmark  seed=${SEED}  sessions=${benchmark.length}  tasks=${totalTasks}${STUB_MODE ? "  [STUB MODE]" : ""}`);
  console.log(`Engram DB: :memory:   Results DB: ${dbPath}   arm-C budget: ${ENGRAM_BUDGET}   arm-B cap: ${FULL_HISTORY_CAP_TOKENS}`);
  console.log();

  resetUsage();

  const allMetrics: SessionMetrics[] = [];
  const fullHistory: string[] = [];

  for (const session of benchmark) {
    const start = Date.now();
    process.stdout.write(`Session ${session.index}/${benchmark.length}  `);

    if (session.prologues.length > 0) {
      process.stdout.write(`ingest(${session.prologues.length}) `);
      for (const text of session.prologues) {
        await ingestPrologue(text, userId, `session-${session.index}`, engDb);
        fullHistory.push(text);
      }
    }

    const taskResults: TaskResult[] = [];
    process.stdout.write(`tasks(${session.tasks.length}) `);

    for (const task of session.tasks) {
      const [noMem, fullHist, engram] = await Promise.all([
        runNoMemory(task, fullHistory, session.index),
        runFullHistory(task, fullHistory, session.index),
        runEngram(task, userId, engDb, fullHistory, session.index),
      ]);
      taskResults.push(noMem, fullHist, engram);

      // Q&A joins the transcript so later sessions see it (arm B realism).
      fullHistory.push(`User: ${task.question}\nAssistant: ${fullHist.answer.slice(0, 300)}`);
    }

    for (const arm of ["no_memory", "full_history", "engram"] as const) {
      const m = aggregateArm(taskResults, arm, session.index);
      allMetrics.push(m);
      persistMetrics(m, arm === "full_history" ? { truncations: _truncationEvents } : {});
    }

    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    process.stdout.write(`done (${elapsed}s)\n`);
  }

  printTable(allMetrics);

  if (_truncationEvents > 0) {
    console.log(`\n  full-history arm truncated its transcript ${_truncationEvents}× (16k cap) — the overflow story.`);
  }
  if (_gradeFallbacks > 0) {
    console.log(`  flash grader fell back to keyword scoring ${_gradeFallbacks}× (API errors).`);
  }

  if (RUN_SWEEP) await runSweep(benchmark, userId);

  assertHeadlines(allMetrics);

  if (STUB_MODE) {
    console.log(
      `\nStub usage: ${_stubTokens.calls} simulated calls · ` +
      `≈$${estimateCost(_stubTokens.prompt, _stubTokens.completion).toFixed(4)} (if real)`,
    );
  } else {
    const u = getUsageStats();
    console.log(
      `\nTotal LLM spend: ${u.calls} calls · ${u.promptTokens} prompt · ${u.completionTokens} completion ` +
      `· ≈$${estimateCost(u.promptTokens, u.completionTokens).toFixed(4)}`,
    );
    for (const [model, mu] of Object.entries(u.byModel)) {
      console.log(`  ${model}: ${mu.calls} calls · ${mu.totalTokens} tok`);
    }
  }

  engDb.close();
  resultsDb.close();
}

main().catch((err: unknown) => {
  console.error("Benchmark failed:", err);
  process.exit(1);
});
