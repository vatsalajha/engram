/**
 * src/agent/loop.ts
 *
 * The Engram agent loop (SPEC §4.6) — stateless per request, server-authoritative.
 *
 * act(userId, sessionId, input, db, opts):
 *   1. perceive  — log the user turn to episodic_log
 *   2. recall()  — hybrid retrieve → RRF → score → pack(budget)
 *   3. decide()  — qwen-plus, project-copilot persona, packed context injected
 *                  as "What you know about this user"; streams via opts.onToken
 *   4. post-response queue (never blocks the stream, guaranteed to complete):
 *        a. log the assistant turn to episodic_log
 *        b. extractMemories + writeMemories (dedupe → supersede → store+edges)
 *        c. implicitJudge — belief updates on every memory in the manifest,
 *           anchored to the first memory extracted from this turn (if any)
 *
 * The caller MUST call trackWrite(result.writePromise) or await it to ensure
 * post-response work completes before the process exits. See drainWrites.
 */

import { ulid } from "ulid";
import { chatStream, Models, type ChatResult } from "../llm/qwen.js";
import { recall } from "../memory/read.js";
import { extractMemories, writeMemories } from "../memory/write.js";
import { implicitJudge } from "../memory/belief.js";
import { insertEpisodic } from "../db/queries.js";
import type { Db } from "../db/client.js";
import type { PackedContext } from "../memory/packer.js";

// ─── Post-response task registry ──────────────────────────────────────────────
// Tracks background work so the server can drain it on shutdown and
// the /admin/stats route can surface it.

const _pendingWrites = new Set<Promise<void>>();

export function pendingWriteCount(): number {
  return _pendingWrites.size;
}

export function trackWrite(p: Promise<void>): void {
  _pendingWrites.add(p);
  // .catch() prevents unhandledRejection — writes log errors internally;
  // drainWrites uses allSettled so errors don't propagate to callers.
  p.catch(() => { /* swallowed */ }).finally(() => _pendingWrites.delete(p));
}

/** Wait for all in-flight background work to settle (best-effort). */
export async function drainWrites(): Promise<void> {
  await Promise.allSettled(Array.from(_pendingWrites));
}

// ─── Decision prompt builder ──────────────────────────────────────────────────

const SYSTEM_BASE = `\
You are Engram, the user's project copilot — an assistant that remembers them
across sessions and gets more useful every time you talk.

Rely on what you know about this user: their preferences, decisions, stack,
constraints, and habits. Apply that knowledge without being asked to.
When a remembered fact meaningfully shaped your answer, say so briefly and
naturally (e.g. "since you switched to SQLite…") — one short clause, not a
recital of your memory.

Be concise and direct. If nothing you know is relevant to this turn, just
answer well and mention nothing.`;

export function buildSystemPrompt(packedContext: string): string {
  if (!packedContext.trim()) return SYSTEM_BASE;
  return `${SYSTEM_BASE}

[What you know about this user]
${packedContext}
[end]`;
}

// ─── ActResult ────────────────────────────────────────────────────────────────

export interface ActResult {
  turnId: string;
  answer: string;
  contextManifest: PackedContext["manifest"];
  contextTokens: number;
  usage: ChatResult["usage"];
  model: string;
  writePromise: Promise<void>;
}

export interface ActOptions {
  /** Called with each streamed token; enables SSE forwarding. */
  onToken?: (token: string) => Promise<void> | void;
  /** Token budget for context packing. Default: config.ENGRAM_CONTEXT_BUDGET. */
  budget?: number;
}

// ─── act() ───────────────────────────────────────────────────────────────────

/**
 * Run one full agent-loop turn:
 *   perceive → recall → decide (streaming) → queue post-response work
 *
 * Returns once the LLM answer is complete. Post-response work (episodic log,
 * extraction, writes, belief updates) runs on the tracked background queue.
 */
export async function act(
  userId: string,
  sessionId: string,
  input: string,
  db: Db,
  opts: ActOptions = {},
): Promise<ActResult> {
  const turnId = ulid();

  // ── 1. Perceive: the user turn enters the episodic log immediately ─────────
  insertEpisodic(db, { user_id: userId, session_id: sessionId, role: "user", content: input });

  // ── 2. Recall ───────────────────────────────────────────────────────────────
  const packed: PackedContext = await recall(userId, input, db, opts.budget);

  // ── 3. Decide (stream tokens) — qwen-plus, copilot persona ─────────────────
  const systemPrompt = buildSystemPrompt(packed.text);

  const chatResult = await chatStream(
    {
      system: systemPrompt,
      messages: [{ role: "user", content: input }],
      model: Models.plus,
      temperature: 0.7,
    },
    async (token) => {
      if (opts.onToken) await opts.onToken(token);
    },
  );

  // ── 4. Post-response queue ──────────────────────────────────────────────────
  // Runs after the answer is complete; never blocks the stream. Steps are
  // sequential so belief updates can anchor to this turn's extracted memory.
  // Failures are logged but never propagate to the caller.
  const writePromise: Promise<void> = (async () => {
    try {
      // a. Assistant turn → episodic log (sleep-cycle input + provenance)
      insertEpisodic(db, {
        user_id: userId, session_id: sessionId, role: "assistant", content: chatResult.text,
      });

      // b. Reflect: extract + write typed memories
      let anchorId: string | null = null;
      const candidates = await extractMemories(input, chatResult.text);
      if (candidates.length > 0) {
        const written = await writeMemories(userId, sessionId, candidates, db);
        anchorId = written.inserted[0] ?? null;
      }

      // c. Belief updates: judge the packed manifest against this turn
      if (packed.manifest.length > 0) {
        await implicitJudge(
          db, userId, turnId,
          packed.manifest,
          { user: input, assistant: chatResult.text },
          { anchorMemoryId: anchorId },
        );
      }
    } catch (err) {
      console.error(`[loop] post-response work failed for user=${userId} session=${sessionId}:`, err);
    }
  })();

  return {
    turnId,
    answer:          chatResult.text,
    contextManifest: packed.manifest,
    contextTokens:   packed.totalTokens,
    usage:           chatResult.usage,
    model:           Models.plus,
    writePromise,
  };
}
