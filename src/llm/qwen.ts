/**
 * src/llm/qwen.ts
 *
 * Single point of contact with Alibaba Cloud DashScope (Model Studio).
 * Uses the OpenAI-compatible endpoint — HACKATHON PROOF OF ALIBABA CLOUD API USE.
 *
 * Endpoint: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
 * Auth:     DASHSCOPE_API_KEY  (DashScope API key from Model Studio console)
 *
 * Models used:
 *   qwen-plus   — memory extraction, conflict detection, final decisions
 *   qwen-max    — defined for completeness (escalation cut in v2 scope)
 *   qwen-flash  — high-volume classify / compress / rerank (cheap)
 *   text-embedding-v4 (dim 1024) — semantic embeddings for hybrid retrieval
 */

import OpenAI from "openai";
import { z } from "zod";
import { config } from "../config.js";

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/**
 * Late-bound wrapper around globalThis.fetch so tests can stub the HTTP layer
 * after the client is constructed (see test/qwen.test.ts). The cast bridges
 * the SDK's @types/node-fetch signatures to the native undici fetch — they
 * are structurally compatible at runtime.
 */
const testableFetch = ((url: string | URL | Request, init?: RequestInit) =>
  globalThis.fetch(url, init)) as unknown as NonNullable<
  ConstructorParameters<typeof OpenAI>[0]
>["fetch"];

/**
 * OpenAI SDK pointed at DashScope's OpenAI-compatible endpoint.
 *
 * - timeout: 30s per request.
 * - maxRetries: 1 — the SDK retries once with jittered exponential backoff
 *   on 429, 408, connection errors and 5xx responses.
 */
const client = new OpenAI({
  apiKey: config.DASHSCOPE_API_KEY,
  baseURL: config.ENGRAM_QWEN_BASE_URL,
  timeout: 30_000,
  maxRetries: 1,
  fetch: testableFetch,
});

// ---------------------------------------------------------------------------
// Model names
// ---------------------------------------------------------------------------

export const Models = {
  plus: "qwen-plus",
  max: "qwen-max",
  flash: "qwen-flash",
  embedding: "text-embedding-v4",
} as const;

export type QwenModel = (typeof Models)[keyof typeof Models];

// ---------------------------------------------------------------------------
// Token usage tracking
// ---------------------------------------------------------------------------

interface ModelUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  calls: number;
}

export interface UsageSnapshot extends ModelUsage {
  /** Per-model breakdown, keyed by model name (e.g. "qwen-plus"). */
  byModel: Record<string, ModelUsage>;
}

const _emptyUsage = (): ModelUsage => ({
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
  calls: 0,
});

let _byModel: Record<string, ModelUsage> = {};

function _accrue(
  model: string,
  u: { prompt_tokens: number; completion_tokens: number; total_tokens: number },
) {
  const bucket = (_byModel[model] ??= _emptyUsage());
  bucket.promptTokens += u.prompt_tokens;
  bucket.completionTokens += u.completion_tokens;
  bucket.totalTokens += u.total_tokens;
  bucket.calls += 1;
}

/**
 * Cumulative token usage since last reset: aggregate totals at the top level
 * (backward-compatible) plus a per-model breakdown under `byModel`.
 */
export function getUsageStats(): Readonly<UsageSnapshot> {
  const total = _emptyUsage();
  const byModel: Record<string, ModelUsage> = {};
  for (const [model, u] of Object.entries(_byModel)) {
    byModel[model] = { ...u };
    total.promptTokens += u.promptTokens;
    total.completionTokens += u.completionTokens;
    total.totalTokens += u.totalTokens;
    total.calls += u.calls;
  }
  return { ...total, byModel };
}

export function resetUsage(): void {
  _byModel = {};
}

// ─── Cost estimation ─────────────────────────────────────────────────────────
// Approximate USD per 1M tokens (DashScope international list prices, mid-2026;
// used for the demo's cumulative cost counter, not for billing).

const PRICE_PER_MTOK: Record<string, { input: number; output: number }> = {
  [Models.plus]:      { input: 0.40, output: 1.20 },
  [Models.max]:       { input: 1.60, output: 6.40 },
  [Models.flash]:     { input: 0.05, output: 0.40 },
  [Models.embedding]: { input: 0.07, output: 0 },
};

/** Estimated cumulative USD cost of all calls since the last resetUsage(). */
export function estimateCostUSD(): number {
  let usd = 0;
  for (const [model, u] of Object.entries(_byModel)) {
    const price = PRICE_PER_MTOK[model];
    if (!price) continue;
    usd += (u.promptTokens / 1_000_000) * price.input;
    usd += (u.completionTokens / 1_000_000) * price.output;
  }
  return usd;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatOptions {
  system: string;
  messages: ChatMessage[];
  model?: string;
  temperature?: number;
  /** Appends a JSON-only instruction to the system prompt. */
  jsonMode?: boolean;
}

export interface ChatResult {
  text: string;
  usage: { promptTokens: number; completionTokens: number; totalTokens: number };
}

// ---------------------------------------------------------------------------
// chat()
// ---------------------------------------------------------------------------

/**
 * Single-turn or multi-turn chat completion via DashScope.
 * Returns the assistant text and token usage for this call.
 */
export async function chat(opts: ChatOptions): Promise<ChatResult> {
  const { messages, model = Models.plus, temperature = 0.7, jsonMode = false } = opts;

  let system = opts.system;
  if (jsonMode) {
    system +=
      "\n\nIMPORTANT: Respond with valid JSON only. Do not include markdown code fences, commentary, or any text outside the JSON object.";
  }

  const response = await client.chat.completions.create({
    model,
    temperature,
    messages: [
      { role: "system", content: system },
      ...messages.map((m) => ({ role: m.role, content: m.content })),
    ],
  });

  const choice = response.choices[0];
  if (!choice?.message?.content) {
    throw new Error(`Qwen returned no content (model=${model}, finish_reason=${choice?.finish_reason})`);
  }

  const text = choice.message.content;
  const u = response.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
  _accrue(model, u);

  return {
    text,
    usage: {
      promptTokens: u.prompt_tokens,
      completionTokens: u.completion_tokens,
      totalTokens: u.total_tokens,
    },
  };
}

// ---------------------------------------------------------------------------
// chatJSON<T>()
// ---------------------------------------------------------------------------

/** Strip markdown code fences that some models emit even when asked not to. */
export function stripFences(raw: string): string {
  return raw
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```\s*$/, "")
    .trim();
}

/**
 * Like chat(), but parses the response as JSON and validates it with a Zod schema.
 * Retries once on parse/validation failure before throwing.
 */
export async function chatJSON<T>(
  opts: ChatOptions,
  schema: z.ZodType<T>
): Promise<T> {
  async function attempt(extraHint?: string): Promise<T> {
    const augmentedOpts: ChatOptions = {
      ...opts,
      jsonMode: true,
      system: extraHint ? `${opts.system}\n\n${extraHint}` : opts.system,
    };

    const { text } = await chat(augmentedOpts);
    const stripped = stripFences(text);

    let parsed: unknown;
    try {
      parsed = JSON.parse(stripped);
    } catch (err) {
      throw new Error(`JSON parse failed: ${err instanceof Error ? err.message : err}\nRaw: ${stripped.slice(0, 300)}`);
    }

    const result = schema.safeParse(parsed);
    if (!result.success) {
      throw new Error(`Zod validation failed: ${result.error.message}`);
    }
    return result.data;
  }

  try {
    return await attempt();
  } catch (firstErr) {
    // Retry once with the error fed back so the model can self-correct.
    const hint = `Your previous response failed validation: ${firstErr instanceof Error ? firstErr.message : firstErr}. Fix the JSON and try again.`;
    try {
      return await attempt(hint);
    } catch (secondErr) {
      throw new Error(
        `chatJSON failed after 2 attempts.\nFirst: ${firstErr instanceof Error ? firstErr.message : firstErr}\nSecond: ${secondErr instanceof Error ? secondErr.message : secondErr}`
      );
    }
  }
}

// ---------------------------------------------------------------------------
// chatStream()
// ---------------------------------------------------------------------------

/**
 * Streaming chat completion via DashScope.
 * Calls `onChunk` for each text delta so the caller can forward tokens to SSE.
 * Returns the full text + usage once the stream ends.
 */
export async function chatStream(
  opts: ChatOptions,
  onChunk: (text: string) => Promise<void> | void,
): Promise<ChatResult> {
  const { messages, model = Models.plus, temperature = 0.7 } = opts;

  // stream: true as const lets TypeScript pick the correct overload
  // (Stream<ChatCompletionChunk> rather than ChatCompletion | Stream<…>).
  const stream = await (client.chat.completions.create as (
    params: { model: string; temperature: number; messages: { role: string; content: string }[]; stream: true }
  ) => Promise<AsyncIterable<{ choices: { delta: { content?: string | null } }[]; usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null }>>)({
    model,
    temperature,
    messages: [
      { role: "system", content: opts.system },
      ...messages.map((m) => ({ role: m.role, content: m.content })),
    ],
    stream: true,
  });

  let fullText = "";
  let usage = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };

  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content ?? "";
    if (delta) {
      fullText += delta;
      await onChunk(delta);
    }
    // Some endpoints send usage in the final (empty-choice) chunk
    if (chunk.usage) {
      usage = {
        prompt_tokens:     chunk.usage.prompt_tokens,
        completion_tokens: chunk.usage.completion_tokens,
        total_tokens:      chunk.usage.total_tokens,
      };
    }
  }

  _accrue(model, usage);

  return {
    text: fullText,
    usage: {
      promptTokens:     usage.prompt_tokens,
      completionTokens: usage.completion_tokens,
      totalTokens:      usage.total_tokens,
    },
  };
}

// ---------------------------------------------------------------------------
// embed()
// ---------------------------------------------------------------------------

const EMBED_BATCH_SIZE = 10; // DashScope batch limit

/**
 * Embed an array of texts using text-embedding-v4 (Alibaba Cloud DashScope).
 * Batches automatically in groups of 10.
 * Returns number[][] where each inner array has length `dim` (default 1024).
 */
export async function embed(texts: string[], dim = 1024): Promise<number[][]> {
  if (texts.length === 0) return [];

  const results: number[][] = [];

  for (let i = 0; i < texts.length; i += EMBED_BATCH_SIZE) {
    const batch = texts.slice(i, i + EMBED_BATCH_SIZE);

    // encoding_format is explicit: without it the SDK requests base64 and
    // blindly decodes the response, which corrupts embeddings if the server
    // returns plain float arrays (DashScope compatible-mode behavior).
    const response = await client.embeddings.create({
      model: Models.embedding,
      input: batch,
      dimensions: dim,
      encoding_format: "float",
    } as Parameters<typeof client.embeddings.create>[0]);

    // Sort by index to guarantee order matches input
    const sorted = [...response.data].sort((a, b) => a.index - b.index);
    results.push(...sorted.map((d) => d.embedding));

    if (response.usage) {
      // Embed calls only have prompt_tokens; treat total = prompt
      _accrue(Models.embedding, {
        prompt_tokens: response.usage.prompt_tokens,
        completion_tokens: 0,
        total_tokens: response.usage.total_tokens ?? response.usage.prompt_tokens,
      });
    }
  }

  return results;
}
