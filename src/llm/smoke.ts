/**
 * src/llm/smoke.ts — live sanity check for DashScope (Alibaba Cloud) connectivity
 * Run: npm run smoke   (requires a real DASHSCOPE_API_KEY)
 *
 * Exercises all three client surfaces:
 *   1. chat()      — must reply with exactly "ENGRAM_OK"
 *   2. chatJSON()  — round-trip against a tiny Zod schema
 *   3. embed()     — 3 texts, expect 3 × 1024-dim vectors
 * Prints outputs, embedding dims, and cumulative per-model usage stats.
 */

import { z } from "zod";
import { chat, chatJSON, embed, getUsageStats, Models } from "./qwen.js";

async function main() {
  console.log("=== Engram / Qwen smoke test ===\n");

  // 1. chat — exact-reply contract
  console.log("[ 1 ] chat — qwen-plus, expecting exactly ENGRAM_OK");
  const { text } = await chat({
    system: 'Reply with exactly: ENGRAM_OK — no punctuation, no other words.',
    messages: [{ role: "user", content: "ping" }],
    model: Models.plus,
    temperature: 0,
  });
  const chatOk = text.trim() === "ENGRAM_OK";
  console.log(`  Reply: "${text.trim()}"  ${chatOk ? "✓" : "✗ (expected ENGRAM_OK)"}\n`);

  // 2. chatJSON — tiny schema round-trip
  console.log("[ 2 ] chatJSON — tiny schema round-trip");
  const TinySchema = z.object({
    project: z.string(),
    memory_types: z.array(z.string()).min(3),
    confident: z.boolean(),
  });
  const parsed = await chatJSON(
    {
      system: "You are Engram's extraction module.",
      messages: [
        {
          role: "user",
          content:
            'Return JSON: {"project": "engram", "memory_types": ["preference","fact","hypothesis"], "confident": true}',
        },
      ],
      model: Models.plus,
      temperature: 0,
    },
    TinySchema,
  );
  const jsonOk = parsed.project === "engram" && parsed.memory_types.length >= 3;
  console.log(`  Parsed: ${JSON.stringify(parsed)}  ${jsonOk ? "✓" : "✗"}\n`);

  // 3. embed — 3 texts
  console.log("[ 3 ] embed — text-embedding-v4, 3 texts, dim 1024");
  const texts = [
    "The user prefers dark mode in all applications.",
    "The user switched their project database from Postgres to SQLite.",
    "Hypothesis: the user cuts scope aggressively near deadlines.",
  ];
  const vectors = await embed(texts, 1024);
  const embedOk = vectors.length === 3 && vectors.every((v) => v.length === 1024);
  for (let i = 0; i < texts.length; i++) {
    console.log(
      `  [${i}] dim=${vectors[i].length} first3=[${vectors[i].slice(0, 3).map((v) => v.toFixed(4)).join(", ")}]`,
    );
  }
  console.log(`  ${embedOk ? "✓ all 1024-dim" : "✗ wrong dims/count"}\n`);

  // 4. Usage stats — per model
  const stats = getUsageStats();
  console.log("[ 4 ] Cumulative usage");
  console.log(`  Total: calls=${stats.calls} prompt=${stats.promptTokens} completion=${stats.completionTokens} total=${stats.totalTokens}`);
  for (const [model, u] of Object.entries(stats.byModel)) {
    console.log(`  ${model}: calls=${u.calls} prompt=${u.promptTokens} completion=${u.completionTokens} total=${u.totalTokens}`);
  }

  const allOk = chatOk && jsonOk && embedOk && stats.totalTokens > 0;
  console.log(`\n${allOk ? "✓ smoke passed" : "✗ smoke FAILED"}`);
  process.exit(allOk ? 0 : 1);
}

main().catch((err) => {
  console.error("smoke error:", err);
  process.exit(1);
});
