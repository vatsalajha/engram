# Engram — Hackathon Submission

**Hackathon:** Global AI Hackathon Series with Qwen Cloud
**Track:** **Track 1 — MemoryAgent**
**Deadline:** 9 Jul 2026, 2:00 PM PDT
**Team / Author:** Vatsala Jha (vj193@scarletmail.rutgers.edu)
**Repository:** https://github.com/vatsalajha/engram (public, MIT)

---

## Description (≤150 words)

Engram is an agent with a hippocampus. After every turn it autonomously
extracts typed memories (qwen-plus), dedupes them, and supersedes contradicted
facts at write time — forgetting is timely and auditable. Every memory carries
a confidence that moves with evidence: confirmed outcomes raise it,
contradictions lower it, and below a floor hypotheses die. Between sessions
the agent "sleeps": consolidating episodic events into semantic memories,
decaying stale ones, and *inferring* new low-confidence hypotheses about the
user from raw transcripts. Retrieval fuses BM25 (SQLite FTS5) with vector
search (sqlite-vec + text-embedding-v4) via Reciprocal Rank Fusion, then packs
the highest-utility memories into a hard token budget with full provenance.
A three-arm benchmark charts the claim: accuracy climbs across sessions at a
fraction of full-history tokens, and only Engram passes forgetting-precision.
Runs entirely on Alibaba Cloud: ECS + DashScope (Qwen).

---

## Feature list

- **Autonomous accumulation** — qwen-plus extraction after every turn; no manual `remember()` required (it exists for explicit control via MCP)
- **Belief engine** — per-memory confidence updated by explicit `/feedback` and an implicit per-turn judge; append-only `belief_events` audit trail
- **Sleep cycle** — consolidation (with `derived_from` graph edges), retention-scored decay + 48h expiry, and hypothesis **inference** ("the agent dreams")
- **Write-time supersession** — batched qwen-plus verdicts; superseded memories get struck through live in the demo UI
- **Hybrid retrieval** — FTS5 BM25 ∥ sqlite-vec cosine → RRF (k=60) → optional qwen-flash rerank
- **Token-budget packer** — hard ≤B guarantee, 25% reserved for pinned/preferences, qwen-flash compression of overflow, provenance manifest
- **Memory graph** — `supersedes` / `derived_from` / `supports` / `contradicts` edges, browsable per-memory in the UI
- **Live budget dial** — drag 150↔2500 tokens mid-conversation; cumulative Qwen cost always visible
- **3-arm benchmark** — accuracy/tokens/recall@5/forgetting-precision/latency/cost per session + accuracy-vs-budget sweep
- **MCP server** — 6 tools over streamable HTTP, same engine as the REST API
- **Server-authoritative, idempotent, atomic** — content-hash idempotency, SQLite WAL transactions, status-flag deletes only

## Links

| Artifact | Link |
|---|---|
| 🎬 3-min demo video | **TODO — YouTube link** |
| 🎥 ECS proof recording | **TODO — link** (terminal `scripts/proof.sh` + ECS console side-by-side) |
| 🌐 Live instance | **TODO — https://\<domain\>/health** |
| 🧠 Alibaba Cloud API code | [`src/llm/qwen.ts`](../src/llm/qwen.ts) |
| ☁️ ECS deployment runbook | [`infra/deploy.md`](../infra/deploy.md) |
| 📊 Benchmark harness | [`eval/`](../eval/) |
| 📝 Build-journey blog post (bonus track) | **TODO — link** |

## Alibaba Cloud services used

1. **ECS** — Ubuntu 22.04, Singapore (ap-southeast-1); Node 20 + pm2 + Caddy
2. **Model Studio / DashScope** — `qwen-plus` (decide/extract/supersede/infer), `qwen-flash` (rerank/compress/grade), `text-embedding-v4` (1024-dim embeddings), all via the OpenAI-compatible endpoint

## Submission checklist

- [x] Public repo with OSS license visible (MIT)
- [x] Architecture diagram (`docs/architecture.mmd`, rendered in README)
- [ ] ~3-min demo video (public link)
- [ ] ECS proof recording
- [x] Alibaba Cloud API code file linked
- [x] Feature/functionality description
- [x] Track declared: Track 1 — MemoryAgent
- [ ] (Bonus) blog post
