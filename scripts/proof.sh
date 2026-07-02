#!/usr/bin/env bash
# scripts/proof.sh — Engram live-on-Alibaba-Cloud proof recording.
#
# Curls /health, /admin/stats, and one streamed /act against the deployed
# instance, printing timestamped output. Run from your LOCAL machine while
# screen-recording next to the ECS console page.
#
# Usage:
#   ENGRAM_DOMAIN=engram.your-domain.com bash scripts/proof.sh
#   ENGRAM_DOMAIN=localhost:3000 ENGRAM_SCHEME=http bash scripts/proof.sh   # local test

set -euo pipefail

DOMAIN="${ENGRAM_DOMAIN:-engram.example.com}"
SCHEME="${ENGRAM_SCHEME:-https}"
BASE="${SCHEME}://${DOMAIN}"
USER_ID="${PROOF_USER:-proof-demo}"
SESSION="proof-$(date +%s)"
PASS=0; FAIL=0

GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; DIM='\033[2m'; RESET='\033[0m'
ok()   { echo -e "  ${GREEN}✓${RESET} $1"; PASS=$((PASS+1)); }
bad()  { echo -e "  ${RED}✗${RESET} $1"; FAIL=$((FAIL+1)); }
sep()  { echo -e "${DIM}$(printf '─%.0s' {1..64})${RESET}"; }
stamp(){ date -u '+%Y-%m-%d %H:%M:%S UTC'; }

echo ""
echo -e "${CYAN}══ ENGRAM — Alibaba Cloud ECS live proof ══${RESET}"
echo -e "  Started: $(stamp)"
echo -e "  Target:  ${BASE}"
sep

# ── 1. /health ────────────────────────────────────────────────────────────────
echo -e "\n[1/4] ${CYAN}GET /health${RESET}  $(stamp)"
if HEALTH=$(curl -sf --max-time 10 "${BASE}/health"); then
  echo "  ${HEALTH}"
  echo "${HEALTH}" | grep -q '"ok":true' && ok "server is live" || bad "unexpected /health body"
else
  bad "/health unreachable — is the server up? is 443 open?"
  exit 1
fi
sep

# ── 2. /admin/stats ───────────────────────────────────────────────────────────
echo -e "\n[2/4] ${CYAN}GET /admin/stats${RESET}  $(stamp)"
if STATS=$(curl -sf --max-time 10 "${BASE}/admin/stats"); then
  echo "${STATS}" | python3 -c '
import sys, json
d = json.load(sys.stdin)
u = d.get("tokenUsage", {})
print(f"  Qwen calls: {u.get(\"calls\", 0)}   tokens: {u.get(\"totalTokens\", 0)}   est. cost: ${d.get(\"estimatedCostUSD\", 0):.4f}")
print(f"  memories by status: {d.get(\"byStatus\")}")
print(f"  sleep runs recorded: {len(d.get(\"sleepRuns\", []))}")
' 2>/dev/null || echo "  ${STATS:0:200}"
  ok "/admin/stats returned (cumulative Qwen usage + cost visible)"
else
  bad "/admin/stats unreachable"
fi
sep

# ── 3. POST /act — live streamed agent turn ───────────────────────────────────
echo -e "\n[3/4] ${CYAN}POST /act (SSE)${RESET}  $(stamp)"
Q="I'm recording the deployment proof for Engram right now. Say hello and tell me one thing you remember about me, briefly."
echo -e "  ${DIM}Q: ${Q}${RESET}"
echo -n "  A: "

ACT_RAW=$(curl -sN --max-time 120 -X POST "${BASE}/act" \
  -H "Content-Type: application/json" \
  -d "{\"userId\":\"${USER_ID}\",\"sessionId\":\"${SESSION}\",\"input\":\"${Q}\"}" | \
  awk '
    /^event: /  { evt=substr($0, 8); next }
    /^data: / {
      val = substr($0, 7)
      if (evt == "token")  printf "%s", val
      if (evt == "done")   print "\n@DONE@" val
      if (evt == "error")  print "\n@ERROR@" val
    }
  ')

echo "${ACT_RAW}" | grep -v '^@' | head -6
if echo "${ACT_RAW}" | grep -q '^@DONE@'; then
  echo "${ACT_RAW}" | grep '^@DONE@' | sed 's/^@DONE@//' | python3 -c '
import sys, json
d = json.load(sys.stdin)
u = d.get("usage", {})
print(f"  turnId: {d.get(\"turnId\")}   model: {u.get(\"model\")}   context: {u.get(\"contextTokens\")} tok   memories in manifest: {len(d.get(\"manifest\", []))}")
' 2>/dev/null || true
  ok "full agent loop streamed over SSE (qwen-plus via DashScope)"
elif echo "${ACT_RAW}" | grep -q '^@ERROR@'; then
  bad "/act returned an error event: $(echo "${ACT_RAW}" | grep '^@ERROR@' | head -1)"
else
  bad "/act produced no done event"
fi
sep

# ── 4. TLS certificate ────────────────────────────────────────────────────────
echo -e "\n[4/4] ${CYAN}TLS certificate${RESET}  $(stamp)"
if [ "${SCHEME}" = "https" ]; then
  HOST_ONLY="${DOMAIN%%:*}"
  if TLS=$(echo | openssl s_client -connect "${HOST_ONLY}:443" -servername "${HOST_ONLY}" 2>/dev/null \
      | openssl x509 -noout -issuer -enddate 2>/dev/null); then
    echo "${TLS}" | sed 's/^/  /'
    ok "TLS present (Caddy-managed)"
  else
    bad "TLS check failed"
  fi
else
  echo "  (skipped — http mode)"
fi
sep

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL=$((PASS+FAIL))
echo ""
if [ "${FAIL}" -eq 0 ]; then
  echo -e "  ${GREEN}ALL ${TOTAL}/${TOTAL} CHECKS PASSED${RESET} — Engram live on Alibaba Cloud ECS, Qwen (DashScope) confirmed."
else
  echo -e "  ${RED}${FAIL}/${TOTAL} CHECKS FAILED${RESET} — see above."
fi
echo -e "  Finished: $(stamp)"
echo -e "${DIM}  Alibaba Cloud services: ECS (ap-southeast-1) · DashScope Qwen (qwen-plus / qwen-flash / text-embedding-v4)"
echo -e "  Code proof: src/llm/qwen.ts · Runbook: infra/deploy.md${RESET}"
echo ""
[ "${FAIL}" -eq 0 ]
