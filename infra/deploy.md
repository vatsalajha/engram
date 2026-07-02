# Deploying Engram on Alibaba Cloud ECS

Copy-paste runbook: **ECS Ubuntu 22.04 (Singapore) · Node 20 · pm2 · Caddy auto-TLS**.
Every command runs verbatim; substitute only values in `ANGLE_BRACKETS`.

This file + [`src/llm/qwen.ts`](../src/llm/qwen.ts) are the submission's
**Alibaba Cloud proof artifacts** (ECS compute + DashScope/Qwen APIs).

---

## 0 · Provision (Alibaba Cloud console, ~5 min)

1. **ECS console → Create Instance**
   - Region: **Singapore (ap-southeast-1)** — matches the `dashscope-intl` endpoint
   - Image: **Ubuntu 22.04 64-bit** · Type: `ecs.e-c1m2.large` (2 vCPU / 4 GB) or larger
   - System disk 40 GB · assign a **public IPv4** · add your SSH key
2. **Security group** — allow inbound: `22` (your IP), `80`, `443` (0.0.0.0/0)
3. **DNS** — A record: `<DOMAIN>` → the ECS public IP
4. **DashScope key** — [Model Studio console](https://dashscope.console.aliyun.com/apiKey) → create API key

```bash
ssh root@<ECS_PUBLIC_IP>
```

## 1 · System packages + Node 20

```bash
apt-get update && apt-get install -y git curl ufw

# Node 20 via nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh"
nvm install 20 && nvm alias default 20
node -v   # v20.x

npm install -g pm2
```

## 2 · Caddy (auto-TLS reverse proxy)

```bash
apt-get install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
  | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  | tee /etc/apt/sources.list.d/caddy-stable.list
apt-get update && apt-get install -y caddy
```

## 3 · Clone + build

```bash
mkdir -p /opt && cd /opt
git clone https://github.com/vatsalajha/engram.git
cd engram
npm ci
npm run build          # typecheck (src+eval+test) + emit dist/

# Web demo (served separately or via any static host; optional on the box)
cd web && npm ci && npm run build && cd ..
```

## 4 · Environment

```bash
cp .env.example .env
nano .env
```

Required values:

```ini
DASHSCOPE_API_KEY=sk-<YOUR_REAL_KEY>
ENGRAM_QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
ENGRAM_DB_PATH=/opt/engram/data/engram.db
PORT=3000
NODE_ENV=production
ENGRAM_SCHEDULER=on          # sleep cycle runs between sessions
# REDIS_URL is OPTIONAL — leave unset for in-process rate-limit/idempotency
```

Smoke-test DashScope from the box before going further:

```bash
npm run smoke   # chat ENGRAM_OK · chatJSON round-trip · 3×1024-dim embeddings
```

## 5 · pm2

```bash
mkdir -p /var/log/engram data
pm2 start infra/ecosystem.config.cjs
pm2 save && pm2 startup     # run the printed command to persist across reboots
pm2 logs engram --lines 20  # expect: [api] Engram listening on http://localhost:3000
```

## 6 · Caddy site

```bash
sed "s/engram.example.com/<DOMAIN>/" /opt/engram/infra/Caddyfile > /etc/caddy/Caddyfile
systemctl reload caddy
# Caddy obtains the Let's Encrypt certificate automatically (~10 s)
```

## 7 · Firewall + verify

```bash
ufw allow 22 && ufw allow 80 && ufw allow 443 && ufw --force enable

curl https://<DOMAIN>/health
# {"ok":true,"service":"engram",...}
```

## 8 · Proof recording

From your **local** machine:

```bash
ENGRAM_DOMAIN=<DOMAIN> bash scripts/proof.sh
```

Record the terminal + the ECS console page side-by-side. The script prints
timestamped output for `/health`, `/admin/stats`, and a live streamed `/act`.

---

## Operations

| Task | Command |
|---|---|
| Tail logs | `pm2 logs engram` |
| Restart | `pm2 restart engram` |
| Deploy update | `cd /opt/engram && git pull && npm ci && npm run build && pm2 restart engram` |
| DB backup | `sqlite3 data/engram.db ".backup data/backup-$(date +%F).db"` |
| Force sleep cycle | `curl -X POST https://<DOMAIN>/sleep -H 'Content-Type: application/json' -d '{"userId":"demo"}'` |

**Troubleshooting**

| Symptom | Fix |
|---|---|
| `invalid_api_key` from DashScope | Key must be a Model Studio key; check `ENGRAM_QWEN_BASE_URL` uses `dashscope-intl` |
| Caddy has no certificate | DNS A record not propagated yet, or port 80 blocked in the security group |
| `/act` buffers instead of streaming | `flush_interval -1` missing from the Caddyfile `reverse_proxy` block |
| better-sqlite3 build error | `apt-get install -y build-essential python3` then `npm ci` again |

**Scale path** (documented, intentionally not built): ApsaraDB RDS PostgreSQL +
pgvector for storage · Function Compute for the sleep scheduler · Tair for
distributed rate-limit/locks (`REDIS_URL` already supported) · OSS for episodic
cold storage.
