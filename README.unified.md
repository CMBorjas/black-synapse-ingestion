# Black Synapse — Senior Design Project

Unified deployment for the **Lynx** embodied AI robot at CU Denver.

## Repositories in this workspace

| Directory | Purpose |
|---|---|
| `black-synapse-ingestion/` | AI backend — RAG pipeline, Qdrant, Ollama, n8n, ASR, TTS, DeepFace |
| `black-synapse-app/` | Auth portal — team members connect their OAuth accounts (Google, etc.) into n8n |

## How the system fits together

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Browser                             │
│              React App  ·  http://localhost:3000                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ /api/*
                           ▼
              ┌────────────────────────┐
              │  Express (port 4000)   │
              │  login · OAuth flows   │
              └──────────┬─────────────┘
                         │ POST /api/v1/credentials
                         ▼
              ┌────────────────────────┐
              │   n8n  (port 5678)     │  ◀── ASR WAV files from robot mic
              │  stores OAuth tokens   │
              │  runs automations      │
              └──────────┬─────────────┘
                         │ POST /ingest · POST /acknowledge
                         ▼
              ┌────────────────────────┐
              │  FastAPI worker        │  ← run manually (see below)
              │  port 8000             │
              │  Qdrant · Ollama       │
              └────────────────────────┘
```

---

## Quick Start (Docker — recommended)

### 1. Set up environment variables

```bash
# Auth app credentials (Google OAuth, n8n API key, etc.)
cp black-synapse-app/lynx-auth-app/.env.example black-synapse-app/.env
# Edit black-synapse-app/.env and fill in all values

# Ingestion backend credentials (OpenAI, Qdrant, Postgres, etc.)
cp black-synapse-ingestion/.env.example black-synapse-ingestion/.env
# Edit black-synapse-ingestion/.env and fill in all values
```

> ⚠️ **Change `REGISTER_SECRET`** before sharing or deploying. The default
> in `.env.example` is a placeholder — anyone with that value can create
> accounts on your instance. Pick a new random string.

### 2. Start all Docker services

```bash
# From this SeniorDesign/ directory:
docker compose up -d
```

This starts: **Redis, Qdrant, Ollama, n8n, n8n-worker, ASR, Kokoro TTS, DeepFace, Lynx Express server, Lynx React frontend.**

Open the dashboard at **http://localhost:3000**

### 3. Start the FastAPI ingestion worker (manually)

The worker is kept outside Docker because it needs direct hardware access (GPU, microphone, I2C servos):

```bash
cd black-synapse-ingestion/worker
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Requires a `black-synapse-ingestion/worker/.env`:
```
OPENAI_API_KEY=...
QDRANT_URL=http://localhost:6333
POSTGRES_URL=postgresql://...
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
```

### 4. Create your first user account

```bash
curl -X POST http://localhost:4000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your-name",
    "password": "your-password",
    "registerSecret": "YOUR_REGISTER_SECRET"
  }'
```

### 5. Configure n8n

1. Open **http://localhost:5678**
2. Import workflows from `black-synapse-ingestion/n8n/workflows/`
3. In n8n: **Settings → API → Enable Public API** → generate an API key
4. Paste that key into `black-synapse-app/.env` as `N8N_API_KEY`

---

## Service Ports

| Service | Port | Notes |
|---|---|---|
| React frontend | 3000 | Via nginx |
| Express server | 4000 | Auth + OAuth |
| n8n | 5678 | Workflow automation |
| FastAPI worker | 8000 | Run manually |
| ASR | 8002 | Whisper transcription |
| Kokoro TTS | 8880 | Text-to-speech |
| DeepFace | 5000 | Face recognition |
| Qdrant | 6333 | Vector database |
| Ollama | 11434 | Local LLM |
| Redis | 6379 | n8n queue |

---

## Stopping

```bash
# Stop all containers (data preserved)
docker compose down

# Stop and wipe all data volumes
docker compose down -v
```

---

## Running in development (without Docker)

See individual READMEs:
- `black-synapse-ingestion/README.md` — AI backend setup
- `black-synapse-app/lynx-auth-app/README.md` — Auth portal dev setup

---

## Adding a new OAuth service

See `black-synapse-app/lynx-auth-app/README.md → "Adding a new service"`.
