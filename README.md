# Black Synapse

AI data ingestion and perception system for the **Desky / AtlasAI** robot. Ingests data from multiple sources (Notion, Gmail, Drive, Slack, cameras, QR codes) into a Qdrant vector database to power RAG capabilities for the robot's LLM.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Docker Services                      │
│  Qdrant · Ollama · n8n · n8n-worker · Redis             │
│  ASR API · Kokoro TTS · DeepFace                        │
└─────────────────────────────────────────────────────────┘
         ▲                          ▲
         │ POST /ingest             │ face recognition
┌────────┴────────┐        ┌────────┴────────┐
│  Worker (FastAPI)│        │  Perception     │
│  port 8000      │        │  (cameras)      │
└─────────────────┘        └─────────────────┘
         ▲
         │ run manually
┌────────┴────────────────────────────────────┐
│  ASR: wake_word.py  ·  TTS: speaker_api.py  │
│  Action Servos: PCA9685 servo controller    │
└─────────────────────────────────────────────┘
```

## Service Ports

| Service       | Port  |
|---------------|-------|
| Worker        | 8000  |
| Qdrant        | 6333  |
| n8n           | 5678  |
| Ollama        | 11434 |
| ASR           | 8002  |
| Kokoro TTS    | 8880  |
| DeepFace      | 5000  |
| Redis         | 6379  |

## Quick Start

### 1. Start Docker services

```bash
docker-compose up -d
```

This starts: Qdrant, Ollama (+ model init), n8n, n8n-worker, Redis, ASR API, Kokoro TTS, DeepFace.

> **Note:** Postgres and the Worker are commented out in `docker-compose.yml` and must be run manually (see below).

### 2. Start the ingestion worker (manually)

```bash
cd worker
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Requires a `worker/.env` file:
```
OPENAI_API_KEY=...
QDRANT_URL=http://localhost:6333
POSTGRES_URL=postgresql://...
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
```

### 3. Start the microphone (wake word)

```bash
cd ASR
pip install -r requirements.txt
python wake_word.py
```

Listens for the wake word via openWakeWord, records audio, and writes WAV files to `./ASR/` (mounted as `/data/asr` in n8n).

### 4. Start the TTS speaker

```bash
cd TTS
pip install -r requirements.txt
python speaker_api.py
```

FastAPI wrapper around Kokoro TTS, runs on port 8880.

### 5. Configure n8n

1. Open n8n at `http://localhost:5678`
2. Import workflows from `n8n/workflows/`
3. If services are unreachable inside Docker, replace `localhost` with `host.docker.internal`

**Main workflow flow:**
```
Local File Trigger (/data/asr)
        │
        ▼
  Read WAV from disk
        │
        ▼
  ASR: Whisper (port 8002)
        │
        ▼
  LLM: Ollama (qwen2.5:7b)  ◀── Chat message trigger
  • memory
  • RAG search tool
        │
        ▼
  TTS: Kokoro (port 8880)
        │
        ▼
     Speaker
```

## Subsystems

### Ingestion Worker (`worker/`)

FastAPI ETL engine. n8n POSTs a normalized document → `POST /ingest` → pipeline:
1. SHA-256 content hash (skips if unchanged, idempotent)
2. Token-based chunking (~500 tokens, 50-token overlap, `cl100k_base`)
3. Batched OpenAI embeddings
4. Upsert to Qdrant collection `atlasai_documents`
5. Upsert metadata to Postgres `documents` table

Unified document schema: `doc_id, source, title, uri, text, author, created_at, updated_at`

### ASR (`ASR/`)

- `wake_word.py` — microphone listener; on wake word, records audio and writes WAV to `./ASR/`
- `asr_server.py` — `POST /transcribe` accepts raw WAV bytes, runs Whisper `base`, returns transcript

### TTS (`TTS/`)

- `speaker_api.py` — FastAPI wrapper for Kokoro TTS
- Docker image: `ghcr.io/remsky/kokoro-fastapi-cpu:latest` on port 8880

### Action Servos (`action_servos/`)

PCA9685-based servo controller for the robot's arm, head (pan/tilt), and ear over I2C.

```bash
python -m action_servos [center|arm|head|ear]
```

### Perception (`perception/`)

Camera utilities (`probe_cameras.py`, `capture_frame.py`). DeepFace runs in Docker for face recognition at port 5000.

### n8n Workflows (`n8n/workflows/`)

Low-code ETL: polls/listens to a source → normalizes to unified schema → `POST /ingest` on the worker.

### Qdrant (`qdrant/`)

Standalone scripts for managing the vector DB: `qdrant_ingestion.py`, `qdrant_search.py`, `merge_qdrant_db.py`.

Collections:
- `atlasai_documents` — main knowledge base
- `atlasai_chat_memory` — consolidated chat session embeddings

## Ollama Models

Models are pulled automatically on first `docker-compose up` via `ollama/init-models.sh`:
- `qwen2.5:7b` — main chat model
- `moondream` — vision model

To add more models:
```bash
ollama run gemma3
```

> Monitor storage — models are large and will fill disk silently.

## Running Tests

```bash
cd worker
pytest tests/
pytest tests/test_utils.py
pytest tests/test_pipeline.py -k test_compute_content_hash
```

## Wake Word Customization

Modify or train custom wake word models using:
https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb

## Requirements

- Docker and Docker Compose
- Python 3.10+
- OpenAI API key (for embeddings)
