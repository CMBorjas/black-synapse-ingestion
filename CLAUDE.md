# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Black Synapse** is the AI data ingestion and perception system for the Desky/AtlasAI robot. It ingests data from multiple sources (Notion, Gmail, Drive, Slack, cameras, QR codes) into a Qdrant vector database to power RAG capabilities for the robot's LLM.

Start all services (Docker + Python):
```bash
./start.sh
```

Stop all services and free up ports:
```bash
./stop.sh
```

Manage as a systemd service:
```bash
sudo systemctl [start|stop|status|restart] atlasai
```

Manual Docker commands:
```bash
docker-compose up -d
docker-compose down
```

The worker (FastAPI) and postgres are **commented out** in `docker-compose.yml` and must be run manually:
```bash
cd worker
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Start ASR (wake word + microphone, runs outside Docker):
```bash
cd ASR && pip install -r requirements.txt && python wake_word.py
```

Start TTS speaker (runs outside Docker):
```bash
cd TTS && pip install -r requirements.txt && python speaker_api.py
```

## Running Tests

Tests live in `worker/tests/`. Run from the `worker/` directory:
```bash
cd worker
pytest tests/
pytest tests/test_utils.py          # single file
pytest tests/test_pipeline.py -k test_compute_content_hash  # single test
```

## Service Ports

| Service   | Port  |
|-----------|-------|
| Worker    | 8000  |
| Qdrant    | 6333  |
| n8n       | 5678  |
| Ollama    | 11434 |
| ASR       | 8002  |
| Kokoro TTS| 8880  |
| DeepFace  | 5000  |

## Environment Variables

Required in `worker/.env`:
```
OPENAI_API_KEY=...
QDRANT_URL=http://localhost:6333
POSTGRES_URL=postgresql://...
EMBEDDING_MODEL=text-embedding-3-small   # or text-embedding-3-large
EMBEDDING_DIM=1536                        # 3072 for large model
```

## Architecture

The system has several loosely coupled subsystems:

### Ingestion Worker (`worker/`)
FastAPI service that is the core ETL engine. Flow: n8n POSTs a normalized document → `POST /ingest` → `IngestionPipeline.process_document()`:
1. SHA-256 content hash for deduplication (skips if unchanged)
2. `chunk_text()` — token-based chunking, ~500 tokens, 50-token overlap via `cl100k_base`
3. `get_embedding()` — batched OpenAI embeddings
4. Upsert to Qdrant collection `atlasai_documents`
5. Upsert metadata to Postgres `documents` table

Key files:
- `worker/app/main.py` — FastAPI routes and Pydantic models
- `worker/app/pipeline.py` — `IngestionPipeline` class; all DB interactions
- `worker/app/utils.py` — `chunk_text`, `get_embedding`, and misc utilities
- `worker/app/scraper.py` — URL scraping (used by `/ingest/user` and `/analyze/qr`)
- `worker/app/qr_analyzer.py` — QR code decode + content classification

### Unified Document Schema
All ingested documents must conform to:
```json
{ "doc_id", "source", "title", "uri", "text", "author", "created_at", "updated_at" }
```
Qdrant vector payloads add `chunk_index` to this schema.

### Qdrant Collections
- `atlasai_documents` — main knowledge base
- `atlasai_chat_memory` — consolidated chat session embeddings

### Postgres Tables
- `documents` — tracks doc_id, content_hash, chunk_count, is_deleted
- `ingestion_log` — per-event audit log
- `chat_logs` — raw chat messages, marked `is_indexed` after consolidation

### Chat Memory Pipeline
`POST /chat/log` → stores raw messages → `POST /chat/memory/consolidate` groups by `session_id`, chunks + embeds, upserts to `atlasai_chat_memory`, marks rows `is_indexed=TRUE`.

### n8n Workflows (`n8n/workflows/`)
Low-code ETL. Each workflow: polls/listens to a source → normalizes to unified schema → `POST /ingest` on the worker. Import JSON workflows into n8n at `http://localhost:5678`. If connecting from inside Docker, replace `localhost` with `host.docker.internal`.

### ASR (`ASR/`)
- `wake_word.py` — microphone listener with openWakeWord; on trigger, records audio and writes WAV to `./ASR/` (which is mounted as `/data/asr` in n8n)
- `asr_server.py` — FastAPI service; `POST /transcribe` accepts raw WAV bytes, runs Whisper `base` model, returns transcript

### TTS (`TTS/`)
- `speaker_api.py` — FastAPI service wrapping Kokoro TTS
- Docker uses `ghcr.io/remsky/kokoro-fastapi-cpu:latest` on port 8880

### Action Servos (`action_servos/`)
PCA9685-based servo controller for the robot's arm, head (pan/tilt), and ear. Hardware access via I2C. CLI entry point: `python -m action_servos [center|arm|head|ear]`.

### Perception (`perception/`)
Camera utilities (`probe_cameras.py`, `capture_frame.py`). DeepFace runs in Docker for face recognition at port 5000.

### Qdrant Utilities (`qdrant/`)
Standalone scripts for managing the vector DB: `qdrant_ingestion.py`, `qdrant_search.py`, `merge_qdrant_db.py`, `combine_crawl_files.py`.

## Key Design Rules

- **Idempotency**: always compute `content_hash` before processing; skip if unchanged unless `force_reindex=True`
- **Embedding model**: configurable via `EMBEDDING_MODEL` env var; dimension must match collection config (1536 for small, 3072 for large)
- **Qdrant initialization**: retries up to 10× with exponential backoff on startup since Qdrant may not be ready immediately
- **Type hints + docstrings** are expected on all functions per the project's `.cursor/rules`
