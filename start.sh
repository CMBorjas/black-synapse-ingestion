#!/bin/bash

# AtlasAI Startup Script - SPOT Robot AI System

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting AtlasAI System..."

# Ensure a clean slate by stopping any running services
if [ -f "./stop.sh" ]; then
    bash ./stop.sh
fi

# Check if .env file exists
if [ -d .env ]; then
    echo "ERROR: .env is a directory, not a file. Remove it with: rm -rf .env"
    exit 1
fi
if [ ! -f .env ]; then
    echo "WARNING: .env file not found. Creating from template (edit it to add OPENAI_API_KEY)..."
    cp env.example .env
fi

# Check if Docker is running (try with and without sudo)
if ! docker info > /dev/null 2>&1 && ! sudo docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Use sudo for docker commands if needed
DOCKER_CMD="docker"
if ! docker info > /dev/null 2>&1; then
    DOCKER_CMD="sudo docker"
fi
DOCKER_COMPOSE_CMD=""
if docker compose version > /dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
elif sudo docker compose version > /dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="sudo docker compose"
elif docker-compose version > /dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker-compose"
elif sudo docker-compose version > /dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="sudo docker-compose"
fi

# Check if Docker Compose is available
if [ -z "$DOCKER_COMPOSE_CMD" ]; then
    echo "Docker Compose is not installed."
    exit 1
fi

echo "Starting Docker services..."

# Start each service individually so a port conflict on one does not block the others
SERVICES=(redis n8n n8n-worker qdrant ollama ollama-init asr kokoro-tts deepface)
SKIPPED_SERVICES=()

for SERVICE in "${SERVICES[@]}"; do
    OUTPUT=$($DOCKER_COMPOSE_CMD up -d "$SERVICE" 2>&1)
    EXIT_CODE=$?
    if echo "$OUTPUT" | grep -qiE "port is already allocated|address already in use|bind:"; then
        echo "  WARNING: $SERVICE skipped -- port already in use"
        SKIPPED_SERVICES+=("$SERVICE")
    elif [ $EXIT_CODE -ne 0 ]; then
        echo "  WARNING: $SERVICE failed to start -- $OUTPUT"
        SKIPPED_SERVICES+=("$SERVICE")
    else
        echo "  [OK] $SERVICE started"
    fi
done

if [ ${#SKIPPED_SERVICES[@]} -gt 0 ]; then
    echo ""
    echo "Skipped services (port conflicts or errors): ${SKIPPED_SERVICES[*]}"
    echo ""
fi

# Start local Python APIs immediately (independent of Docker readiness)
echo "Starting local Python services..."
mkdir -p logs

PYTHON_CMD=$(command -v python3 2>/dev/null || command -v python 2>/dev/null)

if [ -n "$PYTHON_CMD" ]; then
    echo "  Installing Python dependencies..."
    pip install -q -r TTS/requirements.txt
    pip install -q -r ASR/requirements.txt
    pip install -q -r perception/requirements.txt
    pip install -q -r worker/requirements.txt
    echo "  [OK] Dependencies installed"

    # Perception: frame capture API (port 8089)
    nohup "$PYTHON_CMD" perception/capture_frame.py >> logs/capture_frame.log 2>&1 &
    echo "  [OK] capture_frame.py started on http://127.0.0.1:8089"

    # TTS: speaker API (port 8001)
    (cd TTS && nohup "$PYTHON_CMD" speaker_api.py >> ../logs/speaker_api.log 2>&1 &)
    echo "  [OK] speaker_api.py started on http://localhost:8001"

    # ASR: wake word listener
    nohup "$PYTHON_CMD" ASR/wake_word.py >> logs/wake_word.log 2>&1 &
    echo "  [OK] wake_word.py running in background"

    # Worker: FastAPI ingestion service (port 8000)
    if [ ! -f worker/.env ]; then
        echo "  WARNING: worker/.env not found -- embedding/vector storage will be skipped (add OPENAI_API_KEY to enable)"
    fi
    (cd worker && nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 >> ../logs/worker.log 2>&1 &)
    echo "  [OK] worker started on http://localhost:8000"
else
    echo "Python not found. Skipping local services (perception, TTS, ASR wake word, worker)."
fi

# Non-blocking health checks (informational only)
echo ""
echo "Checking service health..."

if curl -sf --max-time 3 http://localhost:6333/health > /dev/null 2>&1; then
    echo "  [OK] Qdrant ready"
else
    echo "  Qdrant -- not ready yet (may still be starting)"
fi

if curl -sf --max-time 3 http://localhost:8000/health > /dev/null 2>&1; then
    echo "  [OK] Worker ready"
else
    echo "  Worker -- not ready yet (may still be starting)"
fi

echo ""
echo "AtlasAI is starting up!"
echo ""
echo "Service URLs:"
echo "  Worker API:  http://localhost:8000"
echo "  API Docs:    http://localhost:8000/docs"
echo "  n8n:         http://localhost:5678"
echo "  Qdrant:      http://localhost:6333/dashboard"
echo "  Perception:  http://127.0.0.1:8089"
echo "  TTS:         http://localhost:8001"
echo ""
echo "Logs: $SCRIPT_DIR/logs/"
echo "To view Docker logs: $DOCKER_COMPOSE_CMD logs -f"
echo "To stop:             ./stop.sh"
