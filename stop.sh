#!/bin/bash

# Black Synapse Shutdown Script
# Stops Docker containers and kills local Python services to free up ports.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Stopping Black Synapse System..."

# 1. Stop Docker containers
echo "Stopping Docker services..."
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

if [ -n "$DOCKER_COMPOSE_CMD" ]; then
    $DOCKER_COMPOSE_CMD down
    echo "  [OK] Docker services stopped"
else
    echo "  [SKIP] Docker Compose not found"
fi

# 2. Kill local Python services
echo "Killing local Python processes..."
PKILL_CMD="pkill"
# If we are not root, and pkill fails, we might need sudo
if [ "$EUID" -ne 0 ] && ! pkill -0 -f "python" > /dev/null 2>&1; then
    # This is just a check, pkill -0 doesn't kill. 
    # If we can't even signal, we might need sudo.
    true
fi

$PKILL_CMD -f "perception/capture_frame.py" || sudo $PKILL_CMD -f "perception/capture_frame.py" 2>/dev/null
$PKILL_CMD -f "TTS/speaker_api.py" || sudo $PKILL_CMD -f "TTS/speaker_api.py" 2>/dev/null
$PKILL_CMD -f "ASR/wake_word.py" || sudo $PKILL_CMD -f "ASR/wake_word.py" 2>/dev/null
$PKILL_CMD -f "uvicorn app.main:app" || sudo $PKILL_CMD -f "uvicorn app.main:app" 2>/dev/null
$PKILL_CMD -f "python.*worker" || sudo $PKILL_CMD -f "python.*worker" 2>/dev/null
$PKILL_CMD -f "python.*perception" || sudo $PKILL_CMD -f "python.*perception" 2>/dev/null
$PKILL_CMD -f "python.*TTS" || sudo $PKILL_CMD -f "python.*TTS" 2>/dev/null
$PKILL_CMD -f "python.*ASR" || sudo $PKILL_CMD -f "python.*ASR" 2>/dev/null

echo "  [OK] Python processes signaled to stop"

# 3. Force-free specific ports if still occupied
# List of ports used by the system
PORTS=(3000 4000 5000 5678 6333 6379 8000 8001 8002 8089 8880 11434)

echo "Verifying all ports are free..."
for PORT in "${PORTS[@]}"; do
    # Try to find PID using the port
    PID=$(lsof -t -i:$PORT 2>/dev/null || fuser $PORT/tcp 2>/dev/null | awk '{print $NF}')
    
    if [ -n "$PID" ]; then
        echo "  Port $PORT is still in use by PID $PID. Killing..."
        kill -9 $PID 2>/dev/null || sudo kill -9 $PID 2>/dev/null
    fi
done

echo ""
echo "All Black Synapse services have been stopped and ports cleared."
echo "You can now safely run ./start.sh again."
