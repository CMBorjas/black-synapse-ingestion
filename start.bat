@echo off
setlocal enabledelayedexpansion
REM AtlasAI Startup Script for Windows - SPOT Robot AI System

REM Save script dir before any loops (avoids %~dp0 issues inside blocks)
set SCRIPT_DIR=%~dp0

echo Starting AtlasAI System for SPOT Robot...

REM Check if .env file exists
if not exist "%SCRIPT_DIR%.env" (
    echo .env file not found. Creating from template...
    copy "%SCRIPT_DIR%env.example" "%SCRIPT_DIR%.env"
    echo Please edit .env with your configuration before continuing.
    echo    Required: OPENAI_API_KEY
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Docker is not running. Please start Docker and try again.
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo Docker Compose is not installed. Please install Docker Compose and try again.
    pause
    exit /b 1
)

echo Starting Docker services...

REM Start each service individually so a port conflict on one does not block the others
set SKIPPED=

for %%S in (redis n8n n8n-worker qdrant ollama ollama-init asr kokoro-tts deepface) do (
    docker-compose up -d %%S >"%TEMP%\dc_out.txt" 2>&1
    set DC_EXIT=!errorlevel!
    findstr /i "port is already allocated address already in use bind:" "%TEMP%\dc_out.txt" >nul 2>&1
    if not errorlevel 1 (
        echo    WARNING: %%S skipped -- port already in use
        set SKIPPED=!SKIPPED! %%S
    ) else if !DC_EXIT! neq 0 (
        echo    WARNING: %%S failed to start
        set SKIPPED=!SKIPPED! %%S
    ) else (
        echo    [OK] %%S started
    )
)

if defined SKIPPED (
    echo.
    echo Skipped services ^(port conflicts or errors^):!SKIPPED!
    echo.
)

REM Start local Python APIs immediately (no wait needed - they are independent of Docker)
echo Starting local Python services...
if not exist "%SCRIPT_DIR%logs" mkdir "%SCRIPT_DIR%logs"

REM Pre-resolve TTS dir outside any blocks to avoid nested-quote parse errors
set "TTS_DIR=%SCRIPT_DIR%TTS"

where python >nul 2>&1
set HAVE_PYTHON=!errorlevel!

if !HAVE_PYTHON! equ 0 (
    echo Installing Python dependencies...
    pip install -q -r "%SCRIPT_DIR%TTS\requirements.txt"
    pip install -q -r "%SCRIPT_DIR%ASR\requirements.txt"
    pip install -q -r "%SCRIPT_DIR%perception\requirements.txt"
    echo    [OK] Dependencies installed

    start "Capture Frame" /b python "%SCRIPT_DIR%perception\capture_frame.py"
    echo    [OK] capture_frame.py started on http://127.0.0.1:8089

    start "Speaker API" /d "!TTS_DIR!" /b python speaker_api.py
    echo    [OK] speaker_api.py started on http://localhost:8001

    start "Wake Word" /b python "%SCRIPT_DIR%ASR\wake_word.py"
    echo    [OK] wake_word.py running in background
) else (
    echo Python not found. Skipping local services ^(perception, TTS, ASR wake word^).
)

REM Non-blocking health checks (informational only)
echo.
echo Checking service health...
curl -s --max-time 3 http://localhost:6333/health >nul 2>&1
if errorlevel 1 (echo    Qdrant  -- not ready yet) else (echo    [OK] Qdrant ready)

curl -s --max-time 3 http://localhost:8000/health >nul 2>&1
if errorlevel 1 (echo    Worker  -- not ready yet) else (echo    [OK] Worker ready)

echo.
echo AtlasAI is starting up!
echo.
echo Service URLs:
echo    Worker API:    http://localhost:8000
echo    API Docs:      http://localhost:8000/docs
echo    n8n:           http://localhost:5678
echo    Qdrant:        http://localhost:6333/dashboard
echo    Perception:    http://127.0.0.1:8089
echo    TTS:           http://localhost:8001
echo.
echo To view logs:    docker-compose logs -f
echo To stop:         docker-compose down
pause
