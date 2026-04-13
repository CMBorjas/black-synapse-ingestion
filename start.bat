@echo off
REM AtlasAI Startup Script for Windows - SPOT Robot AI System

echo Starting AtlasAI System for SPOT Robot...

REM Check if .env file exists
if not exist .env (
    echo .env file not found. Creating from template...
    copy env.example .env
    echo Please edit .env file with your configuration before continuing.
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

echo Starting services with Docker Compose...

REM Start the services
docker-compose up -d

echo Waiting for services to be ready...

REM Wait for services to be healthy
timeout /t 10 /nobreak >nul

REM Start local Python APIs (perception, TTS, ASR wake word)
cd /d "%~dp0"
if not exist logs mkdir logs 2>nul
where python >nul 2>&1
if %errorlevel% equ 0 (
    echo Starting local Python services...
    start "Capture Frame" /b python perception\capture_frame.py
    echo    • capture_frame.py (perception) -^> http://127.0.0.1:8089
    start "Speaker API" /b cmd /c "cd /d %~dp0TTS && python speaker_api.py"
    echo    • speaker_api.py (TTS) -^> http://localhost:8001
    start "Wake Word" /b python ASR\wake_word.py
    echo    • wake_word.py (ASR) running in background
) else (
    echo Python not found. Skipping local APIs (perception, TTS, ASR wake word).
)

REM Check service health
echo Checking service health...

REM Check worker health
curl -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo Worker service is not responding yet. It may take a few more minutes to start.
) else (
    echo Worker service is healthy
)

REM Check PostgreSQL
docker-compose exec -T postgres pg_isready -U postgres >nul 2>&1
if errorlevel 1 (
    echo PostgreSQL is not ready yet
) else (
    echo PostgreSQL is ready
)


echo.
echo AtlasAI System for SPOT Robot is starting up!
echo.
echo Service URLs:
echo    • Worker API: http://localhost:8000
echo    • API Docs: http://localhost:8000/docs
echo    • n8n Interface: http://localhost:5678

echo    • Perception (capture frame): http://127.0.0.1:8089
echo    • TTS (speaker API): http://localhost:8001
echo.
echo Next steps:
echo    1. Configure your OpenAI API key in .env
echo    2. Import n8n workflows from n8n/workflows/
echo    3. Set up data source integrations
echo    4. Test the API endpoints
echo.
echo For more information, see README.md
echo.
echo To view logs: docker-compose logs -f
echo To stop services: docker-compose down
pause
