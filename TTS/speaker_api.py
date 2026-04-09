"""
Speaker API (Barge-in / Override capable)
----------------------------------------
FastAPI server that accepts WAV/MP3 and plays them with pygame.
Adds:
- POST /stop: immediately stop playback
- POST /interrupt: stop playback + increment epoch, returns epoch
- POST /play?epoch=N: only plays if epoch matches current epoch (prevents stale audio)
- Non-blocking playback (play runs in background thread)
"""

from __future__ import annotations

import os
import platform
import threading
import uuid
from pathlib import Path
from typing import Optional

import pygame
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

SUPPORTED_SUFFIXES = {".wav", ".mp3"}
SUPPORTED_MIME_TYPES = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
}
RECEIVED_DIR = Path(os.environ.get("RECEIVED_AUDIO_DIR", "received_audio"))

app = FastAPI(title="Kokoro Speaker API", version="2.0.0")

_pygame_initialized = False

# ---- Barge-in state ----
_state_lock = threading.Lock()
_current_epoch = 0          # increments on every interrupt
_current_play_id: Optional[str] = None
_stop_event = threading.Event()


def _init_pygame() -> None:
    global _pygame_initialized
    if not _pygame_initialized:
        preferred_device = os.getenv("AUDIO_DEVICE", "Built-in Audio Digital Stereo (HDMI)")
        try:
            pygame.mixer.init(devicename=preferred_device)
        except pygame.error:
            # Preferred device not available — fall back to system default
            try:
                pygame.mixer.init()
            except pygame.error as e:
                raise RuntimeError(
                    f"Failed to initialize pygame mixer: {e}. "
                    "Make sure audio drivers are available."
                ) from e
        _pygame_initialized = True


def _ensure_tmp_dir() -> None:
    RECEIVED_DIR.mkdir(parents=True, exist_ok=True)


async def _store_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if not suffix and upload.content_type in SUPPORTED_MIME_TYPES:
        suffix = SUPPORTED_MIME_TYPES[upload.content_type]

    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or upload.content_type}'. Only .wav or .mp3 are accepted.",
        )

    tmp_name = f"{uuid.uuid4().hex}{suffix}"
    tmp_path = RECEIVED_DIR / tmp_name

    with tmp_path.open("wb") as buffer:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    if tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return tmp_path


async def _store_stream(request: Request) -> Path:
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip()
    suffix = SUPPORTED_MIME_TYPES.get(content_type)

    if not suffix:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported or missing Content-Type '{content_type}'. "
                   "Use multipart form field 'file' or set Content-Type to audio/wav or audio/mpeg.",
        )

    tmp_name = f"{uuid.uuid4().hex}{suffix}"
    tmp_path = RECEIVED_DIR / tmp_name

    bytes_written = 0
    with tmp_path.open("wb") as buffer:
        async for chunk in request.stream():
            if not chunk:
                continue
            buffer.write(chunk)
            bytes_written += len(chunk)

    if bytes_written == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded body is empty.")

    return tmp_path


def _stop_now() -> None:
    """Immediately stop current playback."""
    if not _pygame_initialized:
        return
    _stop_event.set()
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass


def _background_play(file_path: Path, play_id: str) -> None:
    """
    Play audio in background.
    Stops early if:
      - /stop called (_stop_event set)
      - a new play starts (play_id changed)
    """
    global _current_play_id

    try:
        if not _pygame_initialized:
            _init_pygame()

        # Clear stop flag for this run
        _stop_event.clear()

        # Start playback
        pygame.mixer.music.load(str(file_path))
        pygame.mixer.music.play()

        # Poll until done or cancelled
        while True:
            with _state_lock:
                still_current = (_current_play_id == play_id)
            if not still_current:
                _stop_now()
                return

            if _stop_event.is_set():
                _stop_now()
                return

            if not pygame.mixer.music.get_busy():
                return

            pygame.time.wait(50)

    except Exception:
        # Don't crash server thread; just stop playback
        try:
            _stop_now()
        except Exception:
            pass
    finally:
        # cleanup file
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.on_event("startup")
def startup_event() -> None:
    _ensure_tmp_dir()
    _init_pygame()


@app.get("/healthz")
async def healthz():
    with _state_lock:
        epoch = _current_epoch
        play_id = _current_play_id
    return {
        "status": "ok" if _pygame_initialized else "not_ready",
        "player": "pygame",
        "platform": platform.system(),
        "initialized": _pygame_initialized,
        "epoch": epoch,
        "playing": bool(play_id) and pygame.mixer.music.get_busy() if _pygame_initialized else False,
    }


@app.post("/stop")
async def stop_audio():
    """Stop current playback immediately."""
    if not _pygame_initialized:
        raise HTTPException(status_code=503, detail="Audio player not ready")
    _stop_now()
    with _state_lock:
        # invalidate current play id so background loop exits
        global _current_play_id
        _current_play_id = None
    return JSONResponse({"status": "stopped"})


@app.post("/interrupt")
async def interrupt():
    """
    Barge-in endpoint:
    - stops playback
    - increments epoch (invalidates stale future plays)
    """
    if not _pygame_initialized:
        raise HTTPException(status_code=503, detail="Audio player not ready")

    _stop_now()

    global _current_epoch, _current_play_id
    with _state_lock:
        _current_epoch += 1
        _current_play_id = None
        epoch = _current_epoch

    return JSONResponse({"status": "interrupted", "epoch": epoch})


@app.post("/play")
async def play_audio(
    background_tasks: BackgroundTasks,
    request: Request,
    file: Optional[UploadFile] = File(None),
    epoch: int = Query(..., description="Epoch token returned by /interrupt. Must match current epoch."),
):
    """
    Plays audio only if epoch matches current epoch.
    Non-blocking: returns immediately while audio plays in background.
    """
    if not _pygame_initialized:
        raise HTTPException(status_code=503, detail="Audio player not ready")

    # Store audio to disk
    if file is not None:
        stored_path = await _store_upload(file)
    else:
        stored_path = await _store_stream(request)

    # Epoch gate: refuse stale audio
    with _state_lock:
        if epoch != _current_epoch:
            stored_path.unlink(missing_ok=True)
            raise HTTPException(status_code=409, detail=f"Stale audio (epoch={epoch}, current={_current_epoch}).")

        # New play id becomes the only valid one
        global _current_play_id
        play_id = uuid.uuid4().hex
        _current_play_id = play_id

    # Start playback in background thread
    background_tasks.add_task(_background_play, stored_path, play_id)
    return JSONResponse({"status": "playing", "epoch": epoch, "play_id": play_id})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "speaker_api:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8001")),
        reload=False,
    )
