import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from fastapi import Body, File, FastAPI, Query, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("perception")

app = FastAPI(title="Perception Vision Service")

# ── Tunables ──────────────────────────────────────────────────────────
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
WIDTH = int(os.getenv("CAM_WIDTH", "1280"))
HEIGHT = int(os.getenv("CAM_HEIGHT", "720"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "90"))
CAPTURE_FPS = int(os.getenv("CAPTURE_FPS", "15"))

CHANGE_THRESHOLD = float(os.getenv("CHANGE_THRESHOLD", "30.0"))
CHANGE_CHECK_INTERVAL = float(os.getenv("CHANGE_CHECK_INTERVAL", "1.0"))
MIN_PUSH_INTERVAL = float(os.getenv("MIN_PUSH_INTERVAL", "3.0"))

N8N_WEBHOOK_URL = os.getenv(
    "N8N_WEBHOOK_URL", "http://localhost:5678/webhook/perception-trigger"
)
PUSH_ENABLED = os.getenv("PUSH_ENABLED", "true").lower() == "true"

# DeepFace stream mode: when true, stream runs face analysis + identification (heavy).
# When false, uses simple capture loop — inference only via n8n Perception workflow.
DEEPFACE_STREAM = os.getenv("DEEPFACE_STREAM", "false").lower() == "true"
STREAM_FRAME_THRESHOLD = int(os.getenv("STREAM_FRAME_THRESHOLD", "5"))
STREAM_TIME_THRESHOLD = int(os.getenv("STREAM_TIME_THRESHOLD", "5"))

# State storage: JSON file (default) or Redis
STATE_BACKEND = os.getenv("STATE_BACKEND", "file")
STATE_FILE = os.getenv(
    "STATE_FILE",
    str(Path(__file__).resolve().parent.parent / "perception_state.json"),
)
STREAM_FACE_FILE = os.getenv(
    "STREAM_FACE_FILE",
    str(Path(__file__).resolve().parent.parent / "stream_face_detection.json"),
)
FACES_DB_PATH = os.getenv(
    "FACES_DB_PATH",
    str(Path(__file__).resolve().parent / "faces"),
)

# Embedding-based recognition (stores vectors, assigns user_1, user_2, ...)
EMBEDDINGS_PATH = os.getenv(
    "EMBEDDINGS_PATH",
    str(Path(__file__).resolve().parent.parent / "face_embeddings.json"),
)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Facenet512")
MAX_EMBEDDINGS_PER_USER = int(os.getenv("MAX_EMBEDDINGS_PER_USER", "20"))
_embeddings_lock = threading.Lock()

# Redis (optional, used when STATE_BACKEND=redis)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_KEY = "perception:memo"

_redis_client = None
if STATE_BACKEND == "redis":
    try:
        import redis
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
        log.info("Using Redis for state at %s", REDIS_URL)
    except Exception as exc:
        log.warning("Redis unavailable, falling back to file: %s", exc)
        STATE_BACKEND = "file"

# DeepFace for face identification and stream
_deepface = None


def _get_deepface():
    global _deepface
    if _deepface is None:
        try:
            from deepface import DeepFace
            _deepface = DeepFace
            log.info("DeepFace loaded")
        except ImportError as exc:
            log.warning("DeepFace not available: %s", exc)
    return _deepface


# ── Shared frame buffer ──────────────────────────────────────────────
@dataclass
class FrameBuffer:
    frame: Optional[np.ndarray] = None
    timestamp: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


_buf = FrameBuffer()
_cap: Optional[cv2.VideoCapture] = None
_cap_lock = threading.Lock()
_shutdown_event = threading.Event()


# ── State (file-backed) ───────────────────────────────────────────────
def _read_state_file() -> dict:
    if STATE_BACKEND == "redis" and _redis_client:
        try:
            raw = _redis_client.get(REDIS_KEY)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return {"faces": [], "scene_caption": None, "updated_at": None}
    p = Path(STATE_FILE)
    if not p.exists():
        return {"faces": [], "scene_caption": None, "updated_at": None}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to read state file: %s", exc)
        return {"faces": [], "scene_caption": None, "updated_at": None}


def _write_state_file(data: dict):
    data["updated_at"] = time.time()
    if STATE_BACKEND == "redis" and _redis_client:
        try:
            _redis_client.set(REDIS_KEY, json.dumps(data))
            return
        except Exception as exc:
            log.warning("Redis write failed, falling back to file: %s", exc)
    p = Path(STATE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_stream_face_file(data: dict):
    """Write DeepFace stream output to a separate file (no Redis)."""
    data["updated_at"] = time.time()
    p = Path(STREAM_FACE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_stream_face_file() -> dict:
    """Read DeepFace stream face state."""
    p = Path(STREAM_FACE_FILE)
    if not p.exists():
        return {"faces": [], "updated_at": None}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to read stream face file: %s", exc)
        return {"faces": [], "updated_at": None}


# ── Embedding-based face recognition ─────────────────────────────────
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1]. Same person typically >= 0.5."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    sim = dot / (norm_a * norm_b)
    return float(np.clip(sim, 0, 1))


def _load_embeddings() -> dict:
    """
    Load embedding db. Format: { user_1: { embeddings: [[...], [...], ...], first_seen: ts }, ... }.
    Migrates legacy single-embedding format to embeddings list.
    """
    p = Path(EMBEDDINGS_PATH)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        # Migrate legacy format
        for uid, entry in data.items():
            if isinstance(entry, dict) and "embedding" in entry and "embeddings" not in entry:
                entry["embeddings"] = [entry.pop("embedding", [])]
        return data
    except Exception as exc:
        log.warning("Failed to load embeddings: %s", exc)
        return {}


def _save_embeddings(data: dict):
    p = Path(EMBEDDINGS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _identify_by_embedding(face_img: np.ndarray, add_if_unknown: bool = True, detector_backend: str = "skip") -> tuple[str, float]:
    """
    Match face embedding against stored embeddings (multiple per user).
    Returns (user_id, confidence). Adds new embedding on match to learn variations.
    Uses Facenet512 by default for robustness across angle/lighting.
    """
    df = _get_deepface()
    if df is None:
        return "Unknown", 0.0
    try:
        objs = df.represent(
            img_path=face_img,
            model_name=EMBEDDING_MODEL,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True,
        )
        if not objs or len(objs) == 0:
            return "Unknown", 0.0
        emb = np.array(objs[0]["embedding"], dtype=np.float64)
    except Exception as exc:
        log.debug("Embedding extraction failed: %s", exc)
        return "Unknown", 0.0

    with _embeddings_lock:
        db = _load_embeddings()
        best_id = None
        best_sim = 0.0

        for user_id, entry in db.items():
            emb_list = entry.get("embeddings", entry.get("embedding", []))
            if not isinstance(emb_list, list):
                emb_list = [emb_list]
            for stored_raw in emb_list:
                stored = np.array(stored_raw, dtype=np.float64)
                if len(stored) != len(emb):
                    continue
                sim = _cosine_similarity(emb, stored)
                if sim > best_sim and sim >= SIMILARITY_THRESHOLD:
                    best_sim = sim
                    best_id = user_id

        if best_id is not None:
            # Add this embedding to the user to learn more variations (angle/lighting)
            entry = db[best_id]
            emb_list = entry.get("embeddings", [])
            if not emb_list:
                emb_list = [entry.get("embedding", [])] if entry.get("embedding") else []
            if emb.tolist() not in emb_list and len(emb_list) < MAX_EMBEDDINGS_PER_USER:
                emb_list.append(emb.tolist())
                entry["embeddings"] = emb_list
                if "embedding" in entry:
                    del entry["embedding"]
                _save_embeddings(db)
            return best_id, best_sim

        if add_if_unknown:
            next_num = max(
                (int(k.replace("user_", "")) for k in db if k.startswith("user_") and k[5:].isdigit()),
                default=0,
            ) + 1
            new_id = f"user_{next_num}"
            db[new_id] = {"embeddings": [emb.tolist()], "first_seen": time.time()}
            _save_embeddings(db)
            log.info("New face registered: %s", new_id)
            return new_id, 1.0

    return "Unknown", 0.0


# ── Camera helpers ────────────────────────────────────────────────────
def _open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    for _ in range(5):
        cap.read()
        time.sleep(0.02)
    return cap


def _encode_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> Optional[bytes]:
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpg.tobytes() if ok else None


# ── DeepFace stream loop (like DeepFace.stream() + JSON state) ──────────
def _grab_facial_areas(img: np.ndarray, detector_backend: str = "opencv", threshold: int = 130):
    """Detect faces, return list of (x, y, w, h)."""
    df = _get_deepface()
    if df is None:
        return []
    try:
        face_objs = df.extract_faces(
            img_path=img,
            detector_backend=detector_backend,
            expand_percentage=0,
            enforce_detection=False,
        )
        return [
            (int(f["facial_area"]["x"]), int(f["facial_area"]["y"]),
             int(f["facial_area"]["w"]), int(f["facial_area"]["h"]))
            for f in face_objs
            if f["facial_area"]["w"] > threshold
        ]
    except Exception:
        return []


def _draw_overlay(img: np.ndarray, faces_data: list, countdown: Optional[str] = None):
    """Draw bounding boxes and labels on frame."""
    for (x, y, w, h), data in faces_data:
        color = (67, 67, 67)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label_parts = []
        if data.get("name"):
            label_parts.append(f"{data['name']} ({data.get('confidence', 0):.0%})")
        if data.get("emotion"):
            label_parts.append(data["emotion"])
        if data.get("age"):
            label_parts.append(f"{int(data['age'])}y")
        if data.get("gender"):
            label_parts.append(data["gender"][:1])
        if label_parts:
            label = " | ".join(label_parts)
            cv2.putText(img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if countdown:
        h, w = img.shape[:2]
        cv2.rectangle(img, (10, 10), (90, 50), (67, 67, 67), -1)
        cv2.putText(img, countdown, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def _deepface_stream_loop():
    """Run DeepFace stream logic: detect face 5 frames -> analyze -> show 5s -> save to JSON."""
    df = _get_deepface()
    if df is None:
        log.warning("DeepFace stream disabled: DeepFace not available")
        _simple_capture_loop()
        return
    detector_backend = "opencv"
    frame_threshold = max(STREAM_FRAME_THRESHOLD, 1)
    time_threshold = max(STREAM_TIME_THRESHOLD, 1)

    num_frames_with_faces = 0
    freezed_img = None
    freeze = False
    tic = time.time()
    last_faces_data = []
    last_faces_coords = []

    interval = 1.0 / CAPTURE_FPS
    while not _shutdown_event.is_set():
        with _cap_lock:
            cap = _cap
        if cap is None or not cap.isOpened():
            time.sleep(interval)
            continue

        has_frame, img = cap.read()
        if not has_frame or img is None:
            time.sleep(interval)
            continue

        raw_img = img.copy()
        if not freeze:
            faces_coords = _grab_facial_areas(img, detector_backend)
            last_faces_coords = faces_coords
        else:
            faces_coords = last_faces_coords
        num_frames_with_faces = num_frames_with_faces + 1 if faces_coords else 0
        freeze = num_frames_with_faces > 0 and num_frames_with_faces % frame_threshold == 0

        display_img = raw_img.copy()
        if freeze:
            # Run analysis
            img = raw_img.copy()
            faces_data = []
            for (x, y, w, h) in faces_coords:
                detected_face = img[y : y + h, x : x + w]
                # Recognition (embedding-based: user_1, user_2, ...)
                name, confidence = _identify_by_embedding(detected_face, add_if_unknown=True)
                # Demography
                emotion, age, gender = "neutral", None, None
                try:
                    dem = df.analyze(
                        detected_face,
                        actions=("age", "gender", "emotion"),
                        detector_backend="skip",
                        enforce_detection=False,
                        silent=True,
                    )
                    if dem and len(dem) > 0:
                        d = dem[0]
                        emotion = d.get("dominant_emotion", "neutral")
                        age = d.get("age")
                        gender = (d.get("dominant_gender") or "unknown")[:1]
                except Exception as exc:
                    log.debug("Demography failed: %s", exc)
                faces_data.append(((x, y, w, h), {
                    "name": name, "confidence": confidence,
                    "emotion": emotion, "age": age, "gender": gender,
                }))
                last_faces_data = faces_data
            _draw_overlay(img, faces_data)
            freezed_img = img.copy()
            tic = time.time()
            log.info("DeepFace stream: froze, %d face(s) analyzed", len(faces_data))
            # Save to JSON
            state_faces = [
                {
                    "name": d["name"],
                    "emotion": d.get("emotion"),
                    "age": d.get("age"),
                    "gender": d.get("gender"),
                    "confidence": d.get("confidence", 0),
                }
                for _, d in faces_data
            ]
            _write_stream_face_file({"faces": state_faces})
        elif freezed_img is not None and time.time() - tic < time_threshold:
            # Still frozen: show countdown
            time_left = int(time_threshold - (time.time() - tic) + 1)
            freezed_img = freezed_img.copy()
            _draw_overlay(freezed_img, last_faces_data, countdown=str(time_left))
            display_img = freezed_img
        else:
            # Unfreeze
            if freezed_img is not None:
                freeze = False
                freezed_img = None
                tic = time.time()
                log.info("DeepFace stream: freeze released")
            # Live: show countdown to freeze
            if faces_coords:
                count = str(frame_threshold - (num_frames_with_faces % frame_threshold))
                for x, y, w, h in faces_coords:
                    cv2.rectangle(display_img, (x, y), (x + w, y + h), (67, 67, 67), 2)
                    cv2.putText(display_img, count, (x + w // 4, int(y + h / 1.5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            else:
                display_img = raw_img

        with _buf.lock:
            _buf.frame = display_img
            _buf.timestamp = time.time()
        time.sleep(interval)


def _simple_capture_loop(fps: int = CAPTURE_FPS):
    """Original capture loop without DeepFace stream."""
    interval = 1.0 / fps
    while not _shutdown_event.is_set():
        with _cap_lock:
            cap = _cap
        if cap is None or not cap.isOpened():
            time.sleep(interval)
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            with _buf.lock:
                _buf.frame = frame
                _buf.timestamp = time.time()
        time.sleep(interval)


# ── Change detection + webhook push thread ────────────────────────────
def _change_detection_loop():
    prev_frame: Optional[np.ndarray] = None
    last_push = 0.0

    while not _shutdown_event.is_set():
        with _buf.lock:
            frame = _buf.frame.copy() if _buf.frame is not None else None

        if frame is None:
            time.sleep(CHANGE_CHECK_INTERVAL)
            continue

        if prev_frame is not None:
            diff = _frame_diff(prev_frame, frame)
            now = time.time()

            if diff > CHANGE_THRESHOLD and (now - last_push) > MIN_PUSH_INTERVAL:
                jpg = _encode_jpeg(frame)
                if jpg is not None:
                    log.info("Scene change detected (diff=%.1f), pushing to n8n", diff)
                    try:
                        requests.post(
                            N8N_WEBHOOK_URL,
                            files={"data": ("frame.jpg", jpg, "image/jpeg")},
                            data={"diff_score": str(round(diff, 2))},
                            timeout=5,
                        )
                        last_push = now
                    except Exception as exc:
                        log.warning("Webhook push failed: %s", exc)

        prev_frame = frame
        time.sleep(CHANGE_CHECK_INTERVAL)


def _frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(ga, gb)))


# ── Lifecycle ─────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    global _cap
    _cap = _open_camera()
    Path(FACES_DB_PATH).mkdir(parents=True, exist_ok=True)
    _get_deepface()

    if DEEPFACE_STREAM and _get_deepface():
        threading.Thread(target=_deepface_stream_loop, daemon=True, name="deepface-stream").start()
        log.info("DeepFace stream loop started (frame_thr=%d, time_thr=%ds)", STREAM_FRAME_THRESHOLD, STREAM_TIME_THRESHOLD)
    else:
        threading.Thread(target=_simple_capture_loop, daemon=True, name="capture").start()
        log.info("Simple capture loop started at %d FPS", CAPTURE_FPS)

    if PUSH_ENABLED and not DEEPFACE_STREAM:
        threading.Thread(
            target=_change_detection_loop, daemon=True, name="change-detect"
        ).start()
        log.info("Change detection enabled (webhook=%s)", N8N_WEBHOOK_URL)
    else:
        if DEEPFACE_STREAM:
            log.info("Change detection disabled (DeepFace stream mode)")
        else:
            log.info("Change-detection push is disabled")

    log.info("State: %s, file: %s, stream_faces: %s, embeddings: %s, stream_mode: %s",
             STATE_BACKEND, STATE_FILE, STREAM_FACE_FILE, EMBEDDINGS_PATH, DEEPFACE_STREAM)


@app.on_event("shutdown")
def shutdown_event():
    global _cap
    _shutdown_event.set()
    with _cap_lock:
        if _cap is not None:
            _cap.release()
            _cap = None


# ── Vision state ──────────────────────────────────────────────────────
@app.post("/state")
def update_state(payload: dict = Body(...)):
    """Write the latest vision state."""
    _write_state_file(payload)
    return {"ok": True}


@app.get("/state")
def get_state():
    """Read the latest vision state (n8n/VLM combined: faces + scene_caption)."""
    return _read_state_file()


@app.get("/stream-faces")
def get_stream_faces():
    """Read face detection from DeepFace stream loop (separate from /state)."""
    return _read_stream_face_file()


# ── Face identification (embedding-based: user_1, user_2, ...) ─────────
@app.post("/identify")
async def identify(data: Optional[UploadFile] = File(None), add_if_unknown: bool = True):
    """Identify face via embeddings. Multipart 'data' or latest frame. Returns user_N or Unknown."""
    img_bytes = None
    if data and data.filename:
        img_bytes = await data.read()
    else:
        with _buf.lock:
            frame = _buf.frame
        if frame is not None:
            img_bytes = _encode_jpeg(frame)
    if not img_bytes:
        return {"name": "Unknown", "confidence": 0, "reason": "no image provided"}

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"name": "Unknown", "confidence": 0, "reason": "invalid image"}

    name, confidence = _identify_by_embedding(
        img, add_if_unknown=add_if_unknown, detector_backend="opencv"
    )
    return {"name": name, "confidence": round(confidence, 2)}


# ── Camera & stream endpoints ────────────────────────────────────────
@app.get("/health")
def health():
    with _buf.lock:
        has_frame = _buf.frame is not None
        ts = _buf.timestamp
    return {
        "ok": True,
        "streaming": has_frame,
        "last_frame_ts": ts,
        "push_enabled": PUSH_ENABLED,
        "state_backend": STATE_BACKEND,
        "state_file": STATE_FILE,
        "stream_face_file": STREAM_FACE_FILE,
        "deepface_stream": DEEPFACE_STREAM,
    }


@app.post("/reopen")
def reopen():
    """Re-initialise the camera."""
    global _cap
    with _cap_lock:
        if _cap is not None:
            _cap.release()
        _cap = _open_camera()
    return {"reopened": True}


@app.get("/latest")
def latest(quality: int = Query(JPEG_QUALITY, ge=30, le=100)):
    """Return the most recent buffered frame (with DeepFace overlays when stream mode)."""
    with _buf.lock:
        frame = _buf.frame
    if frame is None:
        return JSONResponse({"error": "no frame available yet"}, status_code=503)
    jpg = _encode_jpeg(frame, quality)
    if jpg is None:
        return JSONResponse({"error": "jpeg encoding failed"}, status_code=500)
    return Response(content=jpg, media_type="image/jpeg")


@app.get("/snapshot")
def snapshot(quality: int = Query(JPEG_QUALITY, ge=30, le=100)):
    """Backwards-compatible: returns latest frame."""
    return latest(quality)


@app.get("/stream")
def stream(
    fps: int = Query(10, ge=1, le=30),
    quality: int = Query(80, ge=30, le=100),
):
    """MJPEG stream with DeepFace overlays. Open in browser: http://localhost:8089/stream"""
    return StreamingResponse(
        _mjpeg_generator(fps, quality),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _mjpeg_generator(fps: int, quality: int):
    interval = 1.0 / fps
    while not _shutdown_event.is_set():
        with _buf.lock:
            frame = _buf.frame
        if frame is not None:
            jpg = _encode_jpeg(frame, quality)
            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
        time.sleep(interval)


@app.get("/snapshot_to_file")
def snapshot_to_file(
    path: str = Query("current_frame.jpg"),
    quality: int = Query(JPEG_QUALITY, ge=30, le=100),
):
    """Save the latest buffered frame to a file."""
    with _buf.lock:
        frame = _buf.frame
    if frame is None:
        return JSONResponse({"error": "no frame available yet"}, status_code=503)
    ok = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return JSONResponse({"error": f"failed to write: {path}"}, status_code=500)
    return {"saved": True, "path": path}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8089)
