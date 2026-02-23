import threading
import time
from typing import Optional

import cv2
from fastapi import FastAPI, Response, Query
from fastapi.responses import JSONResponse

app = FastAPI()

_lock = threading.Lock()
_cap: Optional[cv2.VideoCapture] = None

# Tune these if you want:
# changet the index cam if you see a black screen
CAM_INDEX = 0
WIDTH = 1280
HEIGHT = 720
JPEG_QUALITY = 90  # 80-95 typical tradeoff


def open_camera() -> cv2.VideoCapture:
    # CAP_DSHOW tends to be most reliable on Windows
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # Warm up: many webcams need a few frames before exposure settles
    for _ in range(5):
        cap.read()
        time.sleep(0.02)
    return cap


@app.on_event("startup")
def startup_event():
    global _cap
    _cap = open_camera()


@app.on_event("shutdown")
def shutdown_event():
    global _cap
    with _lock:
        if _cap is not None:
            _cap.release()
            _cap = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/reopen")
def reopen():
    """If the camera glitches, call this to re-init without restarting the process."""
    global _cap
    with _lock:
        if _cap is not None:
            _cap.release()
        _cap = open_camera()
    return {"reopened": True}


@app.get("/snapshot")
def snapshot(quality: int = Query(JPEG_QUALITY, ge=30, le=100)):
    global _cap
    if _cap is None:
        return JSONResponse({"error": "camera not initialized"}, status_code=500)

    with _lock:
        # Flush buffer
        for _ in range(3):
            _cap.grab()
        ret, frame = _cap.retrieve()

    if not ret or frame is None:
        return JSONResponse({"error": "failed to capture frame"}, status_code=500)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        return JSONResponse({"error": "jpeg encoding failed"}, status_code=500)

    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/snapshot_to_file")
def snapshot_to_file(
    path: str = Query("current_frame.jpg"),
    quality: int = Query(JPEG_QUALITY, ge=30, le=100),
):
    """Save to a file on the API host. Useful if another process expects a file path."""
    global _cap
    if _cap is None:
        return JSONResponse({"error": "camera not initialized"}, status_code=500)

    with _lock:
        ret, frame = _cap.read()

    if not ret or frame is None:
        return JSONResponse({"error": "failed to capture frame"}, status_code=500)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok = cv2.imwrite(path, frame, encode_params)
    if not ok:
        return JSONResponse({"error": f"failed to write: {path}"}, status_code=500)

    return {"saved": True, "path": path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8089)