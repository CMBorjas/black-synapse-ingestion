#!/usr/bin/env python3
"""
Kokoro TTS HTTP API (sherpa-onnx). Loads the model once at startup.

GET /tts?text=...  -> audio/wav

GPU later: set TTS_USE_GPU=1 and provider="cuda" in build_tts (requires CUDA ORT).
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import sherpa_onnx
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

MODEL_DIR = os.environ.get("KOKORO_DIR", "./kokoro-en-v0_19")
USE_GPU = os.environ.get("TTS_USE_GPU", "").lower() in ("1", "true", "yes")

_tts: Optional[sherpa_onnx.OfflineTts] = None


def kokoro_lexicon_for_dir(model_dir: str) -> str:
    us = os.path.join(model_dir, "lexicon-us-en.txt")
    zh = os.path.join(model_dir, "lexicon-zh.txt")
    if os.path.isfile(us) and os.path.isfile(zh):
        return f"{us},{zh}"
    return ""


def build_tts(model_dir: str) -> sherpa_onnx.OfflineTts:
    provider = "cuda" if USE_GPU else "cpu"
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=os.path.join(model_dir, "model.onnx"),
                voices=os.path.join(model_dir, "voices.bin"),
                tokens=os.path.join(model_dir, "tokens.txt"),
                data_dir=os.path.join(model_dir, "espeak-ng-data"),
                lexicon=kokoro_lexicon_for_dir(model_dir),
            ),
            provider=provider,
            debug=False,
            num_threads=int(os.environ.get("TTS_NUM_THREADS", "4")),
        ),
        rule_fsts=os.environ.get("TTS_RULE_FSTS", ""),
        max_num_sentences=int(os.environ.get("TTS_MAX_NUM_SENTENCES", "1")),
    )
    if not tts_config.validate():
        raise RuntimeError("Invalid OfflineTtsConfig — check KOKORO_DIR and files")
    return sherpa_onnx.OfflineTts(tts_config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tts
    _tts = build_tts(MODEL_DIR)
    yield
    _tts = None


app = FastAPI(title="Kokoro TTS (sherpa-onnx)", lifespan=lifespan)


@app.get("/tts")
def tts(text: str = Query(..., min_length=1, max_length=5000)):
    if _tts is None:
        raise HTTPException(status_code=503, detail="TTS not initialized")

    gen = sherpa_onnx.GenerationConfig()
    gen.sid = int(os.environ.get("TTS_SID", "0"))
    gen.speed = float(os.environ.get("TTS_SPEED", "1.0"))
    gen.silence_scale = 0.2

    audio = _tts.generate(text, gen)
    if len(audio.samples) == 0:
        raise HTTPException(status_code=500, detail="Empty audio")

    buf = io.BytesIO()
    samples = np.asarray(audio.samples, dtype=np.float32)
    sf.write(
        buf,
        samples,
        audio.sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
