#!/usr/bin/env python3
"""
Minimal Kokoro TTS test using sherpa-onnx + ONNX Runtime.
CPU by default; for GPU set USE_GPU = True and install a CUDA-capable ONNX Runtime.

Model layout (e.g. kokoro-en-v0_19): model.onnx, voices.bin, tokens.txt, espeak-ng-data/
Multilingual (kokoro-multi-lang-v1_0): also lexicon-us-en.txt + lexicon-zh.txt (set automatically).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import onnxruntime as ort
import sherpa_onnx
import soundfile as sf

MODEL_DIR = os.environ.get("KOKORO_DIR", "./kokoro-en-v0_19")

# ---------------------------------------------------------------------------
# CPU vs GPU: OfflineTtsModelConfig accepts provider "cpu" | "cuda" | "coreml".
# For Jetson, install ONNX Runtime built with CUDA for your JetPack, then confirm
# CUDAExecutionProvider appears in ort.get_available_providers() before USE_GPU=True.
# ---------------------------------------------------------------------------
USE_GPU = False


def print_ort_providers() -> None:
    providers = ort.get_available_providers()
    print("ONNX Runtime available providers:", providers)
    print(
        "CUDAExecutionProvider available:",
        "CUDAExecutionProvider" in providers,
    )


def kokoro_lexicon_for_dir(model_dir: str) -> str:
    us = os.path.join(model_dir, "lexicon-us-en.txt")
    zh = os.path.join(model_dir, "lexicon-zh.txt")
    if os.path.isfile(us) and os.path.isfile(zh):
        return f"{us},{zh}"
    return ""


def build_tts(model_dir: str) -> sherpa_onnx.OfflineTts:
    model = os.path.join(model_dir, "model.onnx")
    voices = os.path.join(model_dir, "voices.bin")
    tokens = os.path.join(model_dir, "tokens.txt")
    data_dir = os.path.join(model_dir, "espeak-ng-data")

    provider = "cuda" if USE_GPU else "cpu"

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=model,
                voices=voices,
                tokens=tokens,
                data_dir=data_dir,
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
        sys.exit("Invalid OfflineTtsConfig — check paths under: " + model_dir)
    return sherpa_onnx.OfflineTts(tts_config)


def main() -> None:
    print_ort_providers()

    model_dir = sys.argv[1] if len(sys.argv) > 1 else MODEL_DIR
    tts = build_tts(model_dir)

    text = (
        "Hello from Kokoro on sherpa-onnx. "
        "This is a short CPU test before optional GPU."
    )
    gen = sherpa_onnx.GenerationConfig()
    gen.sid = int(os.environ.get("TTS_SID", "0"))
    gen.speed = float(os.environ.get("TTS_SPEED", "1.0"))
    gen.silence_scale = 0.2

    audio = tts.generate(text, gen)
    if len(audio.samples) == 0:
        sys.exit("TTS returned no samples — see stderr above.")

    out_wav = os.environ.get("TTS_OUT_WAV", "tts_test_out.wav")
    samples = np.asarray(audio.samples, dtype=np.float32)
    sf.write(out_wav, samples, audio.sample_rate, subtype="PCM_16")

    duration_s = float(samples.shape[0]) / float(audio.sample_rate)
    print(f"sample_rate_hz: {audio.sample_rate}")
    print(f"duration_s: {duration_s:.4f}")
    print(f"wrote: {out_wav}")


if __name__ == "__main__":
    main()
