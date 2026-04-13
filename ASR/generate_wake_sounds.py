#!/usr/bin/env python3
"""
Generate wake word acknowledgment audio files using Kokoro TTS.
Saves WAV files to ASR/wake_sounds/. Re-run to regenerate.

Usage:
    python ASR/generate_wake_sounds.py
    python ASR/generate_wake_sounds.py --tts-url http://localhost:8880 --voice af_sky
"""

import argparse
import os
import sys
from pathlib import Path

import requests

PHRASES = [
    ("mm", "Mm?"),
    ("yes", "Yes?"),
    ("im_here", "I'm here."),
    ("listening", "Listening."),
    ("yeah", "Yeah?"),
    ("mmhmm", "Mm-hmm?"),
]

DEFAULT_TTS_URL = os.getenv("TTS_URL", "http://localhost:8880")
DEFAULT_VOICE = os.getenv("TTS_VOICE", "af_sky")
OUT_DIR = Path(__file__).parent / "wake_sounds"


def generate(tts_url: str, voice: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    for name, text in PHRASES:
        out_path = OUT_DIR / f"{name}.wav"
        print(f"  Generating '{text}' -> {out_path.name} ... ", end="", flush=True)
        try:
            resp = requests.post(
                f"{tts_url}/v1/audio/speech",
                json={"model": "kokoro", "input": text, "voice": voice},
                timeout=30,
            )
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            print("ok")
        except Exception as e:
            print(f"FAILED: {e}")

    generated = list(OUT_DIR.glob("*.wav"))
    print(f"\n{len(generated)} file(s) in {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts-url", default=DEFAULT_TTS_URL)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    args = parser.parse_args()

    print(f"Kokoro TTS: {args.tts_url}  voice: {args.voice}")
    generate(args.tts_url, args.voice)
