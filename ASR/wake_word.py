### voice_trigger.py
# Handles wake word detection + VAD-based recording
# Continuously listens for wake word and records audio after detection

import webrtcvad
import sounddevice as sd
import numpy as np
import wave
import time
from pathlib import Path

from openwakeword import Model

SAMPLE_RATE = 16000
# OpenWakeWord processes audio in chunks (typically 1280 samples = 80ms at 16kHz)
# Using 1280 samples per frame for optimal performance
OWW_FRAME_LENGTH = 1280
# VAD frame size (30ms for optimal VAD performance)
VAD_FRAME_DURATION_MS = 30
VAD_FRAME_SIZE = int(SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)
SILENCE_FRAMES = int(800 / VAD_FRAME_DURATION_MS)
OUTPUT_WAV = "utterance.wav"
# Detection threshold for wake word (adjust between 0.0 and 1.0)
WAKE_WORD_THRESHOLD = 0.5
# Cooldown period after detection (seconds) - prevents immediate re-detection
COOLDOWN_SECONDS = 2.0

# Local OpenWakeWord ONNX; prediction dict key is the stem (e.g. "Atlas").
ATLAS_ONNX = Path(__file__).resolve().parent / "models" / "Atlas.onnx"


def is_speech(frame, vad):
    return vad.is_speech(frame.tobytes(), SAMPLE_RATE)


def _load_wake_model() -> Model:
    if not ATLAS_ONNX.is_file():
        raise FileNotFoundError(f"Wake word ONNX not found: {ATLAS_ONNX}")
    return Model(
        wakeword_models=[str(ATLAS_ONNX)],
        inference_framework="onnx",
    )


def record_after_wake():
    oww_model = None
    stream = None
    wake_word_name = ATLAS_ONNX.stem

    try:
        oww_model = _load_wake_model()
    except Exception as e:
        raise RuntimeError(f"OpenWakeWord initialization error: {e}") from e
    
    vad = webrtcvad.Vad(2)
    # Use OpenWakeWord's recommended frame length for the stream
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=OWW_FRAME_LENGTH)
    
    try:
        stream.start()
        print(f"[Listening for wake word '{wake_word_name}' ({ATLAS_ONNX.name})]")
        
        last_detection_time = 0.0  # Track when last wake word was detected
        cooldown_frames_to_flush = int(COOLDOWN_SECONDS * SAMPLE_RATE / OWW_FRAME_LENGTH)  # Frames to flush during cooldown

        while True:
            pcm = stream.read(OWW_FRAME_LENGTH)[0].flatten()

            # Process audio through OpenWakeWord
            try:
                prediction = oww_model.predict(pcm)
            except MemoryError:
                print("[MemoryError in wake word model — reinitializing...]")
                oww_model = _load_wake_model()
                prediction = {}
                continue
            
            # Check if we're in cooldown period
            current_time = time.time()
            in_cooldown = (current_time - last_detection_time) < COOLDOWN_SECONDS
            
            # Check if wake word detected (prediction is a dict with model names as keys)
            if (not in_cooldown and 
                wake_word_name in prediction and 
                prediction[wake_word_name] > WAKE_WORD_THRESHOLD):
                
                last_detection_time = current_time
                print("[Wake word detected!] Recording...")
                frames = []
                silence_counter = 0
                # Buffer for VAD processing (need 30ms frames for VAD)
                vad_buffer = np.array([], dtype=np.int16)

                while True:
                    frame = stream.read(OWW_FRAME_LENGTH)[0].flatten()
                    frames.append(frame)
                    vad_buffer = np.concatenate([vad_buffer, frame])
                    
                    # Process VAD when we have enough samples (30ms = 480 samples)
                    while len(vad_buffer) >= VAD_FRAME_SIZE:
                        vad_frame = vad_buffer[:VAD_FRAME_SIZE]
                        vad_buffer = vad_buffer[VAD_FRAME_SIZE:]
                        
                        if is_speech(vad_frame, vad):
                            silence_counter = 0
                        else:
                            silence_counter += 1
                            if silence_counter > SILENCE_FRAMES:
                                print("[Silence detected. Stopping recording.]")
                                break
                    
                    if silence_counter > SILENCE_FRAMES:
                        break

                audio = np.concatenate(frames).astype(np.int16)

                with wave.open(OUTPUT_WAV, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio.tobytes())

                print(f"[Audio saved to {OUTPUT_WAV}]")
                
                # Flush model state by processing silence frames during cooldown
                print(f"[Cooldown period: {COOLDOWN_SECONDS}s - flushing model state...]")
                silence_frame = np.zeros(OWW_FRAME_LENGTH, dtype=np.int16)
                for _ in range(cooldown_frames_to_flush):
                    oww_model.predict(silence_frame)
                
                # Reinitialize model to clear accumulated internal buffers (prevents MemoryError on long runs)
                oww_model = _load_wake_model()
                print("[Resuming wake word detection...]\n")
                # Continue listening for next wake word (don't break)
    finally:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        if oww_model is not None:
            try:
                # OpenWakeWord models don't require explicit cleanup, but we can reset if needed
                pass
            except Exception:
                pass

if __name__ == "__main__":
    record_after_wake()