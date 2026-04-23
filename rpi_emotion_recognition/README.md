# Raspberry Pi Emotion Recognition Package

This package contains the necessary files to run the emotion recognition and face display system on a Raspberry Pi.

## Contents

- `rpi_face/`: Python script to display an interactive face using Pygame.
- `emotion_recognition/`: FastAPI server for audio-based emotion recognition using Wav2Vec2.
- `action_servos/`: Hardware control for PCA9685 servo drivers.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Face Client:
   ```bash
   python rpi_face/face_client.py
   ```

3. (Optional) Run the Emotion Server:
   ```bash
   python emotion_recognition/emotion_server.py
   ```
