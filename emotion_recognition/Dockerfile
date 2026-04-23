FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model to avoid slow startup
RUN python -c "from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor; \
    MODEL_ID = 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition'; \
    Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID); \
    Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)"

COPY . .

CMD ["python", "emotion_server.py"]
