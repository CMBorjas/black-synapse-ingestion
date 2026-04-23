from fastapi import FastAPI, Request, HTTPException
import uvicorn
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import io
import wave

app = FastAPI(title="Emotion Recognition Service")

# Load model and extractor
MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading emotion model: {MODEL_ID} on {device}...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID).to(device)

# Label mapping (adjust based on the specific model's config)
# This model typically outputs: angry, calm, disgust, fear, happy, neutral, sad, surprised
config = model.config
id2label = config.id2label

@app.post("/analyze")
async def analyze_emotion(request: Request):
    audio_data = await request.body()
    if not audio_data:
        raise HTTPException(status_code=400, detail="No audio data received")

    try:
        # Load audio using librosa from bytes
        # We assume the input is WAV format
        audio_file = io.BytesIO(audio_data)
        y, sr = librosa.load(audio_file, sr=16000)

        # Preprocess
        inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            logits = model(**inputs).logits

        # Get prediction
        predicted_ids = torch.argmax(logits, dim=-1).item()
        emotion = id2label[predicted_ids]
        
        # Get scores for all emotions
        scores = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        emotion_scores = {id2label[i]: float(scores[i]) for i in range(len(scores))}

        return {
            "emotion": emotion,
            "scores": emotion_scores
        }

    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
