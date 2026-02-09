# Blacksynapse

AI data ingestion system for the Desky robot. Ingests data from multiple sources (Notion, Gmail, Drive, Slack) and stores it in a vector database for RAG capabilities.

## How to start

1. Start the docker container:
- This basically starts:
    - ASR API with model
    - TTS API with model
    - n8n API
    - qdrant API 

- Command on bash:
   ```bash
   docker-compose up -d
   ```

2. Start Speaker and Microphone APIs:
   - Issues with starting this with docker so you'll have to start it manually:

   - Starting the microphone w/wake word
    ```bash
    cd ASR
    pip install -r requirements.txt
    python wake_word.py
    ```   

   - Starting the speaker
    ```bash
    cd TTS
    pip install -r requirements.txt
    python speaker_api.py
    ```   

3. Access to local APIs:
   - Qdrant: http://localhost:6333
   - n8n: http://localhost:5678

4. N8N Configuration:
   - Import the n8n workflow from n8n/workflows in the repo
   - If you can't connect to the APIs, change the "localhost" to "host.docker.internal" or your ip address
   
   **Workflow Overview:**
   ```
   ┌─────────────────────┐
   │ Local File Trigger  │ (Path: /data/asr)
   │ (Audio recordings)  │
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │ Read/Write Files    │
   │    from Disk        │
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │   ASR: Whisper      │
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐     ┌──────────────────┐
   │                     │     │ When chat        │
   │  LLM - Ollama 3     │◀────│ message received │
   │                     │     └──────────────────┘
   │  • Chat Model       │
   │  • Memory           │
   │  • Tool: cu_denver  │───┐
   │    _search          │   │
   └──────────┬──────────┘   │
              │              │
              ▼              │
   ┌─────────────────────┐   │
   │   TTS: Kokoro       │   │
   └──────────┬──────────┘   │
              │              │
              ▼              │
   ┌─────────────────────┐   │
   │      Speaker        │   │
   └─────────────────────┘   │
                             │
                    ┌────────┘
                    │
                    ▼
         ┌──────────────────┐
         │ Embeddings Ollama│
         └──────────────────┘
   ```
   
   **Flow Description:**
   - **Audio Input Path**: Local file trigger → Read files → ASR (Whisper) → LLM
   - **Chat Input Path**: Chat message trigger → LLM directly
   - **LLM Processing**: Uses Ollama chat model with memory and search tool (cu_denver_search with embeddings)
   - **Output Path**: LLM → TTS (Kokoro) → Speaker

5. Wake word:
   - You can modify the wake word by using pre-trained models or training your own models using: https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb

5. Ollama:
   - Currenly, the model for Ollama is installed automatically when you start the docker container. You can find the file in ollama/init.models.sh
   - If you want to modify the model that starts with docker change it in the file here:
   ```
   OLLAMA_HOST=$OLLAMA_HOST ollama pull qwen2.5:7b
   ```
   - You can also install another ollama model locally directly by running this on your terminal or the docker container terminal:
    ```
   ollama run gemma3
    ```
   Note: Make sure to monitor storage as installing a lot of models will fill up your empty storage pretty fast and silently.

6. Qdrant:
   - To add a database in Qdrant, you can upload an existing snapshot.

## Requirements

- Docker and Docker Compose
- Python