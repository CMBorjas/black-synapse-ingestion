FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio processing and build tools for compiling Python packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements
RUN pip install --no-cache-dir -r /tmp/requirements

# Copy the ASR server code (path relative to root context)
COPY ASR/asr_server.py .

# Create directory for received audio
RUN mkdir -p received_audio

# Expose the port
EXPOSE 8002

# Run the ASR server
CMD ["python", "asr_server.py"]