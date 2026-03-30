import pytest
import requests
import json
import uuid
import time
from datetime import datetime

BASE_URL = "http://localhost:8001"

def test_chat_log_and_consolidate():
    # 1. Log a chat message
    session_id = f"test_session_{uuid.uuid4()}"
    payload = {
        "session_id": session_id,
        "role": "user",
        "message": "Hello, this is a test message for long term memory.",
        "timestamp": datetime.now().isoformat()
    }
    
    # Retry mechanism for startup
    for i in range(10):
        try:
            response = requests.post(f"{BASE_URL}/chat/log", json=payload)
            if response.status_code == 200:
                print("Connection successful.")
                break
        except requests.exceptions.ConnectionError:
            print(f"Waiting for server... attempt {i+1}")
            time.sleep(2)
            if i == 9:
                raise Exception("Server not reachable")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    print(f"Logged chat message: {data}")

    # 2. Log another message
    payload2 = {
        "session_id": session_id,
        "role": "assistant",
        "message": "I will remember this.",
        "timestamp": datetime.now().isoformat()
    }
    response = requests.post(f"{BASE_URL}/chat/log", json=payload2)
    assert response.status_code == 200
    
    # 3. Consolidate memory
    # Initialize pipeline first to ensure tables exist if not already
    # Although the worker app should have settled by now.
    
    response = requests.post(f"{BASE_URL}/chat/memory/consolidate")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    print(f"Consolidation result: {data}")
    
    assert data["sessions_processed"] >= 1
    assert data["processed_count"] >= 2

if __name__ == "__main__":
    try:
        test_chat_log_and_consolidate()
        print("Test PASSED")
    except Exception as e:
        print(f"Test FAILED: {e}")
