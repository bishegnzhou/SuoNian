import requests
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).resolve().parent
load_dotenv(project_root / ".env")

API_URL = "http://localhost:8000/api/experiments/single/stream"

def test_stream_endpoint():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not found in .env")
        return

    payload = {
        "task": "Test hypothesis for API debugging",
        "model": "deepseek-chat",
        "test_mode": False,
        "credentials": {
            "deepseek_api_key": api_key
        }
    }
    
    print(f"Sending request to {API_URL}...")
    try:
        with requests.post(API_URL, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"API Error: {response.status_code} {response.reason}")
                print(response.text)
                return

            print("Response received. Reading stream...")
            for line in response.iter_lines():
                if line:
                    print(f"Received: {line.decode('utf-8')[:200]}...")
                    # Keep reading to see if we get an error event
            print("Stream finished.")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_stream_endpoint()
