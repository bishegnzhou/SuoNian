import requests
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).resolve().parent
load_dotenv(project_root / ".env")

API_URL = "http://localhost:8000/api/experiments/orchestrator/stream"

def test_orchestrator_endpoint():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not found in .env")
        return

    payload = {
        "task": "Test hypothesis for API debugging",
        "model": "deepseek-chat",
        "test_mode": False,
        "num_agents": 1,
        "max_rounds": 1,
        "max_parallel": 1,
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
                    decoded = line.decode('utf-8')
                    try:
                        data = json.loads(decoded)
                        if data.get("type") == "line" and "plain" in data:
                            print(f"Log: {data['plain'].strip()}")
                        else:
                            print(f"Event: {decoded[:100]}...")
                    except json.JSONDecodeError:
                        print(f"Raw: {decoded[:100]}...")
                    
                    # Stop after we see the debug message or start
                    if "run_orchestrator_loop received model" in decoded:
                        break
            print("Stream test passed.")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_orchestrator_endpoint()
