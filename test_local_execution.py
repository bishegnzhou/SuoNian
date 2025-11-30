import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    try:
        resp = requests.get(f"{BASE_URL}/health")
        print(f"Health Check: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Health Check Failed: {e}")

def test_start_experiment():
    url = f"{BASE_URL}/api/experiments/orchestrator/stream"
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    payload = {
        "model": "deepseek-chat",
        "task": "Test local execution",
        "deepseekApiKey": os.getenv("DEEPSEEK_API_KEY", "your_api_key_here")
    }
    
    print(f"Testing {url} with payload: {payload}")
    
    try:
        with requests.post(url, json=payload, stream=True) as resp:
            print(f"Response Status: {resp.status_code}")
            if resp.status_code != 200:
                print(f"Error Content: {resp.text}")
            else:
                for line in resp.iter_lines():
                    if line:
                        print(f"Stream: {line.decode('utf-8')}")
    except Exception as e:
        print(f"Request Failed: {e}")

if __name__ == "__main__":
    test_health()
    test_start_experiment()
