import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

def test_deepseek_connectivity():
    # Load .env if present
    load_dotenv()
    
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not found in environment variables.")
        print("Please set DEEPSEEK_API_KEY in your .env file or environment.")
        return False

    print(f"DEEPSEEK_API_KEY found: {api_key[:4]}...{api_key[-4:]}")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    try:
        print("Attempting to connect to DeepSeek API...")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, this is a test. Reply with 'DeepSeek is working'."},
            ],
            stream=False
        )
        content = response.choices[0].message.content
        print(f"Response from DeepSeek: {content}")
        print("SUCCESS: DeepSeek API connectivity verified.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to connect to DeepSeek API. Details: {e}")
        return False

if __name__ == "__main__":
    success = test_deepseek_connectivity()
    sys.exit(0 if success else 1)
