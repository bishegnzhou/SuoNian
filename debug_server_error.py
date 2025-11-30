import os
import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from agent import run_experiment_loop
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def reproduce_issue():
    print("Attempting to run experiment loop with deepseek-chat...")
    try:
        run_experiment_loop(
            hypothesis="Test hypothesis for debugging",
            test_mode=False,
            model="deepseek-chat"
        )
        print("Experiment loop completed successfully.")
    except Exception as e:
        print(f"Caught exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce_issue()
