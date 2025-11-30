import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.getcwd())

# Load environment variables
load_dotenv()

from agent import run_experiment_loop

print("Starting debug run...")
try:
    run_experiment_loop("Test hypothesis", model="deepseek-chat")
    print("Debug run finished successfully.")
except Exception as e:
    print(f"Debug run failed with error: {e}")
    import traceback
    traceback.print_exc()
