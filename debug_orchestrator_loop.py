import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.getcwd())

# Load environment variables
load_dotenv()

from orchestrator import run_orchestrator_loop

print("Starting orchestrator debug run...")
try:
    run_orchestrator_loop("Test hypothesis", model="deepseek-chat")
    print("Orchestrator debug run finished successfully.")
except Exception as e:
    print(f"Orchestrator debug run failed with error: {e}")
    import traceback
    traceback.print_exc()
