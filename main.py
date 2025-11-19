import sys
import argparse
from dotenv import load_dotenv
from agent import run_experiment_loop
from logger import print_status

def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="AI Experiment Agent CLI")
    parser.add_argument("hypothesis", type=str, help="The hypothesis to verify.")
    parser.add_argument("--gpu", type=str, default=None, help="GPU type to request for the sandbox (e.g., 'T4', 'A10G', 'A100', 'any').")
    
    args = parser.parse_args()

    print_status("Initializing Agent...", "bold cyan")
    
    try:
        # Record GPU preference globally for sandbox creation
        import agent as agent_module
        agent_module._selected_gpu = args.gpu
        run_experiment_loop(args.hypothesis)
    except KeyboardInterrupt:
        print_status("\nExperiment interrupted by user.", "bold red")
        sys.exit(0)
    except Exception as e:
        print_status(f"\nFatal Error: {e}", "bold red")
        sys.exit(1)

if __name__ == "__main__":
    main()
