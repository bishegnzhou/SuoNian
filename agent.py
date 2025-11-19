import os
import subprocess
import sys
from typing import Optional
from google import genai
from google.genai import types
from logger import print_panel, print_status, log_step, logger

import modal

# Cache a single sandbox per run so the agent can keep state across tool calls.
_shared_sandbox: Optional[modal.Sandbox] = None
_shared_gpu: Optional[str] = None  # Track which GPU the sandbox was created with
_selected_gpu: Optional[str] = None  # User-selected GPU for this run

def _get_shared_sandbox(gpu: Optional[str]) -> modal.Sandbox:
    """Create (once) and return a persistent sandbox for this run."""
    global _shared_sandbox, _shared_gpu
    if _shared_sandbox is not None:
        # Reuse only if GPU selection matches
        if gpu == _shared_gpu:
            return _shared_sandbox
        _close_shared_sandbox()

    log_step("EXECUTION", "Initializing shared Sandbox...")

    # Define a robust image with common dependencies (built once).
    image = (
        modal.Image.debian_slim()
        .pip_install("numpy", "pandas", "torch", "scikit-learn", "matplotlib")
    )

    # Create a Modal App to associate with the Sandbox
    log_step("EXECUTION", "Looking up Modal App 'agent-sandbox-app'...")
    app = modal.App.lookup("agent-sandbox-app", create_if_missing=True)
    log_step("EXECUTION", "Modal App found/created.")

    # Keep the sandbox alive by running an inert loop; subcommands run via sandbox.exec.
    gpu_msg = f"gpu={gpu}" if gpu else "cpu-only"
    log_step("EXECUTION", f"Creating persistent Sandbox (keep-alive loop, {gpu_msg})...")
    _shared_sandbox = modal.Sandbox.create(
        "bash",
        "-lc",
        "while true; do sleep 3600; done",
        app=app,
        image=image,
        timeout=7200,
        gpu=gpu,
    )
    _shared_gpu = gpu
    log_step("EXECUTION", "Persistent Sandbox ready.")
    return _shared_sandbox


def _close_shared_sandbox():
    """Terminate the shared sandbox if it exists."""
    global _shared_sandbox
    if _shared_sandbox is not None:
        try:
            _shared_sandbox.terminate()
            log_step("EXECUTION", "Persistent Sandbox terminated.")
        except Exception as e:
            log_step("WARNING", f"Failed to terminate sandbox cleanly: {e}")
        _shared_sandbox = None


def execute_in_sandbox(code: str):
    """
    Executes Python code inside a persistent Modal Sandbox using sandbox.exec.
    """
    try:
        sandbox = _get_shared_sandbox(_selected_gpu)

        log_step("EXECUTION", "Launching python exec inside Sandbox...")
        print_panel(code, "Sandbox Code", "code")

        # Run the snippet via exec; stdout/stderr are streamed text.
        proc = sandbox.exec("python", "-u", "-")

        proc.stdin.write(code.encode("utf-8"))
        proc.stdin.write_eof()
        proc.stdin.drain()  # Flush buffered stdin

        stdout = []
        stderr = []

        log_step("EXECUTION", "Reading stdout/stderr streams...")
        for part in proc.stdout:
            stdout.append(part)
            print(part, end="")

        for part in proc.stderr:
            stderr.append(part)
            print(part, end="", file=sys.stderr)

        log_step("EXECUTION", "Waiting for process exit...")
        exit_code = proc.wait()
        log_step("EXECUTION", f"Process exited with code {exit_code}")

        stdout_str = "".join(stdout)
        stderr_str = "".join(stderr)

        return f"Exit Code: {exit_code}\nSTDOUT:\n{stdout_str}\nSTDERR:\n{stderr_str}"

    except Exception as e:
        log_step("ERROR", f"Sandbox Execution Failed: {str(e)}")
        return f"Sandbox Execution Failed: {str(e)}"

def run_experiment_loop(hypothesis):
    """The main loop using native tool calling."""
    print_panel(f"Hypothesis: {hypothesis}", "Starting Experiment", "bold green")
    log_step("START", f"Hypothesis: {hypothesis}")
    print_status(f"Sandbox GPU request: {_selected_gpu or 'CPU'}", "info")

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    
    # Define the tool
    tools = [execute_in_sandbox]

    gpu_hint = _selected_gpu or "CPU"

    system_prompt = f"""You are an autonomous research scientist.
Your goal is to verify the user's hypothesis.

**Your Tool**:
- `execute_in_sandbox(code)`: Runs a Python script in a secure Modal Sandbox.
- **Dependencies**: The environment has `numpy`, `pandas`, `torch`, `sklearn`, `matplotlib`.
- **Compute**: Sandbox GPU request for this run: {gpu_hint}.
- **Usage**: Just write the Python code. No need for `@app.local_entrypoint` or `modal` imports unless you are doing something advanced. The code runs as a standard script.

**The Loop**:
1. **Think**: Plan your next step.
2. **Act**: Call `execute_in_sandbox` with your script.
3. **Observe**: I will give you the results.

Output [DONE] when you have verified the hypothesis.
"""

    history = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=system_prompt),
                types.Part.from_text(text=f"Hypothesis: {hypothesis}")
            ]
        )
    ]

    step_count = 0
    max_steps = 10

    while step_count < max_steps:
        step_count += 1
        print_status(f"Step {step_count}...", "dim")

        try:
            # Generate content with tools
            response = client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=history,
                config=types.GenerateContentConfig(
                    tools=tools,
                    # thinking_level="HIGH", # Enable deep thinking - Commented out due to validation error
                )
            )
        except Exception as e:
            print_status(f"API Error: {e}", "error")
            logger.error(f"API Error: {e}")
            break

        # Handle the response parts
        # Gemini 3 Pro might output thoughts, text, and function calls.
        
        # 1. Log Thoughts (if available in text)
        if response.text:
            print_panel(response.text, "Agent Thought", "thought")
            log_step("THOUGHT", response.text)
            # Add model turn to history
            history.append(types.Content(role="model", parts=[types.Part.from_text(text=response.text)]))

            if "[DONE]" in response.text:
                print_status("Agent signaled completion.", "success")
                break

        # 2. Handle Function Calls
        # We need to check all candidates/parts for function calls
        function_called = False
        for part in response.candidates[0].content.parts:
            if part.function_call:
                function_called = True
                fn_name = part.function_call.name
                fn_args = part.function_call.args
                
                print_panel(f"Calling {fn_name}...", "Tool Call", "code")
                
                if fn_name == "execute_in_sandbox":
                    # Execute
                    result = execute_in_sandbox(**fn_args)
                    
                    # Truncate result
                    if len(result) > 20000:
                        result = result[:10000] + "\n...[TRUNCATED]...\n" + result[-10000:]

                    print_panel(result, "Tool Result", "result")
                    log_step("TOOL_RESULT", "Executed")

                    # Add function call and response to history
                    # Note: We already added the text part above. Now we add the function call part if it wasn't in the text?
                    # Actually, for Gemini, we should construct the Model turn with ALL parts (text + function_call)
                    # and then the User turn with the function_response.
                    
                    # Let's reconstruct the correct history flow:
                    # The 'response' object already has the model's turn content.
                    # We just need to append the *response* content to history properly.
                    
                    # Remove the text-only append we did above to avoid duplication if we do it here
                    if history[-1].role == "model":
                        history.pop() 
                    
                    history.append(response.candidates[0].content)
                    
                    # Create the function response part
                    history.append(types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=fn_name,
                                response={"result": result}
                            )
                        ]
                    ))
        
        if not function_called and not response.text:
             print_status("Empty response from model.", "warning")

    try:
        # Final Report
        print_status("Generating Final Report...", "bold green")
        history.append(types.Content(role="user", parts=[types.Part.from_text(text="Generate a concise InfoDense report.")]))
        
        final_report = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=history,
            # config=types.GenerateContentConfig(thinking_level="HIGH")
        )
        print_panel(final_report.text, "Final Report", "bold green")
    finally:
        _close_shared_sandbox()
