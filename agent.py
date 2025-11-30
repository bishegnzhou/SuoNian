import os
import sys
import subprocess
import threading
import json
from typing import Optional, List

from google import genai
from google.genai import types

import anthropic
import openai
import openai

from logger import print_panel, print_status, log_step, logger

try:
    import modal
    from modal.stream_type import StreamType
except (ImportError, RuntimeError) as e:
    print_status(f"Warning: Modal import failed ({e}). Using mock.", "bold yellow")
    from unittest.mock import MagicMock
    modal = MagicMock()
    StreamType = MagicMock()

# Cache a single sandbox per run so the agent can keep state across tool calls.
_shared_sandbox: Optional[modal.Sandbox] = None
_shared_gpu: Optional[str] = None  # Track which GPU the sandbox was created with
_selected_gpu: Optional[str] = None  # User-selected GPU for this run


def emit_event(event_type: str, data: dict) -> None:
    """Emit a structured event for the frontend."""
    # Only emit structured events when explicitly enabled (e.g. from the web API).
    # This keeps the CLI output clean while still allowing rich UIs to subscribe.
    if not os.environ.get("AI_RESEARCHER_ENABLE_EVENTS"):
        return

    import json
    payload = {
        "type": event_type,
        "timestamp": 0,
        "data": data,
    }
    print(f"::EVENT::{json.dumps(payload)}")
    sys.stdout.flush()


def _build_generation_config(
    *,
    tools: Optional[list] = None,
    system_instruction: Optional[str] = None,
    disable_autofc: bool = False,
) -> types.GenerateContentConfig:
    """
    Build a GenerateContentConfig that:

    - Enables Gemini "thinking mode" with visible thought summaries.
    - Sets thinking_level=HIGH (recommended for Gemini 3 Pro).
    - Optionally disables automatic function calling so we can control
      when tools run and show thoughts before actions.
    """
    thinking_config = types.ThinkingConfig(
        thinking_level=types.ThinkingLevel.HIGH,
        include_thoughts=True,
    )

    config_kwargs = {
        "tools": tools,
        "system_instruction": system_instruction,
        "thinking_config": thinking_config,
    }

    if disable_autofc:
        # Turn off automatic Python function calling so we get function_call
        # parts back and can execute tools manually in our loop.
        config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
            disable=True
        )

    return types.GenerateContentConfig(**config_kwargs)



class LocalSandbox:
    def exec(self, cmd, *args, stdout=None, stderr=None):
        import subprocess
        process = subprocess.Popen(
            [sys.executable, "-u", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        return LocalProcess(process)

class LocalProcess:
    def __init__(self, process):
        self.process = process
        self.stdin = LocalStdin(process.stdin)
        self.stdout = LocalStream(process.stdout)
        self.stderr = LocalStream(process.stderr)
    
    def wait(self):
        return self.process.wait()

class LocalStdin:
    def __init__(self, stdin):
        self.stdin = stdin
    
    def write(self, data):
        self.stdin.write(data)
    
    def write_eof(self):
        self.stdin.close()
    
    def drain(self):
        pass

class LocalStream:
    def __init__(self, stream):
        self.stream = stream
    
    def __iter__(self):
        for line in self.stream:
            yield line.decode("utf-8", errors="replace")

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

    # Check if we have Modal credentials
    if not os.environ.get("MODAL_TOKEN_ID") or not os.environ.get("MODAL_TOKEN_SECRET"):
        print_status("Warning: No Modal credentials found. Using LocalSandbox (subprocess).", "bold yellow")
        return LocalSandbox()

    try:
        app = modal.App.lookup("ai-researcher-sandbox", create_if_missing=True)
        
        # Create the sandbox
        sandbox = modal.Sandbox.create(
            image=image,
            app=app,
            gpu=gpu,
            timeout=600,  # 10 minutes
        )
        _shared_sandbox = sandbox
        _shared_gpu = gpu
        return sandbox
    except Exception as e:
        print_status(f"Warning: Failed to create Modal sandbox ({e}). Falling back to LocalSandbox.", "bold yellow")
        return LocalSandbox()
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

    Behavior:
    - Starts a long-lived `python -u -` process in the sandbox.
    - Streams both STDOUT and STDERR to your local CLI *as they are produced*,
      similar to running a long training job in Colab.
    - Captures full STDOUT/STDERR buffers and returns them as a string so the
      agent can inspect logs after the run finishes.
    """
    try:
        sandbox = _get_shared_sandbox(_selected_gpu)

        log_step("EXECUTION", "Launching python exec inside Sandbox...")
        print_panel(code, "Sandbox Code", "code")

        # Use PIPE on both streams so we can capture and stream them ourselves.
        proc = sandbox.exec(
            "python",
            "-u",
            "-",
            stdout=StreamType.PIPE,
            stderr=StreamType.PIPE,
        )

        # Send the code into the sandboxed Python process.
        proc.stdin.write(code.encode("utf-8"))
        proc.stdin.write_eof()
        proc.stdin.drain()  # Flush buffered stdin

        stdout_chunks: List[str] = []
        stderr_chunks: List[str] = []

        log_step("EXECUTION", "Streaming stdout/stderr from Sandbox...")

        def _drain_stream(reader, buffer: List[str], is_stderr: bool):
            """Continuously read from a StreamReader and mirror to local stdout/stderr."""
            try:
                for chunk in reader:
                    # Modal returns text lines (with trailing newline preserved).
                    buffer.append(chunk)
                    if is_stderr:
                        print(chunk, end="", file=sys.stderr, flush=True)
                    else:
                        print(chunk, end="", flush=True)

                    # Also emit a structured streaming event for the web UI so it can
                    # render progress bars and logs as they happen, without waiting
                    # for the entire sandbox run to complete.
                    try:
                        emit_event(
                            "AGENT_STREAM",
                            {
                                "stream": "stderr" if is_stderr else "stdout",
                                "chunk": chunk,
                            },
                        )
                    except Exception as e:
                        # Structured events are best-effort only; don't break execution.
                        log_step("WARNING", f"Failed to emit AGENT_STREAM event: {e}")
            except Exception as e:
                # Don't crash the whole tool if streaming fails; just log.
                stream_name = "stderr" if is_stderr else "stdout"
                log_step("WARNING", f"Error while streaming {stream_name}: {e}")

        # Read stdout and stderr concurrently so training logs / progress bars
        # appear in real time regardless of which stream they use.
        stdout_thread = threading.Thread(
            target=_drain_stream, args=(proc.stdout, stdout_chunks, False), daemon=True
        )
        stderr_thread = threading.Thread(
            target=_drain_stream, args=(proc.stderr, stderr_chunks, True), daemon=True
        )

        stdout_thread.start()
        stderr_thread.start()

        # Wait for the process to finish.
        log_step("EXECUTION", "Waiting for process exit...")
        exit_code = proc.wait()

        # Make sure we've drained any remaining output.
        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)

        log_step("EXECUTION", f"Process exited with code {exit_code}")

        stdout_str = "".join(stdout_chunks)
        stderr_str = "".join(stderr_chunks)

        return f"Exit Code: {exit_code}\nSTDOUT:\n{stdout_str}\nSTDERR:\n{stderr_str}"

    except Exception as e:
        log_step("ERROR", f"Sandbox Execution Failed: {str(e)}")
        return f"Sandbox Execution Failed: {str(e)}"


def _build_claude_tool_definition() -> dict:
    """Build the tool definition for Claude's format."""
    return {
        "name": "execute_in_sandbox",
        "description": (
            "Executes Python code inside a persistent Modal Sandbox. "
            "The sandbox has numpy, pandas, torch, scikit-learn, and matplotlib installed. "
            "Returns the exit code, stdout, and stderr from the execution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute in the sandbox."
                }
            },
            "required": ["code"]
        }
    }


def _build_system_prompt(gpu_hint: str) -> str:
    """System-level instructions for the Gemini agent."""
    prompt = """You are an autonomous research scientist.
Your job is to rigorously verify the user's hypothesis using experiments
run in a Python sandbox.

Tool:
- `execute_in_sandbox(code: str)`: Runs a Python script in a persistent Modal Sandbox.
  - Preinstalled: numpy, pandas, torch, scikit-learn, matplotlib.
  - Compute: Sandbox GPU request for this run: {gpu_hint}.
  - The code runs as a normal Python script; no need to import `modal`.

Working loop:
1. THINK: Plan the next step.
2. EXECUTE: Write and run Python code to test the hypothesis.
3. OBSERVE: Analyze the output.
4. REPEAT: Iterate until you have a solid conclusion.

Output Format:
You must output your thoughts and code clearly.
"""
    return prompt.replace("{gpu_hint}", str(gpu_hint))



def _run_claude_experiment_loop(hypothesis: str, gpu_hint: str):
    """Run the experiment loop using Claude Opus 4.5 with extended thinking."""
    print_status("Claude extended thinking enabled", "info")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    system_prompt = _build_system_prompt(gpu_hint)
    tool_def = _build_claude_tool_definition()

    # Initial conversation with hypothesis
    messages = [
        {"role": "user", "content": f"Hypothesis: {hypothesis}"}
    ]

    max_steps = 10

    for step in range(1, max_steps + 1):
        print_status(f"Step {step}...", "dim")

        try:
            # Use streaming for Claude with extended thinking enabled
            # We need to track thinking blocks with their signatures for proper history
            thinking_blocks = []  # List of {"thinking": str, "signature": str}
            text_content = []
            tool_use_blocks = []

            with client.messages.stream(
                model="claude-opus-4-5-20251101",
                max_tokens=16000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 10000
                },
                system=system_prompt,
                tools=[tool_def],
                messages=messages,
            ) as stream:
                for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_start':
                            if hasattr(event, 'content_block'):
                                block = event.content_block
                                if hasattr(block, 'type'):
                                    if block.type == 'thinking':
                                        thinking_blocks.append({"thinking": "", "signature": None})
                                    elif block.type == 'text':
                                        text_content.append("")
                                    elif block.type == 'tool_use':
                                        tool_use_blocks.append({
                                            "id": block.id,
                                            "name": block.name,
                                            "input": ""
                                        })
                        elif event.type == 'content_block_delta':
                            if hasattr(event, 'delta'):
                                delta = event.delta
                                if hasattr(delta, 'type'):
                                    if delta.type == 'thinking_delta' and hasattr(delta, 'thinking'):
                                        if thinking_blocks:
                                            thinking_blocks[-1]["thinking"] += delta.thinking
                                            emit_event("AGENT_THOUGHT_STREAM", {"chunk": delta.thinking})
                                    elif delta.type == 'text_delta' and hasattr(delta, 'text'):
                                        if text_content:
                                            text_content[-1] += delta.text
                                    elif delta.type == 'input_json_delta' and hasattr(delta, 'partial_json'):
                                        if tool_use_blocks:
                                            tool_use_blocks[-1]["input"] += delta.partial_json
                                    elif delta.type == 'signature_delta' and hasattr(delta, 'signature'):
                                        # Capture signature for thinking blocks
                                        if thinking_blocks:
                                            if thinking_blocks[-1]["signature"] is None:
                                                thinking_blocks[-1]["signature"] = ""
                                            thinking_blocks[-1]["signature"] += delta.signature

        except Exception as e:
            print_status(f"API Error: {e}", "error")
            logger.error(f"API Error: {e}")
            break

        # Process thinking content
        thinking_texts = [tb["thinking"] for tb in thinking_blocks if tb["thinking"]]
        if thinking_texts:
            joined_thinking = "\n\n".join(thinking_texts)
            if joined_thinking:
                print_panel(joined_thinking, "Agent Thinking", "thought")
                log_step("THOUGHT", joined_thinking)

        # Process text content
        if text_content:
            joined_text = "\n\n".join(t for t in text_content if t)
            if joined_text:
                print_panel(joined_text, "Agent Message", "info")
                log_step("MODEL", joined_text)

        # Check for completion
        combined_text = "\n".join(thinking_texts + text_content)
        if "[DONE]" in combined_text:
            print_status("Agent signaled completion.", "success")
            break

        # Build assistant message for history - include signature for thinking blocks
        assistant_content = []
        for tb in thinking_blocks:
            if tb["thinking"]:
                thinking_block = {"type": "thinking", "thinking": tb["thinking"]}
                if tb["signature"]:
                    thinking_block["signature"] = tb["signature"]
                assistant_content.append(thinking_block)
        for t in text_content:
            if t:
                assistant_content.append({"type": "text", "text": t})

        # Process tool calls
        if not tool_use_blocks:
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
            print_status(
                "No tool calls in this step; assuming experiment is complete.", "info"
            )
            break

        # Execute tool calls
        tool_results = []
        for tool_block in tool_use_blocks:
            fn_name = tool_block["name"]
            try:
                fn_args = json.loads(tool_block["input"]) if tool_block["input"] else {}
            except json.JSONDecodeError:
                fn_args = {}

            print_panel(f"{fn_name}({fn_args})", "Tool Call", "code")
            log_step("TOOL_CALL", f"{fn_name}({fn_args})")
            emit_event("AGENT_TOOL", {"tool": fn_name, "args": fn_args})

            # Add tool_use to assistant content
            assistant_content.append({
                "type": "tool_use",
                "id": tool_block["id"],
                "name": fn_name,
                "input": fn_args
            })

            if fn_name == "execute_in_sandbox":
                result = execute_in_sandbox(**fn_args)
            else:
                result = (
                    f"Unsupported tool '{fn_name}'. "
                    "Only 'execute_in_sandbox' is available."
                )

            # Truncate long outputs
            if isinstance(result, str) and len(result) > 20000:
                result = (
                    result[:10000]
                    + "\n...[TRUNCATED]...\n"
                    + result[-10000:]
                )

            print_panel(result, "Tool Result", "result")
            log_step("TOOL_RESULT", "Executed")
            emit_event("AGENT_TOOL_RESULT", {"tool": fn_name, "result": result})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block["id"],
                "content": result
            })

        # Add assistant message and tool results to history
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

    # Final report generation
    try:
        print_status("Generating Final Report...", "bold green")
        messages.append({
            "role": "user",
            "content": (
                "Generate a concise, information-dense report that explains "
                "how you tested the hypothesis, what you observed, and your "
                "final conclusion."
            )
        })

        final_thinking = []
        final_text = []

        with client.messages.stream(
            model="claude-opus-4-5-20251101",
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": 10000
            },
            system=system_prompt,
            messages=messages,
        ) as stream:
            for event in stream:
                if hasattr(event, 'type'):
                    if event.type == 'content_block_start':
                        if hasattr(event, 'content_block'):
                            block = event.content_block
                            if hasattr(block, 'type'):
                                if block.type == 'thinking':
                                    final_thinking.append("")
                                elif block.type == 'text':
                                    final_text.append("")
                    elif event.type == 'content_block_delta':
                        if hasattr(event, 'delta'):
                            delta = event.delta
                            if hasattr(delta, 'type'):
                                if delta.type == 'thinking_delta' and hasattr(delta, 'thinking'):
                                    if final_thinking:
                                        final_thinking[-1] += delta.thinking
                                        emit_event("AGENT_THOUGHT_STREAM", {"chunk": delta.thinking})
                                elif delta.type == 'text_delta' and hasattr(delta, 'text'):
                                    if final_text:
                                        final_text[-1] += delta.text

        final_report = "\n\n".join(t for t in final_text if t)
        print_panel(final_report, "Final Report", "bold green")
    finally:
        _close_shared_sandbox()



def _build_deepseek_tool_definition() -> dict:
    """Build the tool definition for DeepSeek (OpenAI compatible)."""
    return {
        "type": "function",
        "function": {
            "name": "execute_in_sandbox",
            "description": (
                "Executes Python code inside a persistent Modal Sandbox. "
                "The sandbox has numpy, pandas, torch, scikit-learn, and matplotlib installed. "
                "Returns the exit code, stdout, and stderr from the execution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute in the sandbox."
                    }
                },
                "required": ["code"]
            }
        }
    }


def _run_deepseek_experiment_loop(hypothesis: str, gpu_hint: str):
    """Run the experiment loop using DeepSeek Chat."""
    print_status("DeepSeek Chat enabled", "info")

    client = openai.OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com"
    )
    
    system_prompt = _build_system_prompt(gpu_hint)
    tools = [_build_deepseek_tool_definition()]

    # Initial conversation with hypothesis
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Hypothesis: {hypothesis}"}
    ]

    max_steps = 10

    for step in range(1, max_steps + 1):
        print_status(f"Step {step}...", "dim")

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=tools,
                stream=True
            )

            collected_content = []
            tool_calls = []
            current_tool_call = None

            print_panel("", "DeepSeek Stream", "dim") # Placeholder for stream start

            for chunk in response:
                delta = chunk.choices[0].delta
                
                # Handle content
                if delta.content:
                    content_chunk = delta.content
                    collected_content.append(content_chunk)
                    print(content_chunk, end="", flush=True)
                    emit_event("AGENT_THOUGHT_STREAM", {"chunk": content_chunk})

                # Handle tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.index is not None:
                            # If we have a new index, it might be a new tool call or continuation
                            if len(tool_calls) <= tc.index:
                                tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}})
                            
                            current_tool_call = tool_calls[tc.index]
                            
                            if tc.id:
                                current_tool_call["id"] += tc.id
                            
                            if tc.function:
                                if tc.function.name:
                                    current_tool_call["function"]["name"] += tc.function.name
                                if tc.function.arguments:
                                    current_tool_call["function"]["arguments"] += tc.function.arguments

            print() # Newline after stream

            full_content = "".join(collected_content)
            
            # Add assistant message to history
            assistant_msg = {"role": "assistant", "content": full_content}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": tc["function"]
                    } for tc in tool_calls
                ]
            messages.append(assistant_msg)

            # Log content
            if full_content:
                # Heuristic to separate thought from message if possible, 
                # but DeepSeek might mix them. We'll just log as MODEL.
                log_step("MODEL", full_content)
                
                if "[DONE]" in full_content:
                    print_status("Agent signaled completion.", "success")
                    break

            # Process tool calls
            if not tool_calls:
                if not full_content:
                     print_status("Empty response from DeepSeek.", "warning")
                else:
                    print_status("No tool calls in this step.", "info")
                
                # If no tool calls and we have content, we continue unless [DONE] was found
                if "[DONE]" in full_content:
                    break
                # If just text and no done, maybe it's asking a question or thinking. 
                # We'll let it continue but usually the loop expects action.
                # For now, if no tool call, we might just stop or continue. 
                # The original logic breaks if no tool calls.
                if not "[DONE]" in full_content:
                     print_status("No tool calls and not DONE. Stopping.", "warning")
                     break
                continue

            # Execute tool calls
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}
                
                tool_call_id = tc["id"]

                print_panel(f"{fn_name}({fn_args})", "Tool Call", "code")
                log_step("TOOL_CALL", f"{fn_name}({fn_args})")
                emit_event("AGENT_TOOL", {"tool": fn_name, "args": fn_args})

                if fn_name == "execute_in_sandbox":
                    result = execute_in_sandbox(**fn_args)
                else:
                    result = f"Unsupported tool '{fn_name}'"

                # Truncate result
                if len(result) > 20000:
                    result = result[:10000] + "\n...[TRUNCATED]...\n" + result[-10000:]

                print_panel(result, "Tool Result", "result")
                log_step("TOOL_RESULT", "Executed")
                emit_event("AGENT_TOOL_RESULT", {"tool": fn_name, "result": result})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result
                })

        except Exception as e:
            print_status(f"API Error: {e}", "error")
            logger.error(f"API Error: {e}")
            break

    # Final report generation
    try:
        print_status("Generating Final Report...", "bold green")
        messages.append({
            "role": "user",
            "content": "Generate a concise, information-dense report that explains how you tested the hypothesis, what you observed, and your final conclusion."
        })

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True
        )

        final_content = []
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                final_content.append(content)
                print(content, end="", flush=True)
        
        print()
        final_report = "".join(final_content)
        print_panel(final_report, "Final Report", "bold green")

    finally:
        _close_shared_sandbox()


def _run_gemini_experiment_loop(hypothesis: str, gpu_hint: str):
    """Run the experiment loop using Gemini 3 Pro with thinking mode."""
    print_status("Gemini thinking: HIGH (thought summaries visible)", "info")

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    # Expose the sandbox executor as a tool.
    tools = [execute_in_sandbox]
    system_prompt = _build_system_prompt(gpu_hint)

    # Initial conversation: just the hypothesis as a user message.
    history: List[types.Content] = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"Hypothesis: {hypothesis}")],
        )
    ]

    max_steps = 10

    for step in range(1, max_steps + 1):
        print_status(f"Step {step}...", "dim")

        try:
            # Stream the model's response so we can surface thinking and tool calls in real time.
            response_stream = client.models.generate_content_stream(
                model="gemini-3-pro-preview",
                contents=history,
                config=_build_generation_config(
                    tools=tools,
                    system_instruction=system_prompt,
                    disable_autofc=True,  # manual tool loop
                ),
            )
        except Exception as e:
            print_status(f"API Error: {e}", "error")
            logger.error(f"API Error: {e}")
            break

        # Accumulate full response for history and logic
        accumulated_parts = []

        # Track chunks
        for chunk in response_stream:
             if not chunk.candidates:
                 continue
             
             candidate = chunk.candidates[0]
             if not candidate.content or not candidate.content.parts:
                 continue
             
             for part in candidate.content.parts:
                 # 1. Streaming thoughts
                 if getattr(part, "thought", False) and part.text:
                     emit_event("AGENT_THOUGHT_STREAM", {"chunk": part.text})
                 
                 # Add to accumulator
                 accumulated_parts.append(part)

        # Reconstruct the full Content object (merge logic similar to orchestrator)
        merged_parts = []
        current_text_part = None
        current_thought_part = None
        
        for part in accumulated_parts:
            # Handle Function Calls
            if part.function_call:
                if current_text_part:
                    merged_parts.append(current_text_part)
                    current_text_part = None
                if current_thought_part:
                    merged_parts.append(current_thought_part)
                    current_thought_part = None
                merged_parts.append(part)
                continue
                
            # Handle Thoughts
            if getattr(part, "thought", False):
                if current_text_part:
                    merged_parts.append(current_text_part)
                    current_text_part = None
                
                if current_thought_part:
                    current_thought_part.text += part.text
                else:
                    current_thought_part = part
                continue

            # Handle Text
            if part.text:
                if current_thought_part:
                    merged_parts.append(current_thought_part)
                    current_thought_part = None
                    
                if current_text_part:
                    current_text_part.text += part.text
                else:
                    current_text_part = part
                continue
        
        if current_text_part:
            merged_parts.append(current_text_part)
        if current_thought_part:
            merged_parts.append(current_thought_part)

        if not merged_parts:
            print_status("Empty content from model.", "warning")
            break
            
        model_content = types.Content(role="model", parts=merged_parts)

        # IMPORTANT: append the full model message (including thought signatures
        # and function call parts) so the SDK can preserve reasoning state.
        history.append(model_content)

        thoughts: List[str] = []
        messages: List[str] = []
        function_calls = []

        for part in model_content.parts:
            # Thought summaries from thinking mode.
            if getattr(part, "thought", False) and part.text:
                thoughts.append(part.text)

            # Function/tool call parts.
            if part.function_call:
                function_calls.append(part.function_call)

            # Regular assistant text (exclude thought parts so we don't double-print).
            if part.text and not getattr(part, "thought", False):
                messages.append(part.text)

        # 1. Show reasoning before any action.
        if thoughts:
            joined_thoughts = "\n\n".join(thoughts)
            print_panel(joined_thoughts, "Agent Thinking", "thought")
            log_step("THOUGHT", joined_thoughts)

        # 2. Show natural-language messages (plans, explanations, etc.).
        if messages:
            joined_messages = "\n\n".join(messages)
            print_panel(joined_messages, "Agent Message", "info")
            log_step("MODEL", joined_messages)

        combined_text = "\n".join(thoughts + messages)
        if "[DONE]" in combined_text:
            print_status("Agent signaled completion.", "success")
            break

        # If the model didn't call any tools this turn, assume we're done.
        if not function_calls:
            print_status(
                "No tool calls in this step; assuming experiment is complete.", "info"
            )
            break

        # 3. Execute requested tools (currently just execute_in_sandbox).
        for fn_call in function_calls:
            fn_name = fn_call.name
            fn_args = dict(fn_call.args or {})

            print_panel(f"{fn_name}({fn_args})", "Tool Call", "code")
            log_step("TOOL_CALL", f"{fn_name}({fn_args})")
            emit_event("AGENT_TOOL", {"tool": fn_name, "args": fn_args})

            if fn_name == "execute_in_sandbox":
                result = execute_in_sandbox(**fn_args)
            else:
                result = (
                    f"Unsupported tool '{fn_name}'. "
                    "Only 'execute_in_sandbox' is available."
                )

            # Truncate long outputs to keep console readable.
            if isinstance(result, str) and len(result) > 20000:
                result = (
                    result[:10000]
                    + "\n...[TRUNCATED]...\n"
                    + result[-10000:]
                )

            print_panel(result, "Tool Result", "result")
            log_step("TOOL_RESULT", "Executed")
            emit_event("AGENT_TOOL_RESULT", {"tool": fn_name, "result": result})

            # Feed the tool response back as a TOOL message with a functionResponse part.
            history.append(
                types.Content(
                    role="tool",
                    parts=[
                        types.Part.from_function_response(
                            name=fn_name,
                            response={"result": result},
                        )
                    ],
                )
            )

    # Final report generation.
    try:
        print_status("Generating Final Report...", "bold green")
        history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Generate a concise, information-dense report that explains "
                            "how you tested the hypothesis, what you observed, and your "
                            "final conclusion."
                        )
                    )
                ],
            )
        )

        final_response_stream = client.models.generate_content_stream(
            model="gemini-3-pro-preview",
            contents=history,
            # Still use thinking so the model can reason about its own trace,
            # but tools are not needed here.
            config=_build_generation_config(
                tools=None,
                system_instruction=system_prompt,
                disable_autofc=True,
            ),
        )

        final_parts = []
        for chunk in final_response_stream:
            if chunk.candidates and chunk.candidates[0].content:
                for part in chunk.candidates[0].content.parts:
                    if getattr(part, "thought", False) and part.text:
                        emit_event("AGENT_THOUGHT_STREAM", {"chunk": part.text})
                    final_parts.append(part)

        # Basic merge for final text extraction
        final_text = ""
        for part in final_parts:
            if part.text and not getattr(part, "thought", False):
                final_text += part.text
        
        print_panel(final_text, "Final Report", "bold green")
    finally:
        _close_shared_sandbox()

def _run_deepseek_experiment_loop(hypothesis: str, gpu_hint: str = None):
    print_status("Starting DeepSeek Experiment...", "bold blue")
    
    system_prompt = _build_system_prompt(gpu_hint)
    
    client = openai.OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Hypothesis: {hypothesis}"}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_in_sandbox",
                "description": "Execute Python code in a sandbox environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute."
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    ]

    while True:
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=tools,
                stream=False
            )
            
            message = response.choices[0].message
            messages.append(message)

            if message.content:
                print_panel(message.content, "DeepSeek Thought", "bold yellow")
                emit_event("AGENT_THOUGHT", {"thought": message.content})

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    
                    print_panel(f"{fn_name}({fn_args})", "Tool Call", "code")
                    emit_event("AGENT_TOOL", {"tool": fn_name, "args": fn_args})
                    
                    if fn_name == "execute_in_sandbox":
                        result = execute_in_sandbox(**fn_args)
                    else:
                        result = f"Error: Tool {fn_name} not found."
                    
                    print_panel(result, "Tool Result", "result")
                    emit_event("AGENT_TOOL_RESULT", {"tool": fn_name, "result": result})
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                # No tool calls, assume finished
                print_status("DeepSeek indicated completion.", "success")
                break
                
        except Exception as e:
            print_status(f"Error in DeepSeek loop: {e}", "bold red")
            break

def run_experiment_loop(hypothesis: str, test_mode: bool = False, model: str = "gemini-3-pro-preview"):
    if test_mode:
        print_status("Running in TEST MODE (Mock)", "bold yellow")
        return

    gpu_hint = _selected_gpu
    
    if model == "deepseek-chat":
        _run_deepseek_experiment_loop(hypothesis, gpu_hint)
    elif model == "claude-opus-4-5":
        if "_run_claude_experiment_loop" in globals():
             globals()["_run_claude_experiment_loop"](hypothesis, gpu_hint)
        else:
             print_status("Claude support not available.", "bold red")
    else:
        _run_gemini_experiment_loop(hypothesis, gpu_hint)

