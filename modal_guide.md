# Modal Guide for AI Agents

This guide provides a concise, accurate reference for using Modal to spin up GPUs, run code, track costs, and manage data.

## 1. Core Concepts
- **App**: The unit of deployment. Defined as `app = modal.App("my-app-name")`.
- **Image**: The environment (OS + Python packages).
- **Function**: The code that runs remotely. Decorated with `@app.function`.
- **Volume**: Persistent storage for large files (datasets, models).

## 2. Project Structure
A standard Modal project is a single Python file (e.g., `experiment.py`) or a module.

```python
import modal

app = modal.App("experiment-01")

# Define the environment
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "numpy")
)

@app.function(image=image, gpu="A100")
def run_experiment(params: dict):
    import torch
    print(f"Running on {torch.cuda.get_device_name(0)}")
    # ... experiment logic ...
    return {"status": "success", "loss": 0.01}

@app.local_entrypoint()
def main():
    print("Starting remote experiment...")
    result = run_experiment.remote({"learning_rate": 0.001})
    print(f"Result: {result}")
```

## 3. Spinning Up GPUs
Specify the `gpu` argument in the `@app.function` decorator.

### GPU Types
- **H100**: `gpu="H100"` (Most powerful, scarce)
- **A100**: `gpu="A100"` (Standard for LLM training/inference)
    - `gpu="A100-80GB"` (Force 80GB memory)
- **A10G**: `gpu="A10G"` (Good price/performance for inference)
- **T4**: `gpu="T4"` (Cheap, older)
- **Any**: `gpu="any"` (Lowest availability latency)

### Multi-GPU
Append `:N` to the type string.
```python
@app.function(gpu="A100:4") # Request 4 A100s
```

## 4. Persistent Storage (Volumes)
Use `modal.Volume` to persist data across runs.

### Creating & Mounting
```python
# Create/Get volume
volume = modal.Volume.from_name("my-dataset-vol", create_if_missing=True)

@app.function(volumes={"/data": volume})
def process_data():
    # Read
    with open("/data/input.txt", "r") as f:
        data = f.read()
    
    # Write
    with open("/data/output.txt", "w") as f:
        f.write("processed")
    
    # CRITICAL: Commit changes to persist them!
    volume.commit() 
```

### Reloading
If another function updates the volume, reload it to see changes:
```python
volume.reload()
```

## 5. Secrets & API Keys
Inject environment variables securely.

1.  **Create Secret** (CLI or Dashboard):
    `modal secret create my-huggingface-secret HF_TOKEN=hf_...`

2.  **Use in Code**:
```python
@app.function(secrets=[modal.Secret.from_name("my-huggingface-secret")])
def download_model():
    import os
    token = os.environ["HF_TOKEN"]
```

## 6. Running Code
### CLI
Run the local entrypoint:
```bash
modal run experiment.py
```

### Remote Execution
- **`func.remote(args)`**: Synchronous call. Returns the result.
- **`func.spawn(args)`**: Asynchronous call. Returns a `FunctionCall` object.
    ```python
    job = run_experiment.spawn(params)
    # ... do other work ...
    result = job.get()
    ```

## 7. Web Endpoints
Expose a function as a web endpoint (useful for agent-to-agent communication).

```python
@app.function()
@modal.web_endpoint(method="POST")
def webhook(data: dict):
    return {"received": data}
```
*URL is printed to stdout on deploy.*

## 8. Observability & Costs
### Logs
- **Live**: Streamed to your terminal during `modal run`.
- **Dashboard**: View full logs at `https://modal.com/apps`.
- **Programmatic**: Currently, logs are best consumed via the dashboard or by redirecting stdout in the function to a file on a Volume.

### Cost Tracking
- **Pricing**: Usage-based (per second).
    - A100: ~$0.000694/sec
    - H100: ~$0.001097/sec
    - CPU: Very cheap
- **Dashboard**: View "Usage & Billing" in the Modal dashboard for exact costs per app/function.
- **Optimization**:
    - Use `modal.Image` caching (builds are cached).
    - Scale down to 0 automatically (serverless).

## 9. Quick Reference Checklist
- [ ] **Decorator**: `@app.function(image=..., gpu=..., volumes=..., secrets=...)`
- [ ] **Entrypoint**: `@app.local_entrypoint()`
- [ ] **Run**: `modal run file.py`
- [ ] **Persist**: `volume.commit()` after writes.
- [ ] **Secrets**: `modal.Secret.from_name("name")`
