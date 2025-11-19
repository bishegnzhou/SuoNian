import modal

app = modal.App("fib-gpu-test-1")

image = modal.Image.debian_slim().pip_install("torch")

@app.function(image=image, gpu="any")
def calculate_fib_iterative():
    import torch
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n = 10
    # Initializing tensors on GPU
    t0 = torch.tensor(0, device=device)
    t1 = torch.tensor(1, device=device)
    
    if n == 0:
        res = t0
    elif n == 1:
        res = t1
    else:
        for i in range(2, n + 1):
            temp = t0 + t1
            t0 = t1
            t1 = temp
        res = t1

    print(f"Fibonacci({n}) calculated on {device}: {res.item()}")
    return res.item()

@app.local_entrypoint()
def main():
    print("Starting Test 1: Iterative Approach")
    result = calculate_fib_iterative.remote()
    print(f"Result 1: {result}")
