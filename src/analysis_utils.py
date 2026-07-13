import time
import torch


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_bytes / (1024 ** 2)


@torch.no_grad()
def measure_latency(model, input_size=(1, 3, 416, 416), device="cpu", n_warmup=5, n_runs=30):
    model.eval().to(device)
    x = torch.randn(*input_size, device=device)
    for _ in range(n_warmup):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n_runs
    return dt * 1000  # ms/image
