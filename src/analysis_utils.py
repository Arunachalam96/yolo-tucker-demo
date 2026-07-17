import time
import torch
import torch.nn as nn


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_bytes / (1024 ** 2)


@torch.no_grad()
def count_flops(model, input_size=(1, 3, 416, 416), device="cpu"):
    """
    Multiply-accumulate FLOPs for every Conv2d, via forward hooks that read
    the actual output shape produced during a real forward pass (so it's
    correct regardless of stride/padding/dilation, and works the same for
    both the original and Tucker-2-factored conv sequences without any
    special-casing).

    FLOPs per conv = 2 * Cin * Cout * k_h * k_w * Hout * Wout / groups
    (the factor of 2 counts multiply + accumulate; groups=1 everywhere here).
    """
    total_flops = [0]
    hooks = []

    def hook(module, inp, out):
        Cout, Hout, Wout = out.shape[1], out.shape[2], out.shape[3]
        Cin = module.in_channels
        kh, kw = module.kernel_size
        total_flops[0] += 2 * (Cin // module.groups) * Cout * kh * kw * Hout * Wout

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook))

    model.eval().to(device)
    x = torch.randn(*input_size, device=device)
    model(x)

    for h in hooks:
        h.remove()
    return total_flops[0]


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
