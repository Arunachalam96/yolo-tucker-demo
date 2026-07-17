"""
tucker_pipeline.py
------------------
Tucker-2 compression for YOLOv3, faithfully following the validated CNN
pipeline methodology (VGG/ResNet/MobileNet/ViT), adapted for object
detection. The ONLY task-specific substitution is the quality metric on
the polynomial's y-axis: top-1 accuracy -> mAP@0.5. Everything else -- the
std-based noise metric, the 2D (r_in, r_out) grid sweep, the symmetrized
biquadratic fit, and the invert->Pareto->KMeans rank selection -- matches
the CNN pipeline exactly.

Two-phase structure (unchanged from CNN pipeline):
  Phase 1 (pre-run offline, checkpointed per layer):
    - for each compressible conv, sweep a 2D grid of (r_in, r_out) ranks
      at step = 20% of min(C_in, C_out)
    - per combo: build TuckerBlock from the layer's weights, swap into a
      copy of the model, measure mAP@0.5 (subsampled val) and noise %
      (std-based, via forward hooks on a stratified sample)
    - fit a degree-4 EVEN (biquadratic) polynomial mAP-vs-noise, with data
      symmetrization forcing odd coefficients to zero
  Phase 2 (live, pure math on stored Phase-1 JSON):
    - invert each layer's polynomial for the max tolerable noise given a
      global mAP budget
    - Pareto front over (compression_ratio, mAP)
    - KMeans (k=3) -> Conservative / Balanced / Aggressive suggestions
"""
import copy
import gc
import numpy as np
import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import tucker
from scipy.optimize import brentq
from sklearn.cluster import KMeans

# NumPy backend, matching the reference CNN pipeline exactly (decomposition
# runs on CPU via numpy, results copied back into torch convs).
tl.set_backend("numpy")


# ---------------------------------------------------------------------------
# TuckerBlock -- direct port of the CNN pipeline's block
# ---------------------------------------------------------------------------

class TuckerBlock(nn.Module):
    """
    Replaces a single Conv2d(C_in, C_out, kxk) with three convolutions:
        Conv2d(C_in,  r_in,  1x1, bias=False)   -- input projection
        Conv2d(r_in,  r_out, kxk, bias=False)   -- core spatial conv
        Conv2d(r_out, C_out, 1x1, bias=True)    -- output projection

    No activation/BN inside: in the YOLO model these are separate layers in
    the enclosing Sequential (conv_bn_leaky) and stay in place after the
    block, exactly as ReLU does after the CNN pipeline's TuckerBlock.
    """
    def __init__(self, C_in, r_in, r_out, C_out, kH, kW, padding, stride,
                 dilation=(1, 1), bias_data=None):
        super().__init__()
        self.conv_in = nn.Conv2d(C_in, r_in, 1, bias=False)
        self.conv_core = nn.Conv2d(r_in, r_out, (kH, kW),
                                    padding=padding, stride=stride,
                                    dilation=dilation, bias=False)
        self.conv_out = nn.Conv2d(r_out, C_out, 1, bias=(bias_data is not None))

    def forward(self, x):
        return self.conv_out(self.conv_core(self.conv_in(x)))


def tucker_decompose_layer(module, r_in, r_out):
    """
    Tucker-2 decomposition of a Conv2d -- direct port of the CNN pipeline.

    PyTorch Conv2d weight shape: (C_out, C_in, kH, kW). Decompose ALL 4
    modes with ranks [r_out, r_in, kH, kW] (full spatial rank, channels
    compressed), then fold the spatial factors back into the core via
    mode_dot so conv_core carries the reconstructed spatial kernel.
    """
    W = module.weight.data.cpu().numpy()  # (C_out, C_in, kH, kW)
    C_out, C_in, kH, kW = W.shape
    r_in = max(1, min(r_in, C_in))
    r_out = max(1, min(r_out, C_out))

    core, factors = tucker(W, rank=[r_out, r_in, kH, kW], init="svd")
    # factors[0]:(C_out,r_out) [1]:(C_in,r_in) [2]:(kH,kH) [3]:(kW,kW)

    new_core = tl.tenalg.mode_dot(core, factors[2], mode=2)
    new_core = tl.tenalg.mode_dot(new_core, factors[3], mode=3)

    W_in = factors[1].T.reshape(r_in, C_in, 1, 1).astype(np.float32)
    W_core = new_core.astype(np.float32)
    W_out = factors[0].reshape(C_out, r_out, 1, 1).astype(np.float32)

    bias_data = None
    if module.bias is not None:
        bias_data = module.bias.data.cpu().numpy().astype(np.float32)

    block = TuckerBlock(C_in, r_in, r_out, C_out, kH, kW,
                        module.padding, module.stride,
                        dilation=module.dilation, bias_data=bias_data)
    block.conv_in.weight.data = torch.from_numpy(W_in)
    block.conv_core.weight.data = torch.from_numpy(W_core)
    block.conv_out.weight.data = torch.from_numpy(W_out)
    if bias_data is not None:
        block.conv_out.bias.data = torch.from_numpy(bias_data)
    return block


# ---------------------------------------------------------------------------
# Module utilities
# ---------------------------------------------------------------------------

def get_module_by_name(model, name):
    module = model
    for part in name.split("."):
        module = getattr(module, part)
    return module


def set_module_by_name(model, name, new_module):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def build_decomposed_model(orig_model, layer_name, tucker_block):
    """Deep-copy orig_model and replace the Conv2d at layer_name. Original
    is never modified (matches CNN pipeline)."""
    new_model = copy.deepcopy(orig_model)
    set_module_by_name(new_model, layer_name, tucker_block)
    return new_model


def get_compressible_layers(model, skip_first=True):
    """
    Conv2d layers with spatial kernel > 1x1, EXCEPT the stem (first such
    conv) and the three detection-head prediction convs (head_*.pred) --
    the YOLO analog of the CNN pipeline's 'skip first conv' plus the
    'skip classification head' convention. 1x1 convs are skipped too (the
    CNN pipeline only sweeps kernel>1x1 layers).
    """
    result = []
    first_seen = False
    pred_suffixes = ("head_large.pred", "head_medium.pred", "head_small.pred")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            kH, kW = module.kernel_size
            if kH > 1 or kW > 1:
                if skip_first and not first_seen:
                    first_seen = True
                    continue
                first_seen = True
                if any(name.endswith(s) for s in pred_suffixes):
                    continue
                result.append({
                    "name": name, "C_in": module.in_channels,
                    "C_out": module.out_channels, "kH": kH, "kW": kW,
                })
    return result


def compute_compression_ratio(module, r_in, r_out):
    """Parameter count ratio: original / tucker-decomposed (matches CNN)."""
    C_out, C_in, kH, kW = module.weight.data.shape
    original = C_in * kH * kW * C_out
    decomposed = C_in * r_in + r_in * kH * kW * r_out + r_out * C_out
    return original / decomposed if decomposed > 0 else 0.0


# ---------------------------------------------------------------------------
# Noise metric -- direct port (std-based, via forward hooks)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_noise_percent(orig_model, tmp_model, layer_name, X_sample, device,
                           batch_size=16):
    """
    Noise % = std(error - mean(error)) / std(original_output) * 100

    Compares the ORIGINAL conv's output activations against the
    TuckerBlock's output activations over a stratified sample, via forward
    hooks. Task-agnostic -- identical to the CNN pipeline.
    """
    orig_acts, decomp_acts = [], []

    def make_hook(storage):
        def hook(module, inp, output):
            storage.append(output.detach().cpu())
        return hook

    h_orig = get_module_by_name(orig_model, layer_name).register_forward_hook(make_hook(orig_acts))
    h_decomp = get_module_by_name(tmp_model, layer_name).register_forward_hook(make_hook(decomp_acts))

    orig_model.eval(); tmp_model.eval()
    orig_model.to(device); tmp_model.to(device)

    try:
        n = X_sample.shape[0]
        for start in range(0, n, batch_size):
            batch = X_sample[start:start + batch_size].to(device)
            orig_model(batch)
            tmp_model(batch)
    except Exception as e:
        h_orig.remove(); h_decomp.remove()
        print(f"    [Noise] forward failed: {e}")
        return float("nan")

    h_orig.remove(); h_decomp.remove()
    if not orig_acts or not decomp_acts:
        return float("nan")

    out_orig = torch.cat(orig_acts, dim=0).numpy()
    out_decomp = torch.cat(decomp_acts, dim=0).numpy()
    err = out_orig - out_decomp
    err_centered = err - np.mean(err)
    return float((np.std(err_centered) / (np.std(out_orig) + 1e-12)) * 100.0)


# ---------------------------------------------------------------------------
# Polynomial fitting -- direct port (symmetrized biquadratic)
# ---------------------------------------------------------------------------

def fit_biquadratic_polynomial(noise_values, quality_values):
    """
    Fit degree-4 even (biquadratic) polynomial: quality vs noise.
    Data symmetrization: mirror to (-noise, quality) so odd-degree
    coefficients are forced to exactly 0. Returns [c4, 0, c2, 0, c0].
    `quality` is mAP@0.5 for YOLO (top-1 accuracy in the CNN pipeline).
    """
    noise = np.array(noise_values, dtype=np.float64)
    q = np.array(quality_values, dtype=np.float64)
    x = np.concatenate([noise, -noise])
    y = np.concatenate([q, q])
    coeffs = np.polyfit(x, y, 4)
    coeffs[1] = 0.0
    coeffs[3] = 0.0
    return coeffs.tolist()


# ---------------------------------------------------------------------------
# Phase 2 -- direct port (invert -> Pareto -> KMeans)
# ---------------------------------------------------------------------------

def invert_polynomial_for_noise(coeffs, baseline_quality, quality_budget):
    """
    Max noise where poly(noise) >= baseline_quality - budget.
    `quality_budget` is in the same units as the quality metric (for mAP,
    an absolute mAP drop, e.g. 0.02 = allow 0.02 mAP loss).
    """
    target = baseline_quality - quality_budget
    poly = np.poly1d(coeffs)
    search = np.linspace(0, 100, 10000)
    f = poly(search) - target
    crossings = []
    for i in range(len(f) - 1):
        if f[i] * f[i + 1] < 0:
            try:
                crossings.append(brentq(lambda n: poly(n) - target, search[i], search[i + 1]))
            except Exception:
                pass
    if not crossings:
        return 100.0 if np.all(f >= 0) else None
    return max(crossings)


def compute_pareto_front(candidates):
    """Pareto front maximising compression_ratio and quality (mAP)."""
    n = len(candidates)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (candidates[j]["compression_ratio"] >= candidates[i]["compression_ratio"] and
                    candidates[j]["quality"] >= candidates[i]["quality"] and
                    (candidates[j]["compression_ratio"] > candidates[i]["compression_ratio"] or
                     candidates[j]["quality"] > candidates[i]["quality"])):
                is_pareto[i] = False
                break
    return [i for i in range(n) if is_pareto[i]]


def cluster_pareto_suggestions(pareto_candidates, n_clusters=3):
    label_names = ["Conservative", "Balanced", "Aggressive"]
    if len(pareto_candidates) < n_clusters:
        sorted_c = sorted(pareto_candidates, key=lambda x: x["compression_ratio"])
        for i, c in enumerate(sorted_c):
            c["label"] = label_names[min(i, 2)]
        return sorted_c
    X = np.array([[c["compression_ratio"], c["quality"]] for c in pareto_candidates])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    ids = km.fit_predict(X)
    order = np.argsort(km.cluster_centers_[:, 0])
    lmap = {int(cid): label_names[rank] for rank, cid in enumerate(order)}
    for i, c in enumerate(pareto_candidates):
        c["label"] = lmap[int(ids[i])]
    return pareto_candidates


def select_ranks_for_layer(layer_name, sweep_results, poly_coeffs,
                            baseline_quality, quality_budget):
    max_noise = invert_polynomial_for_noise(poly_coeffs, baseline_quality, quality_budget)
    if max_noise is None:
        return {"layer": layer_name, "max_noise": None, "suggestions": []}

    valid = [r for r in sweep_results
             if not np.isnan(r["noise_percent"]) and r["noise_percent"] <= max_noise]
    if not valid:
        return {"layer": layer_name, "max_noise": float(max_noise), "suggestions": []}

    # rename quality key for the pareto/cluster helpers
    for r in valid:
        r["quality"] = r["mAP"]
    pareto_idx = compute_pareto_front(valid)
    pareto = [copy.deepcopy(valid[i]) for i in pareto_idx]
    suggestions = cluster_pareto_suggestions(pareto)
    suggestions = sorted(suggestions, key=lambda x: x["compression_ratio"])
    return {
        "layer": layer_name, "max_noise": float(max_noise),
        "pareto_count": len(pareto), "suggestions": suggestions,
    }


# ---------------------------------------------------------------------------
# Post-compression BatchNorm recalibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def recalibrate_batchnorm(model, loader, device, num_batches=20):
    """
    Tucker decomposition changes the activation statistics flowing into
    every BatchNorm2d that follows a compressed conv, but each BN layer's
    running mean/variance is still calibrated for the *original* conv's
    output distribution. This mismatch causes erratic, image-dependent
    post-compression predictions even when the low-rank weights are good.

    Recomputes BN running stats via forward-only passes (train() mode so BN
    updates stats, no_grad so nothing else changes) -- no gradient step, no
    weight update, just re-syncing BN to the factored layers' real outputs.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    for i, (imgs, _) in enumerate(loader):
        if i >= num_batches:
            break
        model(imgs.to(device))
    model.eval()
    return model


def apply_selected_ranks(model, selections, device):
    """
    Given Phase-2 per-layer selections (each with a chosen r_in/r_out),
    decompose and swap every selected layer into `model` in place. Used by
    the live Phase-2 notebook after picking one suggestion tier per layer.
    `selections` is {layer_name: {"r_in": int, "r_out": int}}.
    """
    for name, sel in selections.items():
        conv = get_module_by_name(model, name)
        if not isinstance(conv, nn.Conv2d):
            continue  # already replaced or not found
        block = tucker_decompose_layer(conv, sel["r_in"], sel["r_out"])
        set_module_by_name(model, name, block)
    model.to(device)
    return model
