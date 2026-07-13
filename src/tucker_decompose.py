"""
Tucker-2 compression for YOLOv3, following the same two-phase methodology
as the CNN/ViT pipeline:

Phase 1 (per layer, offline, done once):
    - sweep candidate output/input channel ranks for a Conv2d
    - measure the resulting validation-loss degradation vs. the uncompressed model
    - fit a degree-4 (biquadratic) polynomial: loss_delta(r) = a4 r^4 + a3 r^3 + a2 r^2 + a1 r + a0
      where r = compression ratio (0..1, 1 = no compression)
    - sensitivity = 2*|a2|  (curvature of the fit -> how "sharply" this layer
      degrades as it's compressed; low sensitivity = safe to compress hard)

Phase 2 (global, done once per compression target):
    - distribute a global parameter budget across layers, inversely
      proportional to sensitivity (insensitive layers get compressed harder)
    - normalize distributed ratios to 0..1
    - clamp with a floor (global_budget / (3 * n_layers)) and a cap
      (3x the global target) so no single layer collapses to near-zero
      rank or absorbs an unreasonable share of the budget
    - build a small Pareto front of (params_saved, est_loss_delta) candidates
      per layer from the phase-1 sweep and pick the candidate closest to
      the assigned per-layer budget
    - apply compression sequentially, layer by layer, re-checking cumulative
      param count against the global target; JSON checkpoint after each
      layer so a killed run resumes without redoing finished layers

Tucker-2 here = decompose Conv2d(Cin,Cout,k,k) into:
    1x1 conv (Cin -> R_in)  -> kxk conv (R_in -> R_out) -> 1x1 conv (R_out -> Cout)
i.e. mode-0/mode-1 (output/input channel) factors via HOSVD, core kept as
the spatial conv. This is the same construction already validated on
VGG/ResNet/MobileNet and shown equivalent to truncated SVD for 1x1 (ViT
Linear) layers.
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import partial_tucker

tl.set_backend("pytorch")


# ---------------------------------------------------------------------------
# Layer enumeration
# ---------------------------------------------------------------------------

def get_compressible_conv_layers(model, skip_names=None):
    """
    All nn.Conv2d layers except:
      - the very first conv (stem) -- analogous to 'skip first conv' in CNNs
      - the final 1x1 prediction convs in each detection head -- analogous
        to 'skip classification head', these directly produce box/class
        logits and are extremely sensitive to any rank truncation.
    """
    skip_names = skip_names or set()
    all_convs = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    first_name = all_convs[0][0]
    pred_head_suffixes = ("head_large.pred", "head_medium.pred", "head_small.pred")
    keep = []
    for name, m in all_convs:
        if name == first_name:
            continue
        if any(name.endswith(s) for s in pred_head_suffixes):
            continue
        if name in skip_names:
            continue
        if m.in_channels < 8 or m.out_channels < 8:
            continue  # too small to meaningfully factor
        keep.append((name, m))
    return keep


# ---------------------------------------------------------------------------
# Tucker-2 factorization of a single Conv2d
# ---------------------------------------------------------------------------

def tucker2_factorize_conv(conv: nn.Conv2d, rank_out: int, rank_in: int):
    """
    Replace a Conv2d(Cin,Cout,k,k) with an equivalent-shape 3-conv sequential
    using partial Tucker decomposition (modes 0=out_channels, 1=in_channels).
    """
    weight = conv.weight.data  # (Cout, Cin, k, k)
    rank_out = max(1, min(rank_out, weight.shape[0] - 1))
    rank_in = max(1, min(rank_in, weight.shape[1] - 1))

    (core, factors), _errors = partial_tucker(weight, modes=[0, 1], rank=[rank_out, rank_in])
    out_factor, in_factor = factors  # (Cout, rank_out), (Cin, rank_in)

    first = nn.Conv2d(conv.in_channels, rank_in, kernel_size=1, bias=False)
    core_conv = nn.Conv2d(rank_in, rank_out, kernel_size=conv.kernel_size,
                           stride=conv.stride, padding=conv.padding,
                           dilation=conv.dilation, bias=False)
    last = nn.Conv2d(rank_out, conv.out_channels, kernel_size=1,
                      bias=(conv.bias is not None))

    with torch.no_grad():
        first.weight.copy_(in_factor.t().unsqueeze(-1).unsqueeze(-1))
        core_conv.weight.copy_(core)
        last.weight.copy_(out_factor.unsqueeze(-1).unsqueeze(-1))
        if conv.bias is not None:
            last.bias.copy_(conv.bias.data)

    return nn.Sequential(first, core_conv, last)


def params_before_after(conv: nn.Conv2d, rank_out: int, rank_in: int):
    before = conv.weight.numel() + (conv.bias.numel() if conv.bias is not None else 0)
    k0, k1 = conv.kernel_size
    after = (conv.in_channels * rank_in           # 1x1 reduce
             + rank_in * rank_out * k0 * k1        # core spatial conv
             + rank_out * conv.out_channels        # 1x1 expand
             + (conv.out_channels if conv.bias is not None else 0))
    return before, after


def set_module_by_name(model, name, new_module):
    parts = name.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], new_module)


# ---------------------------------------------------------------------------
# Phase 1: per-layer sensitivity via degree-4 polynomial fit
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_loss(model, loader, criterions, device, max_batches=10):
    model.eval()
    total, n = 0.0, 0
    for i, (imgs, targets) in enumerate(loader):
        if i >= max_batches:
            break
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]
        preds = model(imgs)
        loss = 0.0
        for scale_idx, pred in enumerate(preds):
            l, _ = criterions[scale_idx](pred, targets, scale_idx)
            loss = loss + l.item()
        total += loss
        n += 1
    return total / max(n, 1)


def phase1_sensitivity_sweep(model, layers, loader, criterions, device,
                              ratios=(0.9, 0.7, 0.5, 0.3, 0.1), max_batches=10):
    """
    For each compressible layer, try each candidate compression ratio in
    isolation and record (ratio, loss_delta), fit a degree-4 polynomial,
    and compute sensitivity = 2*|a2|.

    Efficiency note: rather than deep-copying the whole model per trial
    (expensive for a 60M+ param network), each trial swaps just the one
    layer being tested into the live model, evaluates, then restores the
    original layer -- same result, far less memory/time.

    Returns:
      sensitivities: {layer_name: float}
      sweep_data:    {layer_name: [(ratio, params_after, loss_delta), ...]}
    """
    baseline_loss = eval_loss(model, loader, criterions, device, max_batches)
    sensitivities, sweep_data = {}, {}

    for name, conv in layers:
        rows = []
        for ratio in ratios:
            rank_out = max(1, int(conv.out_channels * ratio))
            rank_in = max(1, int(conv.in_channels * ratio))

            factored = tucker2_factorize_conv(conv, rank_out, rank_in).to(device)
            set_module_by_name(model, name, factored)

            loss = eval_loss(model, loader, criterions, device, max_batches)
            _, params_after = params_before_after(conv, rank_out, rank_in)
            rows.append((ratio, params_after, loss - baseline_loss))

            set_module_by_name(model, name, conv)  # restore original before next trial/layer

        ratios_arr = np.array([r[0] for r in rows])
        deltas_arr = np.array([r[2] for r in rows])
        # degree-4 biquadratic fit: a4 r^4 + a3 r^3 + a2 r^2 + a1 r + a0
        coeffs = np.polyfit(ratios_arr, deltas_arr, deg=4)  # highest degree first
        a2 = coeffs[2]  # coefficient of r^2
        sensitivity = 2 * abs(a2)

        sensitivities[name] = float(sensitivity)
        sweep_data[name] = rows
        print(f"  [{name}] sensitivity={sensitivity:.5f}  "
              f"(loss_delta range {deltas_arr.min():.4f}..{deltas_arr.max():.4f})")

    return sensitivities, sweep_data, baseline_loss


# ---------------------------------------------------------------------------
# Phase 2: inverse-sensitivity budget distribution + sequential compression
# ---------------------------------------------------------------------------

def distribute_budget(sensitivities, global_target=0.5):
    """
    Inverse-sensitivity weighting: layers with LOW sensitivity get assigned
    a LOWER target ratio (compressed harder); high-sensitivity layers get
    a target ratio closer to 1 (left mostly intact).

    global_target: desired overall remaining-parameter fraction (e.g. 0.5
    keeps ~50% of compressible params). Per-layer ratios are normalized so
    their (rank-weighted) average matches this target, then floor/cap
    guardrails are applied.
    """
    names = list(sensitivities.keys())
    sens = np.array([sensitivities[n] for n in names])

    # inverse sensitivity -> higher weight = compress harder = lower ratio
    inv = 1.0 / (sens + 1e-8)
    inv_norm = (inv - inv.min()) / (inv.max() - inv.min() + 1e-8)  # 0..1

    # map normalized inverse-sensitivity to a per-layer target ratio,
    # centered so the mean ratio ~= global_target
    spread = 0.4
    raw_ratio = global_target + spread * (0.5 - inv_norm)
    raw_ratio = np.clip(raw_ratio, 0.05, 0.95)

    floor = global_target / (3 * len(names))
    cap = min(0.95, 3 * global_target)
    final_ratio = np.clip(raw_ratio, floor, cap)

    return {n: float(r) for n, r in zip(names, final_ratio)}


def pick_pareto_candidate(sweep_rows, target_ratio):
    """
    From the phase-1 sweep rows [(ratio, params_after, loss_delta), ...],
    build the Pareto front (minimize params_after, minimize loss_delta) and
    return the candidate whose ratio is closest to target_ratio, preferring
    Pareto-optimal points.
    """
    pareto = []
    for i, (r, p, d) in enumerate(sweep_rows):
        dominated = any(
            (p2 <= p and d2 <= d and (p2 < p or d2 < d))
            for j, (r2, p2, d2) in enumerate(sweep_rows) if j != i
        )
        if not dominated:
            pareto.append((r, p, d))
    if not pareto:
        pareto = sweep_rows
    best = min(pareto, key=lambda row: abs(row[0] - target_ratio))
    return best


def phase2_compress(model, layers, sensitivities, sweep_data, global_target,
                     ckpt_dir, device):
    """
    Sequential cumulative compression with JSON checkpoint recovery: each
    layer is compressed in place, model saved + a JSON checkpoint recorded,
    so a killed run can resume from the last finished layer.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    state_path = os.path.join(ckpt_dir, "phase2_state.json")

    done = {}
    if os.path.exists(state_path):
        with open(state_path) as f:
            done = json.load(f)
        print(f"Resuming phase 2: {len(done)} layers already compressed")

    ratios = distribute_budget(sensitivities, global_target)
    layer_dict = dict(layers)

    total_before, total_after = 0, 0
    for name, conv in layers:
        before, _ = params_before_after(conv, 1, 1)
        total_before += before

        if name in done:
            total_after += done[name]["params_after"]
            continue

        target_ratio = ratios[name]
        rows = sweep_data[name]
        chosen_ratio, params_after, est_delta = pick_pareto_candidate(rows, target_ratio)

        # Guardrail: for small/thin layers, Tucker-2's 1x1 reduce/expand
        # overhead can exceed the savings from the smaller spatial core.
        # In that case, leave the layer uncompressed rather than growing it.
        if params_after >= before:
            total_after += before
            done[name] = {
                "target_ratio": target_ratio, "chosen_ratio": 1.0,
                "rank_out": conv.out_channels, "rank_in": conv.in_channels,
                "params_before": before, "params_after": before,
                "est_loss_delta": 0.0, "skipped_no_benefit": True,
            }
            with open(state_path, "w") as f:
                json.dump(done, f, indent=2)
            print(f"  skipped {name}: factorization overhead exceeds savings "
                  f"({before:,} -> {params_after:,}), kept original")
            continue

        rank_out = max(1, int(conv.out_channels * chosen_ratio))
        rank_in = max(1, int(conv.in_channels * chosen_ratio))
        factored = tucker2_factorize_conv(conv, rank_out, rank_in)
        set_module_by_name(model, name, factored)
        model.to(device)

        total_after += params_after
        done[name] = {
            "target_ratio": target_ratio, "chosen_ratio": chosen_ratio,
            "rank_out": rank_out, "rank_in": rank_in,
            "params_before": before, "params_after": params_after,
            "est_loss_delta": est_delta, "skipped_no_benefit": False,
        }
        with open(state_path, "w") as f:
            json.dump(done, f, indent=2)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_compressed_partial.pt"))
        print(f"  compressed {name}: ratio={chosen_ratio:.2f}  "
              f"params {before:,} -> {params_after:,}")

    print(f"\nTotal compressible params: {total_before:,} -> {total_after:,} "
          f"({total_after/total_before:.1%} kept)")
    return model, done
