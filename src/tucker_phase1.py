"""
tucker_phase1.py
----------------
Phase 1 sweep for the YOLOv3 Tucker pipeline -- the offline, pre-run,
checkpointed stage. Mirrors the CNN pipeline's sweep_layer, with mAP@0.5
(subsampled) as the quality metric in place of top-1 accuracy.
"""
import gc
import json
import os
import time
import numpy as np
import torch

from postprocess import predict_boxes, simple_map50
from tucker_pipeline import (
    tucker_decompose_layer, build_decomposed_model, get_module_by_name,
    compute_compression_ratio, compute_noise_percent, fit_biquadratic_polynomial,
)


@torch.no_grad()
def evaluate_map(model, loader, num_classes, device, img_size, max_batches=None,
                 conf_thresh=0.25):
    """
    mAP@0.5 on (a subset of) loader. `max_batches` caps the number of
    batches for Phase-1 tractability -- this is the YOLO analog of the CNN
    pipeline's full-val accuracy, subsampled because detection mAP is far
    heavier than top-1. Set max_batches=None for the full loader.
    """
    model.eval().to(device)
    all_preds, all_targets = [], []
    for i, (imgs, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        imgs = imgs.to(device)
        preds = predict_boxes(model, imgs, num_classes, conf_thresh=conf_thresh)
        all_preds.extend([(p[0].cpu(), p[1].cpu(), p[2].cpu()) for p in preds])
        for t in targets:
            if t.numel() == 0:
                all_targets.append((torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)))
                continue
            cx, cy = t[:, 1] * img_size, t[:, 2] * img_size
            w, h = t[:, 3] * img_size, t[:, 4] * img_size
            boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
            all_targets.append((boxes, t[:, 0].long()))
    return simple_map50(all_preds, all_targets, num_classes)


def sweep_layer(orig_model, layer_info, X_sample, val_loader, num_classes,
                device, img_size, map_max_batches=20, noise_batch_size=8):
    """
    Sweep all (r_in, r_out) combos for one layer. Direct structural port of
    the CNN pipeline's sweep_layer:
      - orig_model is NEVER modified (deep-copy per combo)
      - step = 20% of min(C_in, C_out), range step..full inclusive
      - per combo: TuckerBlock -> deepcopy+replace -> mAP (subsampled) ->
        noise % (hooks on X_sample) -> compression ratio -> cleanup
    """
    name = layer_info["name"]
    C_in, C_out = layer_info["C_in"], layer_info["C_out"]

    step = max(1, int(round(0.20 * min(C_in, C_out))))
    r_in_values = list(range(step, C_in + 1, step))
    r_out_values = list(range(step, C_out + 1, step))
    if r_in_values[-1] != C_in:
        r_in_values.append(C_in)
    if r_out_values[-1] != C_out:
        r_out_values.append(C_out)

    total = len(r_in_values) * len(r_out_values)
    results = []
    combo_idx = 0
    print(f"\n  Layer: {name}  C_in={C_in} C_out={C_out} step={step} combos={total}")

    for r_in in r_in_values:
        for r_out in r_out_values:
            combo_idx += 1
            t0 = time.time()
            tmp_model = None
            try:
                orig_layer = get_module_by_name(orig_model, name)
                tucker_block = tucker_decompose_layer(orig_layer, r_in, r_out)
                tmp_model = build_decomposed_model(orig_model, name, tucker_block)

                mAP = evaluate_map(tmp_model, val_loader, num_classes, device,
                                    img_size, max_batches=map_max_batches)
                noise = compute_noise_percent(orig_model, tmp_model, name,
                                               X_sample, device, batch_size=noise_batch_size)
                cr = compute_compression_ratio(orig_layer, r_in, r_out)

                results.append({
                    "r_in": int(r_in), "r_out": int(r_out),
                    "noise_percent": float(noise), "mAP": float(mAP),
                    "compression_ratio": float(cr),
                })
                print(f"    [{combo_idx:3d}/{total}] r_in={r_in:4d} r_out={r_out:4d} "
                      f"| noise={noise:6.2f}% mAP={mAP:.4f} CR={cr:.2f}x "
                      f"({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [{combo_idx:3d}/{total}] r_in={r_in} r_out={r_out} FAILED: {e}")
            finally:
                if tmp_model is not None:
                    tmp_model.cpu()
                    del tmp_model
                if device == "cuda" or (hasattr(device, "type") and device.type == "cuda"):
                    torch.cuda.empty_cache()
                gc.collect()
    return results


def run_phase1(orig_model, compressible, X_sample, val_loader, num_classes,
               device, img_size, ckpt_dir, map_max_batches=20, noise_batch_size=8):
    """
    Full Phase 1 over all compressible layers, with a per-layer JSON
    checkpoint after each (crash recovery + demo-day pre-storage). Skips
    layers whose checkpoint already exists, so an interrupted run resumes.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    layer_results = []
    t_start = time.time()

    for i, layer_info in enumerate(compressible):
        name = layer_info["name"]
        ckpt_path = os.path.join(ckpt_dir, f"phase1_layer{i:03d}.json")

        if os.path.exists(ckpt_path):
            with open(ckpt_path) as f:
                layer_results.append(json.load(f))
            print(f"[Phase1] layer {i+1}/{len(compressible)} {name}: loaded from checkpoint")
            continue

        print(f"\n[Phase1] layer {i+1}/{len(compressible)}: {name}")
        sweep = sweep_layer(orig_model, layer_info, X_sample, val_loader,
                            num_classes, device, img_size,
                            map_max_batches=map_max_batches,
                            noise_batch_size=noise_batch_size)

        valid = [r for r in sweep if not np.isnan(r["noise_percent"])]
        poly_coeffs = None
        if len(valid) >= 3:
            poly_coeffs = fit_biquadratic_polynomial(
                [r["noise_percent"] for r in valid], [r["mAP"] for r in valid])
            print(f"  poly [c4..c0]: {[f'{c:.3e}' for c in poly_coeffs]}")
        else:
            print("  not enough valid points to fit polynomial")

        entry = {**layer_info, "sweep_results": sweep, "poly_coeffs": poly_coeffs}
        layer_results.append(entry)
        with open(ckpt_path, "w") as f:
            json.dump(entry, f, indent=2)
        print(f"  [checkpoint] {ckpt_path}")

    print(f"\n[Phase1] complete in {(time.time()-t_start)/60:.1f} min")
    return layer_results
