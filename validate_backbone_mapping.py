"""
validate_backbone_mapping.py
----------------------------
Checks whether timm's pretrained darknet53 can be mapped INTO our from-scratch
Darknet53 backbone by ORDER (walk both models' conv+BN layers in sequence and
compare shapes position-by-position). Copies NOTHING -- purely a compatibility
report, so we know before building the real loader whether the structures align.

Run on the DGX (needs network access to pull timm weights + our src/ on path):
    python validate_backbone_mapping.py

Reads: src/model.py (our Darknet53)
Needs: pip install timm   (already used for the DeiT/ViT work)
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append("src")

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    print("ERROR: timm not installed. Run: pip install timm")
    sys.exit(1)

from model import Darknet53


def conv_bn_sequence(module):
    """
    Return the ordered list of (name, layer) for every Conv2d and BatchNorm2d
    in `module`, in forward/definition order (named_modules preserves this).
    """
    seq = []
    for name, m in module.named_modules():
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
            seq.append((name, m))
    return seq


def describe(layer):
    if isinstance(layer, nn.Conv2d):
        return f"Conv2d{tuple(layer.weight.shape)}"
    if isinstance(layer, nn.BatchNorm2d):
        return f"BN({layer.num_features})"
    return type(layer).__name__


def main():
    print("=" * 78)
    print("  Backbone mapping validation: timm darknet53  ->  our Darknet53")
    print("=" * 78)

    # --- our backbone ---
    ours = Darknet53()
    ours_seq = conv_bn_sequence(ours)
    ours_convs = [x for x in ours_seq if isinstance(x[1], nn.Conv2d)]
    ours_bns = [x for x in ours_seq if isinstance(x[1], nn.BatchNorm2d)]
    print(f"\n[ours]  conv layers: {len(ours_convs)}   BN layers: {len(ours_bns)}")

    # --- timm backbone ---
    # features_only=False gives the full classification model; we only compare
    # the convolutional trunk. Try the specific ImageNet-1k weights first.
    model_name = "darknet53.c2ns_in1k"
    print(f"[timm]  creating '{model_name}' (pretrained=True) ...")
    try:
        timm_model = timm.create_model(model_name, pretrained=True)
    except Exception as e:
        print(f"  could not load '{model_name}' ({e})")
        print("  retrying with plain 'darknet53' ...")
        timm_model = timm.create_model("darknet53", pretrained=True)

    timm_seq = conv_bn_sequence(timm_model)
    timm_convs = [x for x in timm_seq if isinstance(x[1], nn.Conv2d)]
    timm_bns = [x for x in timm_seq if isinstance(x[1], nn.BatchNorm2d)]
    print(f"[timm]  conv layers: {len(timm_convs)}   BN layers: {len(timm_bns)}")
    print(f"[timm]  (note: timm's model includes a classifier head; the trunk "
          f"should hold the 52 backbone convs)\n")

    # --- compare conv sequences by order ---
    print("-" * 78)
    print("  Conv-by-conv shape alignment (first mismatch stops the report)")
    print("-" * 78)

    n = min(len(ours_convs), len(timm_convs))
    mismatches = 0
    first_mismatch = None
    aligned = 0

    # timm may have leading/trailing convs that aren't part of our trunk; we
    # attempt a direct positional alignment on the first len(ours_convs) convs
    # of timm and report how it lines up.
    for i in range(len(ours_convs)):
        o_name, o = ours_convs[i]
        if i >= len(timm_convs):
            print(f"  [{i:2d}] ours {describe(o):20s}  | timm: (no layer at this index)")
            mismatches += 1
            continue
        t_name, t = timm_convs[i]
        match = tuple(o.weight.shape) == tuple(t.weight.shape)
        if match:
            aligned += 1
        else:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = i
        flag = "OK " if match else "XX "
        if (not match) or i < 6 or i >= len(ours_convs) - 3:
            print(f"  {flag}[{i:2d}] ours {o_name:24s} {describe(o):20s} "
                  f"| timm {t_name:28s} {describe(t)}")

    print("-" * 78)
    print(f"  aligned conv layers: {aligned}/{len(ours_convs)}")
    print(f"  mismatched:          {mismatches}")
    if first_mismatch is not None:
        print(f"  first mismatch at conv index: {first_mismatch}")

    # --- verdict ---
    print("\n" + "=" * 78)
    if aligned == len(ours_convs) and len(ours_convs) == len(timm_convs):
        print("  VERDICT: FULL ALIGNMENT — by-order mapping is safe. "
              "We can build the loader.")
    elif aligned == len(ours_convs):
        print("  VERDICT: our 52 convs all align to timm's first 52 convs, but "
              "timm has extra trailing convs (likely a different head/stem "
              "detail). By-order mapping of the backbone is still safe; we just "
              "align the first 52.")
    else:
        print("  VERDICT: PARTIAL / NO ALIGNMENT — structures differ. By-order "
              "mapping is NOT safe as-is; share this output and we'll adjust "
              "(offset the alignment, or switch source).")
    print("=" * 78)


if __name__ == "__main__":
    main()



