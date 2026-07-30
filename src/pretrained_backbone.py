"""
pretrained_backbone.py
----------------------
Loads ImageNet-pretrained Darknet-53 weights from `timm` into our from-scratch
Darknet53 backbone. The two implementations are structurally identical (52
Conv2d + 52 BatchNorm2d, same shapes, same order -- verified), just named
differently, so we map BY ORDER: walk both models' conv/BN layers in sequence
and copy weights position-by-position, with a shape guard that refuses to copy
if anything ever fails to line up.

Usage (in the training notebook, before training):
    from model import YOLOv3
    from pretrained_backbone import load_pretrained_backbone
    model = YOLOv3(num_classes=NUM_CLASSES)
    load_pretrained_backbone(model)   # fills model.backbone from timm; heads stay random

Only the backbone is initialized -- the detection neck/heads train from scratch
(their weights have no ImageNet counterpart). Then train everything together.
"""
import torch
import torch.nn as nn


def _conv_bn_in_order(module):
    """Ordered list of Conv2d and BatchNorm2d layers (definition/forward order)."""
    convs, bns = [], []
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            convs.append(m)
        elif isinstance(m, nn.BatchNorm2d):
            bns.append(m)
    return convs, bns


def load_pretrained_backbone(model, timm_name="darknet53.c2ns_in1k", verbose=True):
    """
    Copy timm's pretrained Darknet-53 conv+BN weights into model.backbone.

    Returns the number of (conv, bn) layers successfully copied. Raises a
    RuntimeError if the structures don't align (shape mismatch at any position),
    rather than silently corrupting weights.
    """
    try:
        import timm
    except ImportError as e:
        raise ImportError("timm is required: pip install timm") from e

    if verbose:
        print(f"[backbone] loading pretrained '{timm_name}' from timm ...")
    try:
        src = timm.create_model(timm_name, pretrained=True)
    except Exception as e:
        if verbose:
            print(f"[backbone] '{timm_name}' failed ({e}); trying plain 'darknet53'")
        src = timm.create_model("darknet53", pretrained=True)

    # our backbone
    dst_convs, dst_bns = _conv_bn_in_order(model.backbone)
    # timm's trunk -- take its first len(dst_convs) conv/bn layers in order
    src_convs, src_bns = _conv_bn_in_order(src)

    if len(src_convs) < len(dst_convs) or len(src_bns) < len(dst_bns):
        raise RuntimeError(
            f"timm model has fewer layers than our backbone "
            f"(timm convs={len(src_convs)}, ours={len(dst_convs)})")

    # --- shape-guarded by-order copy ---
    with torch.no_grad():
        copied_conv = 0
        for i, (d, s) in enumerate(zip(dst_convs, src_convs)):
            if tuple(d.weight.shape) != tuple(s.weight.shape):
                raise RuntimeError(
                    f"conv shape mismatch at index {i}: "
                    f"ours {tuple(d.weight.shape)} vs timm {tuple(s.weight.shape)}")
            d.weight.copy_(s.weight)
            if d.bias is not None and s.bias is not None:
                d.bias.copy_(s.bias)
            copied_conv += 1

        copied_bn = 0
        for i, (d, s) in enumerate(zip(dst_bns, src_bns)):
            if d.num_features != s.num_features:
                raise RuntimeError(
                    f"BN mismatch at index {i}: ours {d.num_features} vs timm {s.num_features}")
            d.weight.copy_(s.weight)
            d.bias.copy_(s.bias)
            d.running_mean.copy_(s.running_mean)
            d.running_var.copy_(s.running_var)
            if s.num_batches_tracked is not None:
                d.num_batches_tracked.copy_(s.num_batches_tracked)
            copied_bn += 1

    if verbose:
        print(f"[backbone] copied {copied_conv} conv + {copied_bn} BN layers "
              f"into model.backbone (detection neck/heads remain randomly initialized)")
    return copied_conv, copied_bn
