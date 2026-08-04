"""
viz_utils.py
------------
Architecture visualization helpers for the demo: torchview inline graphs,
ONNX export, and Netron (inline-iframe or static-image) display -- for showing
the model before vs after Tucker decomposition, both whole-model and the
single-layer conv -> TuckerBlock transformation.
"""
import os
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# torchview -- pure-Python inline graphs (renders in the notebook cell)
# ---------------------------------------------------------------------------

def torchview_graph(model, input_size=(1, 3, 416, 416), depth=2, device="cpu",
                     graph_name="model"):
    """
    Return a torchview graph object that renders inline in Jupyter.
    `depth` controls how deep the module nesting is expanded (2-3 is readable
    for the whole model; higher shows more internal detail).
    Usage in a cell:  torchview_graph(model).visual_graph
    """
    from torchview import draw_graph
    g = draw_graph(model, input_size=input_size, depth=depth, device=device,
                   graph_name=graph_name, expand_nested=True)
    return g


def single_layer_before_after(model, layer_name, tucker_block, device="cpu"):
    """
    Build two tiny standalone modules -- the original single conv and its
    TuckerBlock replacement -- so torchview can draw the conv -> (1x1,kxk,1x1)
    transformation clearly, side by side, without the whole network's clutter.
    Returns (original_conv_module, tucker_block_module) ready to graph.
    """
    from tucker_pipeline import get_module_by_name
    orig_conv = get_module_by_name(model, layer_name)

    # wrap the single conv so torchview has a forward to trace
    class OneConv(nn.Module):
        def __init__(self, conv):
            super().__init__()
            self.conv = conv
        def forward(self, x):
            return self.conv(x)

    class OneBlock(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block
        def forward(self, x):
            return self.block(x)

    return OneConv(orig_conv), OneBlock(tucker_block)


# ---------------------------------------------------------------------------
# ONNX export -- required for any Netron path
# ---------------------------------------------------------------------------

def export_onnx(model, path, input_size=(1, 3, 416, 416), device="cpu"):
    """Export a model to ONNX for Netron viewing."""
    model = model.eval().to(device)
    dummy = torch.randn(*input_size, device=device)
    kwargs = dict(
        input_names=["input"],
        output_names=["out_large", "out_medium", "out_small"],
        opset_version=12, do_constant_folding=True,
    )
    # Newer torch defaults to the dynamo exporter (needs onnxscript). Prefer the
    # legacy TorchScript exporter (dynamo=False) which has no extra deps; fall
    # back gracefully if the arg isn't supported on older torch.
    try:
        torch.onnx.export(model, dummy, path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, path, **kwargs)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"exported {path}  ({size_mb:.1f} MB)")
    return path


# ---------------------------------------------------------------------------
# Netron -- inline iframe (interactive) OR static image (zero-risk for demo)
# ---------------------------------------------------------------------------

def netron_inline(onnx_path, port=8080, height=600):
    """
    Launch Netron's server (no auto-browser) and embed it inline via IFrame.
    Interactive, but needs `port` reachable -- on the DGX over VS Code Remote-SSH
    the port must be forwarded (VS Code often does this automatically). If the
    iframe is blank, the port isn't forwarded -- use netron_image() instead.
    """
    import netron
    from IPython.display import IFrame
    netron.start(onnx_path, address=("0.0.0.0", port), browse=False)
    return IFrame(src=f"http://localhost:{port}", width="100%", height=height)


def netron_image(image_path, width=900):
    """
    Display a PNG/SVG previously exported from Netron (File -> Export in the
    Netron app, or the download button on netron.app). Zero runtime risk --
    the safest option for a live demo. Pre-export the images beforehand.
    """
    from IPython.display import Image
    return Image(filename=image_path, width=width)
