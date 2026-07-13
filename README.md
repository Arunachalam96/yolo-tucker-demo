# YOLOv3 + Tucker-2 Compression — Demo Pipeline

From-scratch PyTorch YOLOv3, trained on a COCO subset, compressed with your
Phase 1 / Phase 2 Tucker-2 methodology, all in notebooks structured for a
live demo: the slow part (training) is pre-run beforehand, the fast part
(decomposition) runs live in a few minutes.

## Structure

```
src/                        shared modules, imported by all notebooks
  model.py                  Darknet-53 backbone + YOLOv3 3-scale head
  loss.py                   objectness/box(CIoU)/class loss, anchor matching
  dataset.py                COCO-subset Dataset, letterbox + augmentation
  train_utils.py            training loop + JSON checkpoint crash-recovery
  tucker_decompose.py       Phase 1 (sensitivity) + Phase 2 (budget/compress)
  postprocess.py            box decode, NMS, simple mAP@0.5
  analysis_utils.py         param count, model size, latency

notebooks/
  01_data_loading.ipynb        build the COCO subset, visualize a batch
  02_model_definition.ipynb    build model, shape/param sanity checks
  03_training.ipynb            ⏱️ SLOW — run ahead of the demo
  04_decomposition.ipynb       ⚡ FAST — run live during the demo
  05_analysis.ipynb            before/after: params, size, latency, mAP
```

## Demo-day workflow

1. **Before the demo:** run `01`, `02`, `03` in full (03 is the long one —
   leave it running, it checkpoints every epoch and resumes if interrupted).
   This produces `checkpoints/yolov3_best.pt`.
2. **Live, in front of the audience:** run `04_decomposition.ipynb`. Phase 1's
   sensitivity sweep is scoped down (`MAX_BATCHES`, 5 ratios/layer) to finish
   in a couple of minutes; Phase 2 is near-instant. Everyone watches the
   actual compression happen.
3. **Also live (or pre-run, your call):** `05_analysis.ipynb` — params/size/
   latency/mAP comparison and the per-layer compression bar chart.

## Setup

```
pip install torch tensorly pycocotools opencv-python-headless matplotlib pandas jupyter
```

Point `DATA_ROOT` in `01_data_loading.ipynb` at a COCO 2017 layout:
```
DATA_ROOT/annotations/instances_train2017.json
DATA_ROOT/annotations/instances_val2017.json
DATA_ROOT/train2017/*.jpg
DATA_ROOT/val2017/*.jpg
```
Download from https://cocodataset.org/#download. `CLASS_NAMES` in the same
notebook picks a small subset (default: person/car/dog/chair/bottle) so
training finishes in a reasonable time on one GPU — bump `images_per_class`
or add classes once you've confirmed timing on your machine.

## Design notes / things you'll likely want to tune

- **Compressible-layer selection** (`get_compressible_conv_layers`) skips the
  stem conv and the three detection-head prediction convs, matching your
  existing "skip first conv / skip classification head" convention.
- **Sensitivity metric** is the same `2*|a2|` biquadratic-curvature approach
  as the CNN/ViT pipeline, computed per-layer via an in-place swap-evaluate-
  restore (no full-model deepcopy — this is what keeps Phase 1 fast even
  though YOLOv3 has ~70 compressible conv layers).
- **Budget distribution** in Phase 2 mirrors your normalization + floor
  (`global_budget/(3*n_layers)`) + cap (`3x global_target`) guardrails, plus
  one addition worth flagging: a **no-benefit guardrail** that skips
  compressing very thin layers where the 1x1-reduce/expand overhead would
  make the factored layer *larger* than the original (this showed up during
  testing — small early Darknet layers are the main ones affected).
- **Crash recovery**: both `03_training` (epoch-level) and `04_decomposition`
  (layer-level, via `phase2_state.json`) resume automatically if interrupted.
- **mAP@0.5** in `05_analysis` is a lightweight from-scratch implementation
  (not COCOeval) — enough for a relative before/after comparison, not a
  paper-grade benchmark. Swap in `pycocotools.cocoeval` if you need the
  latter.
- **Latency vs. param-count**: Tucker-2's extra sequential 1x1/kxk/1x1 ops
  mean wall-clock speedup doesn't always track parameter reduction 1:1,
  especially on GPU. `05_analysis` reports both so you're not caught off
  guard live if someone asks why a 50%-smaller model isn't 2x faster.

## Extending toward your LLM work

`tucker_decompose.py` is architecture-agnostic — it factors any `Conv2d`,
which per your Tucker-2=SVD-on-2D-weights insight is directly the LLM
extension for attention/FFN Linear layers, just swap `Conv2d` enumeration
for `Linear` and use `perplexity` as the phase-1 evaluation signal in place
of `eval_loss`.
