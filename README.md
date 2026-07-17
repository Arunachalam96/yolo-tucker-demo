# YOLOv3 + Tucker Compression — Demo Pipeline

From-scratch PyTorch YOLOv3, trained on a COCO subset, compressed with the
**same Tucker methodology as the validated CNN/ViT pipeline** (VGG/ResNet/
MobileNet), structured for a live demo: the slow stages (training, Phase 1
sweep) are pre-run and stored; the fast stage (Phase 2 selection) runs live.

## Methodology (matches the CNN pipeline)

- **Decomposition**: full Tucker on the (C_out, C_in, kH, kW) weight, full
  spatial ranks, spatial factors folded back into the core via mode_dot,
  deployed as 1x1 -> kxk -> 1x1 conv sequence (TuckerBlock).
- **Phase 1** (per layer, offline): sweep a 2D (r_in, r_out) grid at
  step = 20% of min(C_in, C_out). Per combo, measure noise % (std-based
  activation-error metric via forward hooks) and mAP@0.5 (object-detection
  analog of top-1 accuracy). Fit a symmetrized degree-4 (biquadratic)
  polynomial of mAP vs noise.
- **Phase 2** (per layer, live): invert the polynomial for max tolerable
  noise under a global mAP budget (as a percentage of baseline), Pareto front
  over (compression ratio, mAP), K-means cluster into Conservative / Balanced
  / Aggressive suggestions.
- **BatchNorm recalibration** after compression: forward-only resync of BN
  running stats (no weight/gradient updates).

The only substitution vs. the CNN pipeline is mAP@0.5 in place of top-1
accuracy on the polynomial's y-axis. Noise metric, sweep, fit, Pareto, and
clustering are identical.

## Structure

    src/
      model.py            Darknet-53 backbone + YOLOv3 3-scale head
      loss.py             objectness/box(CIoU)/class loss, anchor matching
      dataset.py          COCO-subset Dataset, letterbox + augmentation
      train_utils.py      training loop + JSON checkpoint crash-recovery
      postprocess.py      box decode, NMS, simple mAP@0.5
      tucker_pipeline.py  TuckerBlock, decomposition, noise metric,
                          biquadratic fit, Phase-2 (invert/Pareto/KMeans),
                          BN recalibration
      tucker_phase1.py    Phase-1 sweep + subsampled mAP evaluator
      analysis_utils.py   param count, model size, latency

    notebooks/
      01_data_loading.ipynb       build the COCO subset, visualize a batch
      02_model_definition.ipynb   build model, shape/param sanity checks
      03_training.ipynb           SLOW  — train the baseline, pre-run
      04_phase1_sweep.ipynb       SLOW  — offline sweep, per-layer JSON ckpts
      05_phase2_selection.ipynb   FAST  — live: select, apply, recalibrate, save
      06_analysis.ipynb           before/after: params, size, latency, mAP

## Demo-day workflow

Pre-run offline (DGX), ahead of the demo:
1. 01, 02, 03 — data, model, train baseline (yolov3_best.pt).
2. 04_phase1_sweep — 2D sweep across all 37 compressible layers. Heavy;
   checkpoints per layer to checkpoints/phase1/ and resumes if interrupted.
   Carry that folder to the demo machine.

Live:
3. 05_phase2_selection — loads stored Phase 1 JSON, runs invert -> Pareto ->
   K-means (near-instant), pick a tier, applies decomposition + BN
   recalibration, saves yolov3_compressed.pt.
4. 06_analysis — before/after (can also be pre-run).

## Setup

    pip install torch tensorly pycocotools opencv-python-headless \
                scipy scikit-learn matplotlib pandas jupyter ipykernel

Point DATA_ROOT in 01_data_loading.ipynb at a COCO 2017 layout (annotations
+ train2017/ + val2017/). CLASS_NAMES picks a small subset.

## Notes / things to tune

- 37 compressible layers (of 75 Conv2d): all 3x3 convs except the stem; 1x1
  convs and the 3 prediction heads are skipped. These 37 hold ~91% of the
  network's parameters. 28 are in the Darknet-53 backbone, 9 in neck/heads.
- 1x1 convs are skipped by the kernel>1x1 rule (matching the CNN pipeline).
  Tucker on a 1x1 conv reduces to truncated SVD — the same path as the
  ViT/LLM Linear-layer work, if you ever want to include them.
- DataLoader num_workers=0 everywhere, to avoid Docker shared-memory bus
  errors. Raise only if the container is started with --shm-size=8g.
- Budget is a percentage of baseline mAP (e.g. 5.0 = allow the fitted curve
  to drop to 95% of baseline). Baseline measured once up front.
- Phase-1 MAP_MAX_BATCHES controls the mAP-eval subsample size — larger =
  more trustworthy curves (fine, since Phase 1 is offline).
- Latency vs param count: Tucker adds sequential 1x1/kxk/1x1 ops, so
  wall-clock speedup doesn't track parameter reduction — the win is model
  size / memory. 06_analysis reports both.

## Extending toward LLMs

tucker_pipeline.py is architecture-agnostic on the decomposition side. For
LLMs, swap Conv2d enumeration for Linear, use the SVD path (a Linear / 1x1
layer has no spatial mode to fold), and use perplexity as the smooth Phase-1
quality metric in place of mAP — same symmetrized-fit + invert + Pareto
machinery.
