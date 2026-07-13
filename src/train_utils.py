"""
Training loop + JSON checkpoint bookkeeping, deliberately mirroring the
crash-recovery pattern used in the Tucker-2 sweep pipeline: a small JSON
sidecar records epoch/step/best-loss so a killed run can resume without
losing progress on the DGX/shared-GPU setting.
"""
import json
import os
import time
import torch

from loss import YOLOLoss


def save_checkpoint(model, optimizer, epoch, step, best_loss, ckpt_dir, tag="last"):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }, os.path.join(ckpt_dir, f"yolov3_{tag}.pt"))

    meta = {"epoch": epoch, "step": step, "best_loss": best_loss, "timestamp": time.time()}
    with open(os.path.join(ckpt_dir, f"yolov3_{tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint_meta(ckpt_dir, tag="last"):
    meta_path = os.path.join(ckpt_dir, f"yolov3_{tag}_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def train_one_epoch(model, loader, optimizer, criterions, device, log_every=20):
    model.train()
    running = 0.0
    for i, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]

        preds = model(imgs)  # (out_large, out_medium, out_small)
        loss = 0.0
        for scale_idx, pred in enumerate(preds):
            l, _ = criterions[scale_idx](pred, targets, scale_idx)
            loss = loss + l

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()
        if i % log_every == 0:
            print(f"  step {i}/{len(loader)}  loss {loss.item():.4f}")
    return running / max(len(loader), 1)


def fit(model, loader, device, epochs, lr, ckpt_dir, num_classes, resume=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterions = [YOLOLoss(num_classes) for _ in range(3)]

    start_epoch = 0
    best_loss = float("inf")
    if resume:
        meta = load_checkpoint_meta(ckpt_dir, "last")
        ckpt_path = os.path.join(ckpt_dir, "yolov3_last.pt")
        if meta is not None and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            start_epoch = meta["epoch"] + 1
            best_loss = meta["best_loss"]
            print(f"Resumed from epoch {start_epoch} (best_loss={best_loss:.4f})")

    history = []
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        avg_loss = train_one_epoch(model, loader, optimizer, criterions, device)
        dt = time.time() - t0
        print(f"epoch {epoch}: avg_loss={avg_loss:.4f}  ({dt:.1f}s)")
        history.append({"epoch": epoch, "loss": avg_loss, "seconds": dt})

        save_checkpoint(model, optimizer, epoch, len(loader), min(best_loss, avg_loss), ckpt_dir, tag="last")
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, len(loader), best_loss, ckpt_dir, tag="best")

    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history
