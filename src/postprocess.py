"""
Decode raw network outputs -> boxes, run NMS, and compute a simple
mAP@0.5 -- enough to quantify the accuracy gap between the original and
Tucker-2-compressed model without depending on the full COCOeval machinery
(kept dependency-light and easy to read during the demo).
"""
import torch
from loss import ANCHORS, STRIDES


@torch.no_grad()
def decode_scale(pred, scale_idx, num_classes, conf_thresh=0.25):
    device = pred.device
    B, _, H, W = pred.shape
    A = 3
    stride = STRIDES[scale_idx]
    anchors = torch.tensor(ANCHORS[scale_idx], device=device, dtype=torch.float32)

    pred = pred.view(B, A, 5 + num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")

    xy = (pred[..., 0:2].sigmoid() + torch.stack([grid_x, grid_y], -1)) * stride
    wh = pred[..., 2:4].exp() * anchors.view(1, A, 1, 1, 2)
    obj = pred[..., 4].sigmoid()
    cls = pred[..., 5:].sigmoid()

    boxes_xyxy = torch.cat([xy - wh / 2, xy + wh / 2], dim=-1)
    scores, labels = (obj.unsqueeze(-1) * cls).max(dim=-1)

    keep = scores > conf_thresh
    results = []
    for b in range(B):
        results.append((boxes_xyxy[b][keep[b]], scores[b][keep[b]], labels[b][keep[b]]))
    return results


def nms(boxes, scores, iou_thresh=0.45):
    """Pure-PyTorch NMS (no torchvision dependency needed for the demo)."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    x1, y1, x2, y2 = boxes.unbind(-1)
    areas = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.max(x1[i], x1[rest]); yy1 = torch.max(y1[i], y1[rest])
        xx2 = torch.min(x2[i], x2[rest]); yy2 = torch.min(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thresh]
    return torch.tensor(keep, dtype=torch.long)


@torch.no_grad()
def predict_boxes(model, imgs, num_classes, conf_thresh=0.25, iou_thresh=0.45):
    """Full decode across 3 scales + NMS, per image in the batch."""
    outs = model(imgs)
    per_scale = [decode_scale(outs[i], i, num_classes, conf_thresh) for i in range(3)]
    B = imgs.shape[0]
    final = []
    for b in range(B):
        boxes = torch.cat([per_scale[s][b][0] for s in range(3)], dim=0)
        scores = torch.cat([per_scale[s][b][1] for s in range(3)], dim=0)
        labels = torch.cat([per_scale[s][b][2] for s in range(3)], dim=0)
        keep_idx = nms(boxes, scores, iou_thresh)
        final.append((boxes[keep_idx], scores[keep_idx], labels[keep_idx]))
    return final


def box_iou_xyxy(a, b):
    x1 = torch.max(a[0], b[0]); y1 = torch.max(a[1], b[1])
    x2 = torch.min(a[2], b[2]); y2 = torch.min(a[3], b[3])
    inter = max(0.0, (x2 - x1).item()) * max(0.0, (y2 - y1).item())
    area_a = (a[2] - a[0]).item() * (a[3] - a[1]).item()
    area_b = (b[2] - b[0]).item() * (b[3] - b[1]).item()
    return inter / (area_a + area_b - inter + 1e-9)


def simple_map50(all_preds, all_targets, num_classes):
    """
    all_preds:   list per image of (boxes_xyxy_pixels, scores, labels)
    all_targets: list per image of (boxes_xyxy_pixels, labels)  [ground truth]
    Returns overall mAP@0.5 (unweighted mean AP across classes present in GT).
    Simplified single-IoU-threshold AP (11-point interpolation-free, using
    precision-recall step integration) -- adequate for a before/after
    compression comparison, not a COCOeval replacement.
    """
    aps = []
    for c in range(num_classes):
        tp, fp, n_gt = [], [], 0
        # collect all detections of this class across images, with a running
        # image index so we can match against that image's GT only
        dets = []
        gts_by_img = {}
        for img_idx, (pred, target) in enumerate(zip(all_preds, all_targets)):
            boxes, scores, labels = pred
            gt_boxes, gt_labels = target
            gt_mask = gt_labels == c
            gts_by_img[img_idx] = [gt_boxes[i] for i in range(len(gt_labels)) if gt_mask[i]]
            n_gt += int(gt_mask.sum())
            cls_mask = labels == c
            for i in range(len(labels)):
                if cls_mask[i]:
                    dets.append((img_idx, boxes[i], scores[i].item()))

        if n_gt == 0:
            continue
        dets.sort(key=lambda d: -d[2])
        matched = {k: [False] * len(v) for k, v in gts_by_img.items()}

        for img_idx, box, score in dets:
            gts = gts_by_img[img_idx]
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gts):
                if matched[img_idx][j]:
                    continue
                iou = box_iou_xyxy(box, gt)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.5:
                matched[img_idx][best_j] = True
                tp.append(1); fp.append(0)
            else:
                tp.append(0); fp.append(1)

        if not tp:
            aps.append(0.0)
            continue
        tp_c = torch.tensor(tp).cumsum(0).float()
        fp_c = torch.tensor(fp).cumsum(0).float()
        recall = tp_c / n_gt
        precision = tp_c / (tp_c + fp_c + 1e-9)

        # area under PR curve via trapezoid on sorted-by-recall points
        ap = 0.0
        prev_r = 0.0
        for r, p in zip(recall.tolist(), precision.tolist()):
            ap += (r - prev_r) * p
            prev_r = r
        aps.append(ap)

    return sum(aps) / len(aps) if aps else 0.0
