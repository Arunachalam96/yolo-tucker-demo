"""
YOLOv3 loss: objectness BCE + box regression (CIoU) + class BCE,
computed independently at each of the 3 output scales and summed.
"""
import torch
import torch.nn as nn

# Classic YOLOv3 anchors (COCO), grouped large->small scale, pixels @ 416 input.
ANCHORS = [
    [(116, 90), (156, 198), (373, 326)],   # stride 32 (large objects)
    [(30, 61), (62, 45), (59, 119)],       # stride 16 (medium)
    [(10, 13), (16, 30), (33, 23)],        # stride 8  (small)
]
STRIDES = [32, 16, 8]


def bbox_ciou(box1, box2, eps=1e-7):
    """box: (..., 4) in xywh, both same shape, elementwise CIoU."""
    b1x1, b1y1 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2
    b1x2, b1y2 = box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2
    b2x1, b2y1 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2
    b2x2, b2y2 = box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2

    inter_w = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0)
    inter_h = (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)
    inter = inter_w * inter_h
    area1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    area2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    union = area1 + area2 - inter + eps
    iou = inter / union

    cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
    ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = (box1[..., 0] - box2[..., 0]) ** 2 + (box1[..., 1] - box2[..., 1]) ** 2

    v = (4 / (torch.pi ** 2)) * torch.pow(
        torch.atan(box2[..., 2] / (box2[..., 3] + eps)) - torch.atan(box1[..., 2] / (box1[..., 3] + eps)), 2
    )
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + alpha * v)


class YOLOLoss(nn.Module):
    """
    Builds targets on the fly (grid assignment by best-matching anchor per
    scale) and computes the combined loss for one of the 3 scales at a time.
    Called once per scale in the training loop, losses summed.
    """
    def __init__(self, num_classes, img_size=416, lambda_box=5.0, lambda_obj=1.0,
                 lambda_noobj=0.5, lambda_cls=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls

    def forward(self, pred, targets, scale_idx):
        """
        pred: (B, A*(5+C), H, W) raw network output for this scale
        targets: list length B, each (N_i, 5) = [cls, x, y, w, h] normalized 0..1
        scale_idx: 0=large(stride32) 1=medium(stride16) 2=small(stride8)
        """
        device = pred.device
        B, _, H, W = pred.shape
        A = 3
        C = self.num_classes
        stride = STRIDES[scale_idx]
        anchors = torch.tensor(ANCHORS[scale_idx], device=device, dtype=torch.float32) / stride  # in grid units

        pred = pred.view(B, A, 5 + C, H, W).permute(0, 1, 3, 4, 2).contiguous()  # B,A,H,W,5+C

        obj_mask = torch.zeros(B, A, H, W, device=device)
        noobj_mask = torch.ones(B, A, H, W, device=device)
        tbox = torch.zeros(B, A, H, W, 4, device=device)
        tcls = torch.zeros(B, A, H, W, C, device=device)

        for b in range(B):
            t = targets[b]
            if t.numel() == 0:
                continue
            gxy = t[:, 1:3] * torch.tensor([W, H], device=device)
            gwh = t[:, 3:5] * torch.tensor([W, H], device=device)
            gi, gj = gxy.long()[:, 0].clamp(0, W - 1), gxy.long()[:, 1].clamp(0, H - 1)

            # pick best anchor per target by IoU of widths/heights only
            inter = torch.min(gwh[:, None, :], anchors[None, :, :]).prod(-1)
            union = gwh[:, None].prod(-1) + anchors[None, :].prod(-1) - inter
            iou_wh = inter / union
            best_a = iou_wh.argmax(dim=1)

            for k in range(t.shape[0]):
                a, i, j = best_a[k], gi[k], gj[k]
                cls = t[k, 0].long().clamp(0, C - 1)
                obj_mask[b, a, j, i] = 1.0
                noobj_mask[b, a, j, i] = 0.0
                tbox[b, a, j, i, 0] = gxy[k, 0] - i
                tbox[b, a, j, i, 1] = gxy[k, 1] - j
                tbox[b, a, j, i, 2] = gwh[k, 0]
                tbox[b, a, j, i, 3] = gwh[k, 1]
                tcls[b, a, j, i, cls] = 1.0

        pred_xy = pred[..., 0:2].sigmoid()
        pred_wh = pred[..., 2:4]
        anchors_r = anchors.view(1, A, 1, 1, 2)
        pred_box_abs = torch.cat([pred_xy, pred_wh.exp() * anchors_r], dim=-1)
        tbox_abs = tbox.clone()
        tbox_abs[..., 2:4] = tbox[..., 2:4]  # already in grid units, matches anchors_r scale
        # note: gwh above is already grid-unit width/height (no extra anchor mult needed
        # since ciou compares absolute grid-space boxes on both sides)

        ciou = bbox_ciou(pred_box_abs, tbox_abs)
        box_loss = ((1 - ciou) * obj_mask).sum() / obj_mask.sum().clamp(min=1)

        obj_loss = (self.bce(pred[..., 4], obj_mask) * obj_mask).sum() / obj_mask.sum().clamp(min=1)
        noobj_loss = (self.bce(pred[..., 4], obj_mask) * noobj_mask).sum() / noobj_mask.sum().clamp(min=1)

        cls_loss = (self.bce(pred[..., 5:], tcls) * obj_mask.unsqueeze(-1)).sum() / obj_mask.sum().clamp(min=1)

        total = (self.lambda_box * box_loss + self.lambda_obj * obj_loss
                 + self.lambda_noobj * noobj_loss + self.lambda_cls * cls_loss)
        return total, {
            "box": box_loss.item(), "obj": obj_loss.item(),
            "noobj": noobj_loss.item(), "cls": cls_loss.item(),
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    crit = YOLOLoss(num_classes=20)
    pred = torch.randn(2, 75, 13, 13, requires_grad=True)
    targets = [torch.tensor([[3, 0.5, 0.5, 0.2, 0.3]]), torch.zeros(0, 5)]
    loss, parts = crit(pred, targets, scale_idx=0)
    loss.backward()
    print("loss:", loss.item(), parts)
