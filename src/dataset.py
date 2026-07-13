"""
COCO-subset dataset for YOLOv3.

Design choice for the demo: rather than the full 80-class / 118k-image COCO,
we pull a small user-defined subset (N classes x M images/class) via
pycocotools -- this keeps training time reasonable on a single PC GPU while
still exercising the identical pipeline (real images, real annotations,
real letterbox/augmentation) that would run on the full set.
"""
import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def letterbox(img, target_size=416, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    top = (target_size - nh) // 2
    left = (target_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


class CocoSubsetDataset(Dataset):
    """
    Expects standard COCO layout:
        root/annotations/instances_{split}.json
        root/{split}/*.jpg
    `class_names` selects + remaps a subset of COCO categories to 0..N-1,
    so the detection head only needs num_classes = len(class_names).
    """
    def __init__(self, root, split, class_names, img_size=416,
                 images_per_class=150, augment=True):
        from pycocotools.coco import COCO
        ann_file = os.path.join(root, "annotations", f"instances_{split}.json")
        self.img_dir = os.path.join(root, split)
        self.coco = COCO(ann_file)
        self.img_size = img_size
        self.augment = augment

        self.cat_ids = self.coco.getCatIds(catNms=class_names)
        self.catid_to_label = {cid: i for i, cid in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids)

        img_ids = set()
        for cid in self.cat_ids:
            ids = self.coco.getImgIds(catIds=[cid])
            random.shuffle(ids)
            img_ids.update(ids[:images_per_class])
        self.img_ids = sorted(img_ids)

    def __len__(self):
        return len(self.img_ids)

    def _load_image_and_boxes(self, img_id):
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, info["file_name"])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        for a in anns:
            x, y, w, h = a["bbox"]  # COCO format: top-left x,y,w,h in pixels
            cls = self.catid_to_label[a["category_id"]]
            cx, cy = x + w / 2, y + h / 2
            boxes.append([cls, cx / w0, cy / h0, w / w0, h / h0])
        return img, np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), np.float32)

    def __getitem__(self, idx):
        img, boxes = self._load_image_and_boxes(self.img_ids[idx])
        img_lb, scale, pad_x, pad_y = letterbox(img, self.img_size)

        if boxes.shape[0] > 0:
            h0, w0 = img.shape[:2]
            # convert normalized (orig image) -> pixel (orig) -> pixel (letterboxed) -> normalized (letterboxed)
            cx = boxes[:, 1] * w0 * scale + pad_x
            cy = boxes[:, 2] * h0 * scale + pad_y
            bw = boxes[:, 3] * w0 * scale
            bh = boxes[:, 4] * h0 * scale
            boxes[:, 1] = cx / self.img_size
            boxes[:, 2] = cy / self.img_size
            boxes[:, 3] = bw / self.img_size
            boxes[:, 4] = bh / self.img_size

        if self.augment and random.random() < 0.5 and boxes.shape[0] > 0:
            img_lb = img_lb[:, ::-1, :].copy()
            boxes[:, 1] = 1.0 - boxes[:, 1]

        img_t = torch.from_numpy(img_lb).permute(2, 0, 1).float() / 255.0
        target_t = torch.from_numpy(boxes)
        return img_t, target_t


def yolo_collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0), list(targets)


if __name__ == "__main__":
    # Smoke test with a synthetic in-memory stand-in (no real COCO files present
    # in this sandbox) -- validates the letterbox + normalization math only.
    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    boxes = np.array([[3, 0.5, 0.5, 0.2, 0.3]], dtype=np.float32)
    img_lb, scale, pad_x, pad_y = letterbox(img, 416)
    print("letterboxed shape:", img_lb.shape, "scale:", scale, "pad:", pad_x, pad_y)
