"""
PASCAL VOC dataset loader for YOLOv1 person detection.

Training  : VOC 2007 trainval  +  VOC 2012 trainval
Evaluation: VOC 2007 test

Each image is resized to 448x448.  Ground-truth bounding boxes are encoded
into an (S, S, 5+C) target tensor suitable for the YOLOv1 loss.

Data augmentation (Section 2.2):
  - Random horizontal flip.
  - Random scaling and translation of up to 20 %.
  - Random adjustment of exposure and saturation by up to 1.5x in HSV space.
"""

import os
import random
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import torchvision.transforms.functional as TF

import config


# ImageNet channel statistics (used even when training from scratch --
# keeps inputs in a well-conditioned range).
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class VOCPersonDataset(Dataset):
    """Single-year PASCAL VOC split filtered for the *person* class."""

    def __init__(
        self,
        root: str,
        year: str = "2007",
        image_set: str = "trainval",
        augment: bool = False,
        S: int = config.S,
        B: int = config.B,
        C: int = config.C,
        image_size: int = config.IMAGE_SIZE,
    ):
        super().__init__()
        self.augment = augment
        self.S = S
        self.B = B
        self.C = C
        self.image_size = image_size

        voc_root = os.path.join(root, f"VOCdevkit/VOC{year}")
        self.image_dir = os.path.join(voc_root, "JPEGImages")
        self.annot_dir = os.path.join(voc_root, "Annotations")

        id_file = os.path.join(voc_root, "ImageSets", "Main", f"{image_set}.txt")
        with open(id_file) as f:
            self.image_ids = [line.strip().split()[0] for line in f if line.strip()]

        # Pre-parse all annotations.
        self.annotations: list[dict] = []
        for img_id in self.image_ids:
            annot_path = os.path.join(self.annot_dir, f"{img_id}.xml")
            boxes, img_w, img_h = self._parse_xml(annot_path)
            self.annotations.append(
                {"id": img_id, "boxes": boxes, "width": img_w, "height": img_h}
            )

    # ------------------------------------------------------------------
    # XML parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_xml(path: str):
        tree = ET.parse(path)
        root = tree.getroot()

        size = root.find("size")
        img_w = float(size.find("width").text)
        img_h = float(size.find("height").text)

        boxes: list[list[float]] = []
        for obj in root.findall("object"):
            if obj.find("name").text != config.TARGET_CLASS:
                continue
            if int(obj.find("difficult").text):
                continue
            bb = obj.find("bndbox")
            xmin = max(0.0, float(bb.find("xmin").text) / img_w)
            ymin = max(0.0, float(bb.find("ymin").text) / img_h)
            xmax = min(1.0, float(bb.find("xmax").text) / img_w)
            ymax = min(1.0, float(bb.find("ymax").text) / img_h)
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
        return boxes, img_w, img_h

    # ------------------------------------------------------------------
    # Data augmentation (Section 2.2)
    # ------------------------------------------------------------------
    def _augment(
        self, image: Image.Image, boxes: list[list[float]]
    ) -> tuple[Image.Image, list[list[float]]]:
        w, h = image.size

        # --- 1. Random horizontal flip ---
        if random.random() < config.HORIZONTAL_FLIP_PROB:
            image = TF.hflip(image)
            boxes = [[1.0 - b[2], b[1], 1.0 - b[0], b[3]] for b in boxes]

        # --- 2. Random scaling + translation (up to 20 %) ---
        scale = random.uniform(*config.SCALE_RANGE)
        dx = random.uniform(-config.TRANSLATE_MAX, config.TRANSLATE_MAX)
        dy = random.uniform(-config.TRANSLATE_MAX, config.TRANSLATE_MAX)

        tx_px = int(dx * w)
        ty_px = int(dy * h)
        dx_exact = tx_px / w if w else 0
        dy_exact = ty_px / h if h else 0

        image = TF.affine(
            image, angle=0, translate=[tx_px, ty_px],
            scale=scale, shear=0, fill=(128, 128, 128),
        )

        new_boxes: list[list[float]] = []
        for b in boxes:
            nx1 = (b[0] - 0.5) * scale + 0.5 + dx_exact
            ny1 = (b[1] - 0.5) * scale + 0.5 + dy_exact
            nx2 = (b[2] - 0.5) * scale + 0.5 + dx_exact
            ny2 = (b[3] - 0.5) * scale + 0.5 + dy_exact
            nx1, ny1 = max(0.0, nx1), max(0.0, ny1)
            nx2, ny2 = min(1.0, nx2), min(1.0, ny2)
            if nx2 - nx1 > 0.01 and ny2 - ny1 > 0.01:
                new_boxes.append([nx1, ny1, nx2, ny2])
        boxes = new_boxes

        # --- 3. HSV colour jitter (saturation & exposure) ---
        sat = random.uniform(*config.SATURATION_RANGE)
        exp = random.uniform(*config.EXPOSURE_RANGE)
        hsv = image.convert("HSV")
        h_ch, s_ch, v_ch = hsv.split()
        s_ch = s_ch.point(lambda p: min(255, max(0, int(p * sat))))
        v_ch = v_ch.point(lambda p: min(255, max(0, int(p * exp))))
        image = Image.merge("HSV", (h_ch, s_ch, v_ch)).convert("RGB")

        return image, boxes

    # ------------------------------------------------------------------
    # Target encoding
    # ------------------------------------------------------------------
    def _encode_target(self, boxes: list[list[float]]) -> torch.Tensor:
        """Encode bounding boxes into an (S, S, 5+C) target tensor.

        Per cell:
            [x_offset, y_offset, w, h, objectness, class_person]
        """
        target = torch.zeros(self.S, self.S, 5 + self.C)
        for b in boxes:
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            bw = b[2] - b[0]
            bh = b[3] - b[1]

            gi = int(cx * self.S)
            gj = int(cy * self.S)
            gi = min(gi, self.S - 1)
            gj = min(gj, self.S - 1)

            # Only one object per cell (YOLOv1 limitation).
            if target[gj, gi, 4] == 1:
                continue

            x_off = cx * self.S - gi
            y_off = cy * self.S - gj

            target[gj, gi, 0] = x_off
            target[gj, gi, 1] = y_off
            target[gj, gi, 2] = bw
            target[gj, gi, 3] = bh
            target[gj, gi, 4] = 1.0         # objectness
            target[gj, gi, 5] = 1.0         # class = person

        return target

    # ------------------------------------------------------------------
    # __getitem__ / __len__
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        annot = self.annotations[idx]
        boxes = [b[:] for b in annot["boxes"]]  # copy

        image = Image.open(
            os.path.join(self.image_dir, f"{annot['id']}.jpg")
        ).convert("RGB")

        if self.augment and boxes:
            image, boxes = self._augment(image, boxes)

        image = TF.resize(image, (self.image_size, self.image_size))
        image = TF.to_tensor(image)
        image = TF.normalize(image, MEAN, STD)

        target = self._encode_target(boxes)
        return image, target


# ----------------------------------------------------------------------
# Convenience builders
# ----------------------------------------------------------------------
def build_train_dataset(root: str = config.DATA_ROOT) -> ConcatDataset:
    """VOC 2007 trainval + VOC 2012 trainval (with augmentation)."""
    ds07 = VOCPersonDataset(root, year="2007", image_set="trainval", augment=True)
    ds12 = VOCPersonDataset(root, year="2012", image_set="trainval", augment=True)
    return ConcatDataset([ds07, ds12])


def build_val_dataset(root: str = config.DATA_ROOT) -> VOCPersonDataset:
    """VOC 2007 test (no augmentation)."""
    return VOCPersonDataset(root, year="2007", image_set="test", augment=False)
