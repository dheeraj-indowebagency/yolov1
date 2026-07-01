"""
Utility functions: IoU, NMS, decoding predictions, mAP evaluation.
"""

import torch
import numpy as np

import config


# -----------------------------------------------------------------------
# Box IoU (corner format)
# -----------------------------------------------------------------------
def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes in corner format.

    Parameters
    ----------
    box1 : (M, 4)  [x1, y1, x2, y2]
    box2 : (K, 4)  [x1, y1, x2, y2]

    Returns
    -------
    (M, K) IoU matrix.
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # (M,)
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # (K,)

    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


# -----------------------------------------------------------------------
# Non-Maximum Suppression
# -----------------------------------------------------------------------
def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = config.NMS_IOU_THRESHOLD,
) -> torch.Tensor:
    """Greedy NMS.

    Parameters
    ----------
    boxes  : (N, 4)  [x1, y1, x2, y2]
    scores : (N,)
    iou_threshold : suppress boxes with IoU above this value.

    Returns
    -------
    LongTensor of kept indices.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep: list[int] = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = box_iou(boxes[i : i + 1], boxes[remaining])[0]
        order = remaining[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# -----------------------------------------------------------------------
# Decode network output -> list of detections
# -----------------------------------------------------------------------
def decode_predictions(
    predictions: torch.Tensor,
    conf_threshold: float = config.CONFIDENCE_THRESHOLD,
    nms_threshold: float = config.NMS_IOU_THRESHOLD,
    S: int = config.S,
    B: int = config.B,
    C: int = config.C,
) -> list[list[torch.Tensor]]:
    """Decode a batch of YOLOv1 outputs into per-image detection lists.

    Parameters
    ----------
    predictions : (N, S, S, B*5 + C)

    Returns
    -------
    List (length N) of tensors shaped (num_detections, 6) where each row is
    [x1, y1, x2, y2, confidence, class_prob].
    """
    batch_size = predictions.size(0)
    device = predictions.device
    all_detections: list[list[torch.Tensor]] = []

    for n in range(batch_size):
        pred = predictions[n]  # (S, S, B*5+C)

        class_probs = pred[..., B * 5:]                       # (S, S, C)

        boxes_list = []
        scores_list = []

        for b in range(B):
            off = b * 5
            xy = pred[..., off:off + 2]                       # (S,S,2)
            wh = pred[..., off + 2:off + 4]                   # (S,S,2)
            conf = pred[..., off + 4]                         # (S,S)

            # Build cell offsets.
            grid_y, grid_x = torch.meshgrid(
                torch.arange(S, device=device, dtype=pred.dtype),
                torch.arange(S, device=device, dtype=pred.dtype),
                indexing="ij",
            )

            cx = (grid_x + xy[..., 0]) / S
            cy = (grid_y + xy[..., 1]) / S
            w = wh[..., 0].clamp(min=0)
            h = wh[..., 1].clamp(min=0)

            x1 = (cx - w / 2).clamp(min=0)
            y1 = (cy - h / 2).clamp(min=0)
            x2 = (cx + w / 2).clamp(max=1)
            y2 = (cy + h / 2).clamp(max=1)

            boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(-1, 4)

            # Score = conf * max class prob   (person only, so C=1).
            cls_score = class_probs.view(-1, C).max(dim=-1)[0]
            score = (conf.view(-1) * cls_score)

            boxes_list.append(boxes)
            scores_list.append(score)

        boxes_all = torch.cat(boxes_list, dim=0)               # (S*S*B, 4)
        scores_all = torch.cat(scores_list, dim=0)             # (S*S*B,)

        # Filter by confidence.
        mask = scores_all > conf_threshold
        boxes_filt = boxes_all[mask]
        scores_filt = scores_all[mask]

        if boxes_filt.numel() == 0:
            all_detections.append(
                torch.empty(0, 6, device=device)
            )
            continue

        # NMS.
        keep = nms(boxes_filt, scores_filt, nms_threshold)
        boxes_keep = boxes_filt[keep]
        scores_keep = scores_filt[keep]

        # Person class only -> class prob = 1.
        cls_col = torch.ones(scores_keep.size(0), 1, device=device)
        dets = torch.cat(
            [boxes_keep, scores_keep.unsqueeze(-1), cls_col], dim=-1
        )
        all_detections.append(dets)

    return all_detections


# -----------------------------------------------------------------------
# Mean Average Precision (VOC-style 11-point interpolation)
# -----------------------------------------------------------------------
def compute_ap(
    all_pred_boxes: list[torch.Tensor],
    all_gt_boxes: list[torch.Tensor],
    iou_threshold: float = 0.5,
) -> float:
    """Compute Average Precision for a single class (person).

    Parameters
    ----------
    all_pred_boxes : list of (K_i, 5)  [x1, y1, x2, y2, score]  per image.
    all_gt_boxes   : list of (M_i, 4)  [x1, y1, x2, y2]         per image.

    Returns
    -------
    AP (float).
    """
    # Collect all predictions with image indices.
    preds: list[tuple[int, float, np.ndarray]] = []
    for img_idx, boxes in enumerate(all_pred_boxes):
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        for row in boxes:
            score = float(row[4])
            preds.append((img_idx, score, row[:4]))

    # Sort by score descending.
    preds.sort(key=lambda x: x[1], reverse=True)

    # Ground truths per image.
    gt_per_image: dict[int, np.ndarray] = {}
    total_gt = 0
    for img_idx, boxes in enumerate(all_gt_boxes):
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if len(boxes) > 0:
            gt_per_image[img_idx] = boxes
            total_gt += len(boxes)

    if total_gt == 0:
        return 0.0

    # Track which GT boxes have been matched.
    matched: dict[int, list[bool]] = {
        k: [False] * len(v) for k, v in gt_per_image.items()
    }

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for det_idx, (img_idx, _score, box) in enumerate(preds):
        gt_boxes = gt_per_image.get(img_idx)
        if gt_boxes is None or len(gt_boxes) == 0:
            fp[det_idx] = 1
            continue

        # Compute IoU with all GT boxes in this image.
        ious = _numpy_iou(box, gt_boxes)
        best_gt = int(np.argmax(ious))
        best_iou = ious[best_gt]

        if best_iou >= iou_threshold and not matched[img_idx][best_gt]:
            tp[det_idx] = 1
            matched[img_idx][best_gt] = True
        else:
            fp[det_idx] = 1

    # Precision / recall.
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / total_gt
    precision = cum_tp / (cum_tp + cum_fp)

    # 11-point interpolation (VOC2007 style).
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec_at_recall = precision[recall >= t]
        if len(prec_at_recall) == 0:
            p = 0.0
        else:
            p = float(np.max(prec_at_recall))
        ap += p / 11.0

    return ap


def _numpy_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one box against many.  All in [x1,y1,x2,y2]."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / (union + 1e-6)
