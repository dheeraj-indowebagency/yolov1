"""
YOLOv1 multi-part loss function (Equation 3, Section 2.2).

    L  =  lambda_coord  * SUM_ij  1_{ij}^obj  [ (x - x')^2 + (y - y')^2 ]
        + lambda_coord  * SUM_ij  1_{ij}^obj  [ (sqrt(w) - sqrt(w'))^2
                                                + (sqrt(h) - sqrt(h'))^2 ]
        + SUM_ij  1_{ij}^obj    (C - C')^2
        + lambda_noobj  * SUM_ij  1_{ij}^noobj  (C - C')^2
        + SUM_i   1_i^obj  SUM_c (p(c) - p'(c))^2

Notation (from the paper):
    1_{ij}^obj   -- jth bbox predictor in cell i is *responsible* (highest IoU
                    with the ground-truth among the B predictors in that cell).
    1_{ij}^noobj -- complement of 1_{ij}^obj (all other predictor slots).
    1_i^obj      -- an object's centre falls in cell i.
    C'           -- target confidence.  The paper uses IoU(pred, gt) but this
                    causes confidence collapse when training from scratch.
                    Default: 1.0 for responsible predictor (stable from scratch).
    p(c)         -- conditional class probability.

Reference: https://arxiv.org/abs/1506.02640
"""

import torch
import torch.nn as nn

import config


class YOLOv1Loss(nn.Module):
    def __init__(
        self,
        S: int = config.S,
        B: int = config.B,
        C: int = config.C,
        lambda_coord: float = config.LAMBDA_COORD,
        lambda_noobj: float = config.LAMBDA_NOOBJ,
        use_iou_as_conf_target: bool = False,
    ):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        # When False (default), confidence target for the responsible
        # predictor is 1.0.  The paper uses IoU(pred, gt) as target, but
        # this causes confidence collapse when training from scratch:
        # early random predictions yield IoU ≈ 0 everywhere, so the model
        # never learns to produce high confidence scores.
        # Set to True only when using a pretrained backbone.
        self.use_iou_as_conf_target = use_iou_as_conf_target

    # ------------------------------------------------------------------
    # IoU helper
    # ------------------------------------------------------------------
    def _cell_to_abs(
        self, xy: torch.Tensor, wh: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert (cell-offset xy, image-relative wh) to corner coordinates.

        Parameters
        ----------
        xy : (..., S, S, 2)  x/y offsets inside a cell  [0, 1).
        wh : (..., S, S, 2)  width/height relative to the whole image.

        Returns
        -------
        x1, y1, x2, y2  each (..., S, S)
        """
        device = xy.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.S, device=device, dtype=xy.dtype),
            torch.arange(self.S, device=device, dtype=xy.dtype),
            indexing="ij",
        )
        # Absolute centre (image-relative 0-1).
        cx = (grid_x + xy[..., 0]) / self.S
        cy = (grid_y + xy[..., 1]) / self.S
        half_w = wh[..., 0] / 2
        half_h = wh[..., 1] / 2
        return cx - half_w, cy - half_h, cx + half_w, cy + half_h

    def _iou(
        self, xy1: torch.Tensor, wh1: torch.Tensor,
        xy2: torch.Tensor, wh2: torch.Tensor,
    ) -> torch.Tensor:
        """Element-wise IoU between two sets of boxes in cell-offset format.

        Both inputs are broadcastable with shape (..., S, S, 2).
        Returns (..., S, S).
        """
        x1_min, y1_min, x1_max, y1_max = self._cell_to_abs(xy1, wh1)
        x2_min, y2_min, x2_max, y2_max = self._cell_to_abs(xy2, wh2)

        inter_w = (torch.min(x1_max, x2_max) - torch.max(x1_min, x2_min)).clamp(min=0)
        inter_h = (torch.min(y1_max, y2_max) - torch.max(y1_min, y2_min)).clamp(min=0)
        inter = inter_w * inter_h

        area1 = (x1_max - x1_min).clamp(min=0) * (y1_max - y1_min).clamp(min=0)
        area2 = (x2_max - x2_min).clamp(min=0) * (y2_max - y2_min).clamp(min=0)
        union = area1 + area2 - inter

        return inter / (union + 1e-6)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : (N, S, S, B*5 + C)
            Raw network output (linear activation).
        targets : (N, S, S, 5 + C)
            Encoded ground truth: [x, y, w, h, obj, class...].

        Returns
        -------
        Scalar loss averaged over the batch.
        """
        N = predictions.size(0)

        # -- Unpack targets --------------------------------------------------
        tgt_xy = targets[..., 0:2]           # (N, S, S, 2)
        tgt_wh = targets[..., 2:4]           # (N, S, S, 2)
        obj_mask = targets[..., 4]            # (N, S, S)   1 where object
        tgt_cls = targets[..., 5:]            # (N, S, S, C)

        # -- Determine the responsible predictor (highest IoU with GT) --------
        # Clamp predicted w/h to non-negative for IoU computation so that
        # early random (negative) predictions don't collapse all IoU to 0.
        best_iou = torch.full_like(obj_mask, -1.0)
        best_idx = torch.zeros_like(obj_mask, dtype=torch.long)

        for b in range(self.B):
            offset = b * 5
            pred_xy = predictions[..., offset:offset + 2]
            pred_wh = predictions[..., offset + 2:offset + 4].clamp(min=0)
            iou = self._iou(pred_xy, pred_wh, tgt_xy, tgt_wh)
            better = iou > best_iou
            best_iou = torch.where(better, iou, best_iou)
            best_idx = torch.where(better, b, best_idx)

        # -- Accumulate loss terms -------------------------------------------
        coord_loss = torch.tensor(0.0, device=predictions.device)
        obj_conf_loss = torch.tensor(0.0, device=predictions.device)
        noobj_conf_loss = torch.tensor(0.0, device=predictions.device)

        for b in range(self.B):
            offset = b * 5
            pred_xy = predictions[..., offset:offset + 2]       # (N,S,S,2)
            pred_wh = predictions[..., offset + 2:offset + 4]   # (N,S,S,2)
            pred_conf = predictions[..., offset + 4]             # (N,S,S)

            # 1_{ij}^obj : responsible predictor in cells that have an object.
            responsible = ((best_idx == b).float() * obj_mask)   # (N,S,S)
            # 1_{ij}^noobj : everything else.
            not_responsible = 1.0 - responsible

            resp = responsible.unsqueeze(-1)                     # (N,S,S,1)

            # ---- Coordinate loss (xy) --------------------------------------
            xy_loss = (resp * (pred_xy - tgt_xy) ** 2).sum()

            # ---- Coordinate loss (sqrt wh) ---------------------------------
            # Guard against negative raw predictions with sign-preserving sqrt.
            pred_w_sqrt = torch.sign(pred_wh[..., 0:1]) * torch.sqrt(
                torch.abs(pred_wh[..., 0:1]) + 1e-6
            )
            pred_h_sqrt = torch.sign(pred_wh[..., 1:2]) * torch.sqrt(
                torch.abs(pred_wh[..., 1:2]) + 1e-6
            )
            tgt_w_sqrt = torch.sqrt(tgt_wh[..., 0:1] + 1e-6)
            tgt_h_sqrt = torch.sqrt(tgt_wh[..., 1:2] + 1e-6)
            wh_loss = (resp * (pred_w_sqrt - tgt_w_sqrt) ** 2).sum() + \
                      (resp * (pred_h_sqrt - tgt_h_sqrt) ** 2).sum()

            coord_loss = coord_loss + xy_loss + wh_loss

            # ---- Confidence loss (obj) --------------------------------------
            # Target confidence: use 1.0 (stable for from-scratch training)
            # or IoU(pred, gt) as described in the paper (requires pretrained
            # backbone to avoid confidence collapse).
            if self.use_iou_as_conf_target:
                pred_wh_pos = pred_wh.clamp(min=0)
                conf_target = self._iou(
                    pred_xy, pred_wh_pos, tgt_xy, tgt_wh
                ).detach()
            else:
                conf_target = obj_mask  # 1.0 where object exists
            obj_conf_loss = obj_conf_loss + (
                responsible * (pred_conf - conf_target) ** 2
            ).sum()

            # ---- Confidence loss (noobj) ------------------------------------
            noobj_conf_loss = noobj_conf_loss + (
                not_responsible * (pred_conf - 0) ** 2
            ).sum()

        # ---- Class probability loss -----------------------------------------
        pred_cls = predictions[..., self.B * 5:]                 # (N,S,S,C)
        cls_loss = (obj_mask.unsqueeze(-1) * (pred_cls - tgt_cls) ** 2).sum()

        # ---- Total ----------------------------------------------------------
        total = (
            self.lambda_coord * coord_loss
            + obj_conf_loss
            + self.lambda_noobj * noobj_conf_loss
            + cls_loss
        )
        return total / N
