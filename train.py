"""
YOLOv1 training script for person detection.

Training procedure follows Section 2.2 of:
  "You Only Look Once: Unified, Real-Time Object Detection"
  Redmon et al., CVPR 2016  --  https://arxiv.org/abs/1506.02640

Optimizer : SGD, momentum 0.9, weight decay 5e-4.
LR schedule:
    epoch  0       : linear warm-up  1e-3 -> 1e-2
    epochs 1 - 74  : 1e-2
    epochs 75 - 104: 1e-3
    epochs 105-134 : 1e-4

Usage
-----
    python train.py --data_root ./data [--epochs 135] [--batch_size 64]
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

import config
from model import YOLOv1, count_parameters
from loss import YOLOv1Loss
from dataset import build_train_dataset, build_val_dataset
from utils import decode_predictions, compute_ap


# -----------------------------------------------------------------------
# Learning-rate helpers
# -----------------------------------------------------------------------
def get_lr(epoch: int, batch_idx: int, batches_per_epoch: int) -> float:
    """Return the learning rate for a given epoch and batch index."""
    if epoch < config.WARMUP_EPOCHS:
        # Linear warm-up within the first epoch.
        frac = batch_idx / max(batches_per_epoch, 1)
        return config.WARMUP_LR_START + frac * (
            config.WARMUP_LR_END - config.WARMUP_LR_START
        )
    for milestone, lr in zip(
        config.LR_MILESTONES + [float("inf")], config.LR_VALUES
    ):
        if epoch < milestone:
            return lr
    return config.LR_VALUES[-1]


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# -----------------------------------------------------------------------
# Training & evaluation loops
# -----------------------------------------------------------------------
def train_one_epoch(
    model: YOLOv1,
    loader: DataLoader,
    criterion: YOLOv1Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    batches_per_epoch = len(loader)

    for batch_idx, (images, targets) in enumerate(loader):
        # Update LR per batch (warm-up) or per epoch.
        lr = get_lr(epoch, batch_idx, batches_per_epoch)
        set_lr(optimizer, lr)

        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)
        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents explosion when training from scratch
        # with large lambda_coord and deep network.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.GRAD_CLIP_NORM
        )
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            print(
                f"  Epoch [{epoch+1}/{config.EPOCHS}] "
                f"Batch [{batch_idx+1}/{batches_per_epoch}]  "
                f"Loss: {loss.item():.4f}  LR: {lr:.6f}"
            )

    return total_loss / batches_per_epoch


@torch.no_grad()
def evaluate(
    model: YOLOv1,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Run inference on the validation set and return AP (person)."""
    model.eval()
    all_pred_boxes: list[torch.Tensor] = []
    all_gt_boxes: list[torch.Tensor] = []

    for images, targets in loader:
        images = images.to(device)
        preds = model(images)
        detections = decode_predictions(preds)

        # Collect per-image results.
        for i in range(images.size(0)):
            # Predictions: [x1,y1,x2,y2, score, cls].
            dets = detections[i]
            if dets.numel() > 0:
                all_pred_boxes.append(dets[:, :5].cpu())
            else:
                all_pred_boxes.append(torch.empty(0, 5))

            # Ground truth: decode target tensor back to boxes.
            tgt = targets[i]  # (S, S, 5+C)
            obj_mask = tgt[..., 4] == 1
            if obj_mask.any():
                gt_cells = tgt[obj_mask]  # (K, 6)
                gt_xy = gt_cells[:, :2]
                gt_wh = gt_cells[:, 2:4]

                # Recover absolute coords.
                indices = obj_mask.nonzero(as_tuple=False)  # (K, 2) row=gy, col=gx
                gx = indices[:, 1].float()
                gy = indices[:, 0].float()
                cx = (gx + gt_xy[:, 0]) / config.S
                cy = (gy + gt_xy[:, 1]) / config.S
                x1 = cx - gt_wh[:, 0] / 2
                y1 = cy - gt_wh[:, 1] / 2
                x2 = cx + gt_wh[:, 0] / 2
                y2 = cy + gt_wh[:, 1] / 2
                all_gt_boxes.append(torch.stack([x1, y1, x2, y2], dim=-1))
            else:
                all_gt_boxes.append(torch.empty(0, 4))

    ap = compute_ap(all_pred_boxes, all_gt_boxes)
    return ap


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv1 Person Detection Training")
    parser.add_argument("--data_root", type=str, default=config.DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--eval_interval", type=int, default=5,
                        help="Evaluate every N epochs.")
    args = parser.parse_args()

    # Device.
    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Reproducibility.
    torch.manual_seed(config.SEED)

    # Data.
    print("Loading datasets ...")
    train_ds = build_train_dataset(args.data_root)
    val_ds = build_val_dataset(args.data_root)
    print(f"  Train images: {len(train_ds)}")
    print(f"  Val   images: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model.
    model = YOLOv1().to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Loss.
    criterion = YOLOv1Loss().to(device)

    # Optimiser (Section 2.2).
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.WARMUP_LR_START,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )

    start_epoch = 0
    best_ap = 0.0

    # Resume from checkpoint.
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_ap = ckpt.get("best_ap", 0.0)
        print(f"Resumed from epoch {start_epoch}, best AP {best_ap:.4f}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # ---- Training loop --------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"Avg Loss: {avg_loss:.4f}  "
            f"Time: {elapsed:.1f}s"
        )

        # Periodic evaluation.
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            ap = evaluate(model, val_loader, device)
            print(f"  -> AP@0.5 (person): {ap:.4f}")

            if ap > best_ap:
                best_ap = ap
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_ap": best_ap,
                    },
                    os.path.join(config.CHECKPOINT_DIR, "best.pth"),
                )
                print(f"  -> Saved new best model (AP={best_ap:.4f})")

        # Save latest checkpoint every epoch.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_ap": best_ap,
            },
            os.path.join(config.CHECKPOINT_DIR, "latest.pth"),
        )

    print(f"\nTraining complete.  Best AP: {best_ap:.4f}")


if __name__ == "__main__":
    main()
