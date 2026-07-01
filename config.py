"""
YOLOv1 Configuration.

All hyperparameters follow "You Only Look Once: Unified, Real-Time Object
Detection" (Redmon et al., CVPR 2016).

Reference: https://arxiv.org/abs/1506.02640
"""

import os

# ---------------------------------------------------------------------------
# Grid & prediction parameters (Section 2)
# ---------------------------------------------------------------------------
S = 7               # Divide the image into an S x S grid.
B = 2               # Each cell predicts B bounding boxes.
C = 1               # Number of classes (person only).

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
IMAGE_SIZE = 448     # The network input resolution (Section 2).

# ---------------------------------------------------------------------------
# Loss weights (Equation 3, Section 2.2)
# ---------------------------------------------------------------------------
LAMBDA_COORD = 5.0   # Up-weight coordinate predictions.
LAMBDA_NOOBJ = 0.5   # Down-weight confidence for cells without objects.

# ---------------------------------------------------------------------------
# Training hyper-parameters (Section 2.2)
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 135
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DROPOUT = 0.5        # Applied after the first fully-connected layer.

# Learning-rate schedule (Section 2.2):
#   epoch  0      : linearly warm up from 1e-3 to 1e-2
#   epochs 1 - 74 : 1e-2
#   epochs 75-104 : 1e-3
#   epochs 105-134: 1e-4
WARMUP_LR_START = 1e-3
WARMUP_LR_END = 1e-2
WARMUP_EPOCHS = 1
LR_MILESTONES = [75, 105]       # Epochs at which the rate drops.
LR_VALUES = [1e-2, 1e-3, 1e-4]  # Rates for each segment.

# ---------------------------------------------------------------------------
# Data augmentation (Section 2.2)
# ---------------------------------------------------------------------------
SCALE_RANGE = (0.8, 1.2)           # Random scaling up to 20%.
TRANSLATE_MAX = 0.2                 # Random translation up to 20%.
SATURATION_RANGE = (1.0 / 1.5, 1.5)  # HSV saturation factor.
EXPOSURE_RANGE = (1.0 / 1.5, 1.5)    # HSV value (exposure) factor.
HORIZONTAL_FLIP_PROB = 0.5

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get("YOLO_DATA_ROOT", "./data")
TARGET_CLASS = "person"

# ---------------------------------------------------------------------------
# Inference (Section 2.4)
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.2
NMS_IOU_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------
NUM_WORKERS = 4
DEVICE = "cuda"          # Overridden to "cpu" when CUDA is unavailable.
CHECKPOINT_DIR = "./checkpoints"
SEED = 42
