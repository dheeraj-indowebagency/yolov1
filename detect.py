"""
YOLOv1 inference / detection script.

Loads a trained checkpoint, runs detection on one or more images, and
optionally saves visualisations.

Usage
-----
    python detect.py --image photo.jpg --checkpoint checkpoints/best.pth
    python detect.py --image_dir images/ --checkpoint checkpoints/best.pth
"""

import argparse
import os

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

import config
from model import YOLOv1
from dataset import MEAN, STD
from utils import decode_predictions


def load_model(checkpoint_path: str, device: torch.device) -> YOLOv1:
    model = YOLOv1().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def preprocess(image: Image.Image) -> torch.Tensor:
    img = TF.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    img = TF.to_tensor(img)
    img = TF.normalize(img, MEAN, STD)
    return img.unsqueeze(0)


@torch.no_grad()
def detect(
    model: YOLOv1,
    image: Image.Image,
    device: torch.device,
    conf_threshold: float = config.CONFIDENCE_THRESHOLD,
    nms_threshold: float = config.NMS_IOU_THRESHOLD,
) -> torch.Tensor:
    """Return detections as (K, 6): [x1,y1,x2,y2, score, class]."""
    inp = preprocess(image).to(device)
    pred = model(inp)
    dets = decode_predictions(
        pred, conf_threshold=conf_threshold, nms_threshold=nms_threshold
    )
    return dets[0]


def draw_boxes(
    image: Image.Image, detections: torch.Tensor, line_width: int = 3
) -> Image.Image:
    """Draw detected bounding boxes on the image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        x1 = int(det[0].item() * w)
        y1 = int(det[1].item() * h)
        x2 = int(det[2].item() * w)
        y2 = int(det[3].item() * h)
        score = det[4].item()

        draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)
        label = f"person {score:.2f}"
        draw.text((x1, max(0, y1 - 18)), label, fill="red", font=font)

    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv1 Person Detection")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image.")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory of images to process.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint.")
    parser.add_argument("--conf_threshold", type=float,
                        default=config.CONFIDENCE_THRESHOLD)
    parser.add_argument("--nms_threshold", type=float,
                        default=config.NMS_IOU_THRESHOLD)
    parser.add_argument("--output_dir", type=str, default="./detections",
                        help="Directory to save annotated images.")
    args = parser.parse_args()

    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )

    model = load_model(args.checkpoint, device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Gather input images.
    image_paths: list[str] = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        for f in sorted(os.listdir(args.image_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(args.image_dir, f))

    if not image_paths:
        print("No images specified.  Use --image or --image_dir.")
        return

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        dets = detect(model, image, device, args.conf_threshold, args.nms_threshold)
        print(f"{path}: {len(dets)} detection(s)")

        annotated = draw_boxes(image, dets)
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        annotated.save(out_path)
        print(f"  -> saved to {out_path}")


if __name__ == "__main__":
    main()
