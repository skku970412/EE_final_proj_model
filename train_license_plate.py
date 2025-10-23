"""License plate detector training entry point using Ultralytics YOLO.

This script reproduces the configuration that was used for the bundled
`runs/license_plate_yolov8n` checkpoint. It lets you retrain the model from
scratch (or continue training) against the Roboflow-provided dataset that lives
next to this project.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = Path("yolov8n.pt")
DEFAULT_DATA = Path("../Korea Car License Plate/data.yaml")
DEFAULT_PROJECT = Path("runs")
DEFAULT_NAME = "license_plate_yolov8n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the YOLOv8 license plate detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Initial model weights (can be a pretrained checkpoint or YAML).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Ultralytics data YAML describing train/val/test splits.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT,
        help="Directory where Ultralytics will save runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_NAME,
        help="Name of this training run.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device identifier passed to Ultralytics (e.g. '0' for the first GPU or 'cpu').",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = (
        args.model
        if args.model.is_absolute()
        else (PROJECT_ROOT / args.model).resolve()
    )
    data_path = (
        args.data
        if args.data.is_absolute()
        else (PROJECT_ROOT / args.data).resolve()
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights or YAML not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_path}")

    model = YOLO(str(model_path))

    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(
            args.project
            if args.project.is_absolute()
            else (PROJECT_ROOT / args.project).resolve()
        ),
        name=args.name,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        patience=100,
        pretrained=True,
        val=True,
        save=True,
        close_mosaic=10,
        amp=True,
    )

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
