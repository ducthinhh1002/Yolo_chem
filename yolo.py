import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

# Paths
DATASET_DIR = Path("graph_dataset")
SPLIT_DIR = Path("graph_dataset_split")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def prepare_dataset(
    src: Path = DATASET_DIR,
    dest: Path = SPLIT_DIR,
    test_ratio: float = 0.2,
) -> Path:
    """Split dataset into train and test folders expected by YOLO classification."""
    if dest.exists():
        # Assume dataset already prepared
        return dest

    for cls_dir in src.iterdir():
        if cls_dir.is_dir():
            for split in ("train", "test"):
                (dest / split / cls_dir.name).mkdir(parents=True, exist_ok=True)

            images = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
            random.shuffle(images)
            test_count = int(len(images) * test_ratio)
            splits = {"test": images[:test_count], "train": images[test_count:]}
            for split, files in splits.items():
                for f in files:
                    shutil.copy2(f, dest / split / cls_dir.name / f.name)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO model for image classification")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size for training")
    args = parser.parse_args()

    data_path = prepare_dataset()
    model = YOLO("yolov8n-cls.pt")
    model.train(data=str(data_path), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, val=False)

    metrics = model.val(data=str(data_path / "test"))
    print(f"Top1 accuracy: {metrics.top1:.4f}")
    print(f"Top5 accuracy: {metrics.top5:.4f}")

    results = model.predict(source=str(data_path / "test"), max_det=1)
    for r in results[:5]:
        name = r.names[r.probs.top1]
        conf = float(r.probs.top1conf)
        print(f"{r.path}: {name} ({conf:.2f})")


if __name__ == "__main__":
    main()
