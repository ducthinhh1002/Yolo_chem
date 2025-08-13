import argparse
import random
import shutil
from pathlib import Path
from ultralytics import YOLO

# Cấu hình
DATASET_DIR = Path("graph_dataset")
SPLIT_DIR = Path("graph_dataset_split")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def prepare_dataset(
    src: Path = DATASET_DIR,
    dest: Path = SPLIT_DIR,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Path:
    """Chuẩn bị dataset theo định dạng YOLO Classification"""
    # Xóa dataset cũ nếu tồn tại
    if dest.exists():
        shutil.rmtree(dest)

    # Tạo thư mục train/val/test
    for split in ("train", "val", "test"):
        (dest / split).mkdir(parents=True, exist_ok=True)

    # Lấy danh sách class từ thư mục gốc (giả sử mỗi class có thư mục riêng)
    class_dirs = [d for d in src.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"Không tìm thấy thư mục class trong {src}")

    for cls_dir in class_dirs:
        # Lấy tất cả ảnh trong class (bao gồm cả subdirectories)
        images = []
        for ext in IMG_EXTS:
            images.extend(cls_dir.rglob(f"*{ext}"))
        
        if not images:
            print(f"⚠️ Cảnh báo: Class {cls_dir.name} không có ảnh hợp lệ")
            continue

        # Chia tỷ lệ
        random.shuffle(images)
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        # Copy ảnh vào thư mục đích
        for split_name, files in splits.items():
            if not files:
                print(f"⚠️ Cảnh báo: Không đủ ảnh cho {split_name} trong class {cls_dir.name}")
                continue

            # Tạo thư mục class trong từng split
            class_dest_dir = dest / split_name / cls_dir.name
            class_dest_dir.mkdir(parents=True, exist_ok=True)

            for img_file in files:
                shutil.copy2(img_file, class_dest_dir / img_file.name)

    return dest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=224)
    args = parser.parse_args()

    # Chuẩn bị dataset
    try:
        data_path = prepare_dataset()
        print("✅ Đã chia dataset thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi chuẩn bị dataset: {e}")
        return

    # Huấn luyện model
    model = YOLO("yolov8n-cls.pt")
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
    )

    # Validate
    metrics = model.val(data=str(data_path))
    print(f"\nKết quả validation:")
    print(f"Top-1 Accuracy: {metrics.top1:.2f}%")
    print(f"Top-5 Accuracy: {metrics.top5:.2f}%")

    # Dự đoán trên tập test
    test_path = data_path / "test"
    if test_path.exists() and any(test_path.iterdir()):
        print("\nDự đoán trên tập test:")
        results = model.predict(source=str(test_path))
        for r in results[:3]:  # Hiển thị 3 kết quả đầu
            print(f"{Path(r.path).name}: {r.names[r.probs.top1]} (confidence: {r.probs.top1conf:.2f})")
    else:
        print("\n⚠️ Không tìm thấy ảnh trong tập test")

if __name__ == "__main__":
    main()