import argparse
import random
import shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image  # dùng PIL

# =========================
# Cấu hình
# =========================
DATASET_DIR = Path("graph_dataset")          # nguồn: mỗi class là 1 thư mục
SPLIT_DIR = Path("graph_dataset_split_det")  # đích: detection format
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
RNG_SEED = 1337


def save_as_rgb(src_path: Path, dest_path: Path, force_ext: str = ".jpg",
                background=(255, 255, 255)) -> tuple[Path, int, int]:
    """
    Mở ảnh bất kỳ, ép về RGB 3 kênh (xử lý RGBA/grayscale), lưu ra đích (mặc định .jpg).
    Trả về: (đường dẫn ảnh đã lưu, width, height)
    """
    im = Image.open(src_path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, background)
        bg.paste(im, mask=im.split()[3])
        im = bg
    elif im.mode != "RGB":
        im = im.convert("RGB")
    if force_ext:
        dest_path = dest_path.with_suffix(force_ext)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest_path, quality=95)
    w, h = im.size
    return dest_path, w, h


def prepare_dataset(
    src: Path = DATASET_DIR,
    dest: Path = SPLIT_DIR,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    bbox_min: float = 0.95,
    bbox_max: float = 0.99,
) -> Path:
    """
    Chuẩn bị dataset theo định dạng YOLO Detection:
    - images/{train,val,test}
    - labels/{train,val,test}
    - data.yaml
    Mỗi ảnh sinh 1 bbox phủ ~95-99% ảnh: (x=0.5,y=0.5,w∈[bbox_min,bbox_max],h∈[bbox_min,bbox_max])
    Trả về: đường dẫn file data.yaml
    """
    # đảm bảo tổng = 1.0
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Tổng tỉ lệ phải = 1.0"

    # validate bbox range
    if not (0 < bbox_min <= bbox_max <= 1.0):
        raise ValueError("bbox_min/bbox_max phải nằm trong (0, 1] và bbox_min ≤ bbox_max")

    random.seed(RNG_SEED)

    # Xóa dataset cũ nếu tồn tại
    if dest.exists():
        shutil.rmtree(dest)

    # Tạo thư mục đích
    for split in ("train", "val", "test"):
        (dest / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Lấy danh sách class
    class_dirs = sorted([d for d in src.iterdir() if d.is_dir()], key=lambda p: p.name)
    if not class_dirs:
        raise ValueError(f"Không tìm thấy thư mục class trong {src}")

    class_names = [d.name for d in class_dirs]
    name_to_id = {name: i for i, name in enumerate(class_names)}
    print("🔖 Class mapping:", name_to_id)

    # Duyệt từng class
    for cls_dir in class_dirs:
        images = []
        for ext in IMG_EXTS:
            images.extend(cls_dir.rglob(f"*{ext}"))

        if not images:
            print(f"⚠️ Cảnh báo: Class {cls_dir.name} không có ảnh hợp lệ")
            continue

        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        splits = {"train": train_files, "val": val_files, "test": test_files}

        for split_name, files in splits.items():
            if not files:
                print(f"⚠️ Không đủ ảnh cho {split_name} trong class {cls_dir.name}")
                continue

            for img_file in files:
                # chuẩn hoá tên + ép về JPG RGB
                stem = f"{cls_dir.name}__{img_file.stem}"  # tránh trùng tên giữa các class
                img_dest_path = dest / "images" / split_name / f"{stem}.jpg"
                img_dest_path, w, h = save_as_rgb(img_file, img_dest_path)  # ép về RGB 3 kênh

                # Sinh nhãn YOLO: <class_id> x y w h (normalized)
                # bbox phủ 95–99% theo 2 chiều, tâm ở giữa ảnh
                bw = random.uniform(bbox_min, bbox_max)
                bh = random.uniform(bbox_min, bbox_max)
                x_center = 0.5
                y_center = 0.5

                label_dest = dest / "labels" / split_name / f"{stem}.txt"
                with open(label_dest, "w", encoding="utf-8") as f:
                    # format 6 chữ số thập phân cho ổn định
                    f.write(f"{name_to_id[cls_dir.name]} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

    # Viết data.yaml
    data_yaml = dest / "data.yaml"
    yaml_text = (
        "path: " + dest.as_posix() + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n" +
        "".join([f"  - {n}\n" for n in class_names])
    )
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    return data_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    # Thêm tham số điều khiển bbox và lưu .pt
    parser.add_argument("--bbox_min", type=float, default=0.95, help="Chiều rộng/cao tối thiểu (normalized)")
    parser.add_argument("--bbox_max", type=float, default=0.99, help="Chiều rộng/cao tối đa (normalized)")
    parser.add_argument("--save_best_as", type=str, default="graph_yolov8n_best.pt",
                        help="Đường dẫn file .pt để lưu bản tốt nhất")
    parser.add_argument("--save_last_as", type=str, default="",
                        help="(Tuỳ chọn) Lưu thêm bản last.pt, để trống nếu không dùng")
    args = parser.parse_args()

    # Chuẩn bị dataset (detection)
    try:
        data_yaml = prepare_dataset(
            bbox_min=args.bbox_min,
            bbox_max=args.bbox_max,
        )
        print("✅ Đã chuẩn bị dataset YOLO Detection và tạo data.yaml!")
    except Exception as e:
        print(f"❌ Lỗi khi chuẩn bị dataset: {e}")
        return

    data_root = data_yaml.parent  # thư mục gốc chứa images/, labels/

    # Huấn luyện YOLOv8 DETECT với mosaic/mixup
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        mosaic=1.0,        # ép bật mosaic
        mixup=0.15,        # trộn ảnh nhẹ để đa dạng
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        patience=20,
    )

    # Lưu trọng số .pt: best và (tuỳ chọn) last
    try:
        best_pt = getattr(model.trainer, "best", None)
        last_pt = getattr(model.trainer, "last", None)
        if best_pt and args.save_best_as:
            Path(args.save_best_as).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(best_pt, args.save_best_as)
            print(f"💾 Đã lưu best model → {args.save_best_as}")
        if last_pt and args.save_last_as:
            Path(args.save_last_as).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(last_pt, args.save_last_as)
            print(f"💾 Đã lưu last model → {args.save_last_as}")
        # In kèm đường dẫn gốc trong runs/ để tiện kiểm tra
        if best_pt:
            print(f"(best.pt gốc của Ultralytics: {best_pt})")
        if last_pt:
            print(f"(last.pt gốc của Ultralytics: {last_pt})")
    except Exception as e:
        print(f"⚠️ Không thể lưu .pt: {e}")

    # Validate
    metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch)
    try:
        print("\nKết quả validation (mAP):")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50   : {metrics.box.map50:.4f}")
        print(f"mAP75   : {metrics.box.map75:.4f}")
    except Exception:
        print("ℹ️ Không đọc được metrics.box, vui lòng kiểm tra log trong runs/detect.")

    test_images_dir = data_root / "images" / "test"
    if test_images_dir.exists():
        print("\nDự đoán trên tập test (kết quả sẽ được lưu trong runs/detect/predict*):")
        results = model.predict(
            source=str(test_images_dir),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            save=True,
        )
        for r in results[:3]:
            num = 0 if r.boxes is None else len(r.boxes)
            print(f"{Path(r.path).name}: {num} bbox")
    else:
        print("⚠️ Không tìm thấy thư mục test để predict.") 


if __name__ == "__main__":
    main()
