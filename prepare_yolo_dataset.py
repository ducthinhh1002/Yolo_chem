import random
import shutil
from pathlib import Path
from typing import List

def prepare_dataset(source_dir: str = './graph_dataset', train_ratio: float = 0.8, val_ratio: float = 0.2) -> List[str]:
    """Organize class-labelled images into YOLO detection format.

    Each image is assumed to contain a single object occupying the entire image.
    Label files are generated with a bounding box covering the full image.

    Args:
        source_dir: Root directory containing class subfolders with images.
        train_ratio: Fraction of images from each class to use for training.
        val_ratio: Fraction of images from each class to use for validation.

    Returns:
        List of class names discovered in the dataset.
    """
    source = Path(source_dir)
    classes = [p.name for p in source.iterdir() if p.is_dir() and p.name not in {'train', 'val', 'test'}]
    classes.sort()
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}

    # Create split directories
    for split in ['train', 'val', 'test']:
        (source / split / 'images').mkdir(parents=True, exist_ok=True)
        (source / split / 'labels').mkdir(parents=True, exist_ok=True)

    for cls in classes:
        class_folder = source / cls
        images = [p for p in class_folder.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        if not images:
            continue
        random.shuffle(images)
        n_train = int(len(images) * train_ratio)
        n_val = int(len(images) * val_ratio)
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        for split, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            for img_path in img_list:
                dest_img = source / split / 'images' / img_path.name
                shutil.copy2(img_path, dest_img)
                label_path = source / split / 'labels' / (img_path.stem + '.txt')
                with open(label_path, 'w') as f:
                    f.write(f"{class_to_id[cls]} 0.5 0.5 1 1\n")
        # Optionally remove original file
    # Write data.yaml
    yaml_lines = [
        f"path: {source_dir}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"names: {classes}",
        ""
    ]
    (source / 'data.yaml').write_text("\n".join(yaml_lines))
    return classes

if __name__ == '__main__':
    prepare_dataset()
