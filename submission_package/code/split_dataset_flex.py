from pathlib import Path
import argparse
import random
import shutil

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True, help="Source raw dataset folder")
parser.add_argument("--dest", required=True, help="Destination split dataset folder")
args = parser.parse_args()

SOURCE = Path(args.source)
DEST = Path(args.dest)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_images(folder):
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

for split in ["train", "val", "test"]:
    (DEST / split).mkdir(parents=True, exist_ok=True)

class_dirs = [d for d in SOURCE.iterdir() if d.is_dir()]

if not class_dirs:
    raise SystemExit(f"No class folders found in {SOURCE}")

for class_dir in class_dirs:
    images = get_images(class_dir)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_files = images[:n_train]
    val_files = images[n_train:n_train + n_val]
    test_files = images[n_train + n_val:]

    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        out_dir = DEST / split_name / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, out_dir / f.name)

    print(f"{class_dir.name}: total={n}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

print("Done.")
