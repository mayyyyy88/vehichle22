from pathlib import Path
import argparse
import json
from collections import Counter
import csv

import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True, help="Folder containing images")
parser.add_argument("--out", default="results/batch_predictions.csv", help="CSV output file")
args = parser.parse_args()

model = tf.keras.models.load_model("models/best_vehicle_model.keras")

with open("results/class_names.json", "r") as f:
    class_names = json.load(f)

folder = Path(args.dir)
files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
files.sort()

if not files:
    raise SystemExit("No images found.")

counts = Counter()
rows = []

for file_path in files:
    img = tf.keras.utils.load_img(file_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(pred))
    label = class_names[idx]
    conf = float(pred[idx])

    counts[label] += 1
    rows.append([str(file_path), label, conf])

with open(args.out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "predicted_class", "confidence"])
    writer.writerows(rows)

print("Vehicle counts:")
for cls in class_names:
    print(f"{cls}: {counts[cls]}")

print("\nSaved detailed predictions to:", args.out)
