import argparse
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

TARGET_CLASSES = {"car", "bus", "truck", "motorcycle"}

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to crowded road image")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
parser.add_argument("--out", default="detector_demo/output.jpg", help="Annotated output image")
args = parser.parse_args()

Path("detector_demo").mkdir(exist_ok=True)

model = YOLO("yolo11n.pt")
results = model(args.image, conf=args.conf, verbose=False)

r = results[0]
names = model.names
counts = Counter()

for box in r.boxes:
    cls_id = int(box.cls[0].item())
    label = names[cls_id]
    if label in TARGET_CLASSES:
        counts[label] += 1

annotated = r.plot()
cv2.imwrite(args.out, annotated)

print("\nVehicle counts in this image:")
for cls in sorted(TARGET_CLASSES):
    print(f"{cls}: {counts[cls]}")
print(f"\nAnnotated image saved to: {args.out}")
