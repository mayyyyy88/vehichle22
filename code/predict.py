import argparse
import json
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to image")
args = parser.parse_args()

model = tf.keras.models.load_model("models/best_vehicle_model.keras")

with open("results/class_names.json", "r") as f:
    class_names = json.load(f)

img = tf.keras.utils.load_img(args.image, target_size=IMG_SIZE)
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)

pred = model.predict(x, verbose=0)[0]
best_idx = int(np.argmax(pred))

print("Image:", args.image)
print("Predicted class:", class_names[best_idx])
print("Confidence: {:.2f}%".format(pred[best_idx] * 100))

print("\nAll class scores:")
for idx in np.argsort(pred)[::-1]:
    print(f"{class_names[idx]}: {pred[idx]*100:.2f}%")
