from pathlib import Path
import argparse
import csv
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 3

tf.random.set_seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True, help="Folder containing train/val/test")
parser.add_argument("--out-dir", required=True, help="Output folder")
args = parser.parse_args()

ROOT = Path(args.data_root)
OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = ROOT / "train"
VAL_DIR = ROOT / "val"
TEST_DIR = ROOT / "test"

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, label_mode="int", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    shuffle=True, seed=SEED
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, label_mode="int", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    shuffle=False
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, label_mode="int", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
with open(OUT_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
])

MODELS = [
    ("MobileNetV2", tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
    ("ResNet50", tf.keras.applications.ResNet50, tf.keras.applications.resnet.preprocess_input),
    ("EfficientNetB0", tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
]

rows = []

for model_name, backbone_fn, preprocess_fn in MODELS:
    print(f"\n===== Training {model_name} =====")
    tf.keras.backend.clear_session()

    base = backbone_fn(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = aug(inputs)
    x = preprocess_fn(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    ckpt = OUT_DIR / f"{model_name}_best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(ckpt), monitor="val_loss", save_best_only=True),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    hist1 = model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS, callbacks=callbacks, verbose=1)

    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    hist2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=HEAD_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=len(hist1.history["loss"]),
        callbacks=callbacks, verbose=1
    )

    best_model = tf.keras.models.load_model(ckpt)
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
    best_val_acc = max(hist1.history["val_accuracy"] + hist2.history.get("val_accuracy", []))
    best_val_loss = min(hist1.history["val_loss"] + hist2.history.get("val_loss", []))

    rows.append({
        "model": model_name,
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
        "best_val_accuracy": round(float(best_val_acc), 4),
        "best_val_loss": round(float(best_val_loss), 4),
    })

csv_path = OUT_DIR / "comparison_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "test_accuracy", "test_loss", "best_val_accuracy", "best_val_loss"])
    writer.writeheader()
    writer.writerows(rows)

models = [r["model"] for r in rows]
accs = [r["test_accuracy"] for r in rows]
losses = [r["test_loss"] for r in rows]

plt.figure(figsize=(7,4))
plt.bar(models, accs)
plt.ylabel("Test Accuracy")
plt.title("CNN Model Comparison - Test Accuracy")
plt.tight_layout()
plt.savefig(OUT_DIR / "test_accuracy_bar.png")
plt.close()

plt.figure(figsize=(7,4))
plt.bar(models, losses)
plt.ylabel("Test Loss")
plt.title("CNN Model Comparison - Test Loss")
plt.tight_layout()
plt.savefig(OUT_DIR / "test_loss_bar.png")
plt.close()

print("\n===== Final Comparison =====")
for r in sorted(rows, key=lambda x: x["test_accuracy"], reverse=True):
    print(r)

print(f"\nSaved table to: {csv_path}")
print(f"Saved charts to: {OUT_DIR}")
