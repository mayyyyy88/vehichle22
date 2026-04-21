from pathlib import Path
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 3

tf.random.set_seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True, help="dataset root with train/val/test")
parser.add_argument("--model-dir", required=True, help="where to save model files")
parser.add_argument("--results-dir", required=True, help="where to save metrics/plots")
args = parser.parse_args()

DATA_ROOT = Path(args.data_root)
MODEL_DIR = Path(args.model_dir)
RESULTS_DIR = Path(args.results_dir)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_ROOT / "train",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_ROOT / "val",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_ROOT / "test",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

with open(RESULTS_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
])

base_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=IMG_SIZE + (3,)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

best_model_path = MODEL_DIR / "best_vehicle_model.keras"
final_model_path = MODEL_DIR / "final_vehicle_model.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(best_model_path), monitor="val_loss", save_best_only=True),
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    callbacks=callbacks,
    verbose=1
)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=len(history1.history["loss"]),
    callbacks=callbacks,
    verbose=1
)

model.save(final_model_path)

best_model = tf.keras.models.load_model(best_model_path)
test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)

y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
pred_probs = best_model.predict(test_ds, verbose=0)
y_pred = np.argmax(pred_probs, axis=1)

report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
cm = confusion_matrix(y_true, y_pred)

acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss = history1.history["loss"] + history2.history["loss"]
val_loss = history1.history["val_loss"] + history2.history["val_loss"]

plt.figure(figsize=(8, 5))
plt.plot(acc, label="train_accuracy")
plt.plot(val_acc, label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "loss.png")
plt.close()

plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(num_classes), class_names, rotation=45)
plt.yticks(range(num_classes), class_names)
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png")
plt.close()

with open(RESULTS_DIR / "test_metrics.txt", "w") as f:
    f.write(f"Classes: {class_names}\n")
    f.write(f"Test loss: {test_loss:.4f}\n")
    f.write(f"Test accuracy: {test_acc:.4f}\n\n")
    f.write("Per-class metrics:\n")
    for cls in class_names:
        f.write(
            f"{cls}: precision={report[cls]['precision']:.4f}, "
            f"recall={report[cls]['recall']:.4f}, "
            f"f1={report[cls]['f1-score']:.4f}\n"
        )
    f.write("\nConfusion matrix:\n")
    f.write(str(cm))

print("\nTraining complete.")
print("Classes:", class_names)
print("Best model saved to:", best_model_path)
print("Final model saved to:", final_model_path)
print("Test accuracy:", round(float(test_acc) * 100, 2), "%")
print("Results folder:", RESULTS_DIR)
