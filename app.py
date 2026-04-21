import json
from collections import Counter

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO

IMG_SIZE = (224, 224)
TARGET_CLASSES = {"car", "bus", "truck", "motorcycle"}

st.set_page_config(page_title="Vehicle Type Classification Demo", layout="centered")
st.title("Vehicle Type Classification and Counting Demo")
st.write("This app supports classification, batch counting, and crowded-image vehicle detection.")

@st.cache_resource
def load_classifier():
    return tf.keras.models.load_model("models/best_vehicle_model.keras")

@st.cache_resource
def load_detector():
    return YOLO("yolo11n.pt")

@st.cache_data
def load_class_names():
    with open("results/class_names.json", "r") as f:
        return json.load(f)

classifier = load_classifier()
detector = load_detector()
class_names = load_class_names()

def predict_image(img: Image.Image):
    img = img.convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    x = tf.keras.utils.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)

    pred = classifier.predict(x, verbose=0)[0]
    best_idx = int(np.argmax(pred))
    return class_names[best_idx], float(pred[best_idx]), pred

tab1, tab2, tab3 = st.tabs([
    "Single Image Prediction",
    "Count Vehicle Types",
    "Crowded Image Detection"
])

with tab1:
    st.subheader("Single Image Prediction")
    uploaded_file = st.file_uploader("Upload one image", type=["jpg", "jpeg", "png"], key="single")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        label, confidence, scores = predict_image(img)

        st.success(f"Predicted class: {label}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
        st.write("All class scores:")
        for idx in np.argsort(scores)[::-1]:
            st.write(f"- {class_names[idx]}: {scores[idx] * 100:.2f}%")

with tab2:
    st.subheader("Count Vehicle Types from Multiple Images")
    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="multi"
    )

    if uploaded_files:
        counts = Counter()
        results = []

        for f in uploaded_files:
            img = Image.open(f)
            label, confidence, _ = predict_image(img)
            counts[label] += 1
            results.append((f.name, label, confidence))

        st.write("Vehicle counts:")
        for cls in class_names:
            st.write(f"- {cls}: {counts[cls]}")

        st.write("Detailed predictions:")
        for name, label, confidence in results:
            st.write(f"- {name}: {label} ({confidence * 100:.2f}%)")

with tab3:
    st.subheader("Crowded Image Detection and Counting")
    uploaded_det = st.file_uploader("Upload a crowded road image", type=["jpg", "jpeg", "png"], key="detect")

    if uploaded_det is not None:
        img = Image.open(uploaded_det).convert("RGB")
        results = detector(img, conf=0.25, verbose=False)
        r = results[0]
        names = detector.names

        counts = Counter()
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            label = names[cls_id]
            if label in TARGET_CLASSES:
                counts[label] += 1

        annotated = r.plot()
        st.image(annotated, caption="Detected Vehicles", use_container_width=True, channels="BGR")

        st.write("Detected vehicle counts:")
        for cls in sorted(TARGET_CLASSES):
            st.write(f"- {cls}: {counts[cls]}")
