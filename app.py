import json
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️"
)

MODEL_PATH = "models/waste_model.keras"
LABEL_PATH = "models/class_names.json"

st.title("♻️ Smart Waste Classification")
st.write(
    "Upload an image to identify its waste category "
    "using fine-tuned MobileNetV2."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_labels():
    with open(LABEL_PATH, "r") as f:
        return json.load(f)

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run python train.py first.")
    st.stop()

if not os.path.exists(LABEL_PATH):
    st.error("Category file not found. Run python train.py first.")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=350)

    if st.button("Classify Waste"):
        with st.spinner("Analyzing image..."):
            model = load_model()
            class_names = load_labels()

            image = image.resize((224, 224))
            image_array = np.array(image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_index = int(np.argmax(predictions))
            predicted_label = class_names[predicted_index]
            confidence = float(predictions[predicted_index])

            st.success(
                f"Predicted category: {predicted_label.title()}"
            )
            st.write(
                f"Model confidence: {confidence * 100:.2f}%"
            )

            st.subheader("Category probabilities")

            for i, label in enumerate(class_names):
                score = float(predictions[i])
                st.write(f"{label.title()}: {score * 100:.2f}%")
                st.progress(score)