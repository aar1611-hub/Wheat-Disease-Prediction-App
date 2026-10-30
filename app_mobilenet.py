
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="Wheat Disease Prediction",
    page_icon="🌾",
    layout="centered"
)

MODEL_PATH = "wheat_mobilenetv2.keras"
CLASS_PATH = "mobilenet_class_names.json"
IMAGE_SIZE = (160, 160)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

st.title("🌾 Wheat Disease Prediction")
st.write(
    "Upload a wheat image to receive a preliminary "
    "Healthy Wheat or Sick Wheat classification."
)

model = load_model()
class_names = load_classes()

uploaded_file = st.file_uploader(
    "Choose a wheat image",
    type=["jpg", "jpeg", "png","webp","avif"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", width="stretch")

        resized_image = image.resize(IMAGE_SIZE)
        image_array = np.array(resized_image, dtype=np.float32)
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)

        if st.button("Predict", type="primary"):
            with st.spinner("Analyzing image..."):
                probabilities = model.predict(
                    image_array,
                    verbose=0
                )[0]

            predicted_index = int(np.argmax(probabilities))
            predicted_class = class_names[predicted_index]
            confidence = float(probabilities[predicted_index]) * 100

            st.subheader("Prediction")
            st.write(f"**Result:** {predicted_class}")
            st.write(f"**Model confidence:** {confidence:.2f}%")

            st.write("Class probabilities")
            for name, probability in zip(class_names, probabilities):
                st.write(f"{name}: {probability * 100:.2f}%")

            st.warning(
                "This is an experimental image classification result, "
                "not a confirmed plant diagnosis. Check the plant "
                "carefully or consult an agricultural expert."
            )

    except Exception as error:
        st.error(f"Unable to process this image: {error}")
