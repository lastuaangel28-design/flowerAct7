import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------------------
# PAGE SETTINGS
# ------------------------------------
st.set_page_config(
    page_title="Flower Classification",
    page_icon="🌻",
    layout="centered"
)

# ------------------------------------
# LOAD MODEL
# ------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("MobileNetV2.h5")

model = load_model()

# ------------------------------------
# CLASS NAMES
# IMPORTANT:
# Alphabetical Order -> 0: Sunflower, 1: Tulip
# ------------------------------------
CLASS_NAMES = [
    "Sunflower",
    "Tulip"
]

# ------------------------------------
# TITLE
# ------------------------------------
st.title("🌸 Flower Classification System")

st.write(
    "Upload a flower image and the AI will identify whether it is a Sunflower or Tulip."
)

# ------------------------------------
# FILE UPLOAD
# ------------------------------------
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

# ------------------------------------
# PREDICTION
# ------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize image
    image_resized = image.resize((224, 224))

    img_array = np.array(image_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = float(model.predict(img_array, verbose=0)[0][0])

    # --------------------------------
    # CLASSIFICATION
    # --------------------------------
    predicted_index = int(prediction >= 0.5)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = prediction * 100 if predicted_index == 1 else (1 - prediction) * 100

    # Safety clamp
    confidence = max(
        0.0,
        min(confidence, 100.0)
    )

    # --------------------------------
    # DISPLAY RESULTS
    # --------------------------------
    st.markdown("## Prediction")

    st.success(predicted_class)

    st.markdown("## Confidence")

    st.progress(int(confidence))

    st.write(
        f"**{confidence:.2f}%**"
    )

    # --------------------------------
    # CONFIDENCE MESSAGE
    # --------------------------------
    if confidence >= 90:
        st.success(
            "Very High Confidence"
        )
    elif confidence >= 80:
        st.info(
            "High Confidence"
        )
    elif confidence >= 70:
        st.warning(
            "Moderate Confidence"
        )
    else:
        st.error(
            "Low Confidence"
        )

    # Debug Section
    with st.expander("Model Details"):
        st.write(
            f"Raw Prediction Value: {prediction:.4f}"
        )
        st.write(
            "Model Used: MobileNetV2"
        )
        st.write(
            "Test Accuracy: 96.80%"
        )
