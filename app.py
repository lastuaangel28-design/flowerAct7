import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Flower AI | Sunflower vs Tulip",
    page_icon="🌻",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. MODEL LOADING (Cached)
# ==========================================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("MobileNetV2.h5")
        return model
    except OSError:
        st.error("❌ Model file 'MobileNetV2.h5' not found.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# ==========================================
# 3. HEADER & UI
# ==========================================
st.title("🌻 Sunflower vs. Tulip Classifier")
st.markdown("""
    <style>
    h1 {color: #FFA500; text-align: center;}
    .stButton>button {width: 100%;}
    </style>
    """, unsafe_allow_html=True
)

st.write("Upload an image to identify if it is a **Sunflower** or a **Tulip**.")

# ==========================================
# 4. FILE UPLOAD
# ==========================================
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
if uploaded_file is not None and model is not None:
    
    # --- Layout ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.write("### Analysis")
        
        # --- Preprocessing (CRITICAL FIX) ---
        # 1. Resize to 224x224
        # 2. Convert to array
        # 3. Apply MobileNetV2 preprocessing (Scales to -1 to 1)
        #    *This fixes the low confidence issue*
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = preprocess_input(img_array) 
        img_array = np.expand_dims(img_array, axis=0)

        # --- Predict ---
        with st.spinner("Analyzing..."):
            # Returns a value between 0.0 and 1.0
            prediction = float(model.predict(img_array, verbose=0)[0][0])

        # --- Interpret Results ---
        # Folders are usually sorted alphabetically: 
        # 'sunflower' (Index 0) < 'tulip' (Index 1)
        
        CLASS_NAMES = ["Sunflower", "Tulip"]
        THRESHOLD = 0.5

        if prediction < THRESHOLD:
            predicted_class = CLASS_NAMES[0] # Sunflower
            confidence_score = (1 - prediction) * 100
            emoji = "🌻"
            color = "🟡" # Yellowish theme
        else:
            predicted_class = CLASS_NAMES[1] # Tulip
            confidence_score = prediction * 100
            emoji = "🌷"
            color = "🔴"

        # --- Display Output ---
        st.metric(label="Prediction", value=f"{emoji} {predicted_class}")
        
        # Confidence Bar
        st.progress(int(confidence_score))
        st.write(f"**Confidence:** {confidence_score:.2f}%")

        # Feedback Logic
        if confidence_score > 85:
            st.success("✅ High Confidence Match")
        elif confidence_score > 65:
            st.info("ℹ️ Good Match")
        else:
            st.warning("⚠️ Low Confidence (Ensure image is clear)")

    # --- Debugging Section ---
    with st.expander("🔧 Technical Details"):
        st.write(f"**Raw Score:** {prediction:.5f}")
        st.write(f"**Threshold:** 0.5")
        st.write(f"**Detected Class:** {predicted_class}")
        st.caption("If confidence is low, ensure the model was trained with `preprocess_input`.")

else:
    st.info("👆 Waiting for image upload...")
