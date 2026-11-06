"""
Go Ananas! - Plant Doctor
Interactive Streamlit Web Dashboard for Pineapple Disease Diagnostics
"""

import os
import json
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# -------------------------------------------------------------------------
# 1. Page Configuration & Styling
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Go Ananas! - Plant Doctor",
    page_icon="🍍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #D4A373;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6C757D;
        margin-bottom: 20px;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 2. Asset Loading (Cached for Performance)
# -------------------------------------------------------------------------
MODEL_PATH = "go_ananas_model.keras"
CLASS_NAMES_PATH = "class_names.json"

@st.cache_resource
def load_disease_model():
    """Loads the trained TensorFlow CNN model."""
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    """Loads the locked class mapping JSON file."""
    if not os.path.exists(CLASS_NAMES_PATH):
        return ["fruit_rot", "heart_rot", "healthy", "mealybug_wilt", "multicrown_disorder", "root_rot"]
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)

model = load_disease_model()
class_names = load_class_names()

# -------------------------------------------------------------------------
# 3. Sidebar Configuration & Info
# -------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/pineapple.png", width=80)
    st.title("Go Ananas!")
    st.markdown("### Plant Doctor")
    st.markdown("Automated crop diagnostics and pattern recognition system for *Ananas comosus*.")
    
    st.markdown("---")
    st.markdown("**Detectable Conditions:**")
    for name in class_names:
        st.markdown(f"- `{name.replace('_', ' ').title()}`")
    
    st.markdown("---")
    st.info("Tip: Upload clear, well-lit photos focusing on the affected crown, fruit, or root zone for high-accuracy inference.")

# -------------------------------------------------------------------------
# 4. Main Interface
# -------------------------------------------------------------------------
st.markdown('<p class="main-header">🍍 Go Ananas! - AI Plant Doctor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time agricultural pattern recognition for farmers and agricultural experts.</p>', unsafe_allow_html=True)

if model is None:
    st.error(f"⚠️ Model file `{MODEL_PATH}` not found in directory. Please run `train_model.py` first to generate your trained model weights.")
else:
    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload a pineapple plant or fruit image...", 
        type=["Upload Pineapple image here!"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1], gap="medium")

        with col1:
            st.markdown("### Uploaded Image")
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("### Diagnostic Results")
            with st.spinner("Analyzing plant pattern..."):
                img_resized = image.resize((224, 224))
                img_array = tf.keras.utils.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                
                predictions = model.predict(img_array)[0]
                predicted_index = np.argmax(predictions)
                confidence = float(predictions[predicted_index]) * 100
                predicted_class = class_names[predicted_index]

            # Display Top Prediction safely
            card_html = f"""
                <div class="prediction-card">
                    <h4 style="margin:0; color: #333;">Diagnosis: <b>{predicted_class.replace('_', ' ').title()}</b></h4>
                    <p style="margin:5px 0 0 0; font-size: 1.1rem; color: #2A9D8F;">Confidence Score: <b>{confidence:.2f}%</b></p>
                </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            st.markdown("#### Class Probability Breakdown")
            for idx, prob in enumerate(predictions):
                label_formatted = class_names[idx].replace('_', ' ').title()
                st.text(f"{label_formatted} ({prob*100:.1f}%)")
                st.progress(float(prob))

    else:
        st.markdown("""
            <div style="text-align: center; padding: 40px; border: 2px dashed #CED4DA; border-radius: 10px; color: #6C757D;">
                <h3>Awaiting Image Input</h3>
                <p>Upload a photo above to run automated disease classification.</p>
            </div>
        """, unsafe_allow_html=True)
