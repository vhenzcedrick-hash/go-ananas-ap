import os
import json
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# -------------------------------------------------------------------------
# 1. Page Configuration
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Go Ananas! - AI Plant Doctor",
    page_icon="🍍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Theme Matching UI)
st.markdown("""
    <style>
    /* Dark Theme Base Adjustments */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Main Headers */
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #B2B9C0;
        margin-bottom: 25px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* Info Card in Sidebar */
    .sidebar-info-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        color: #94A3B8;
        font-size: 0.9rem;
        line-height: 1.4;
        margin-bottom: 20px;
    }
    
    /* Class List Bullets */
    .condition-item {
        margin-bottom: 8px;
        font-size: 0.95rem;
        color: #E2E8F0;
    }

    /* Info Banner */
    .info-banner {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 6px;
        color: #93C5FD;
        font-size: 0.95rem;
        margin-top: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #1E293B;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 2. Asset Loading
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
        return ["fruit_rot", "healthy", "heart_rot", "mealybug_wilt", "multicrown_disorder", "root_rot"]
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)

model = load_disease_model()
class_names = load_class_names()

# Icon mapping for conditions
CONDITION_ICONS = {
    "fruit_rot": "🔴",
    "healthy": "🟢",
    "heart_rot": "🤎",
    "mealybug_wilt": "🔴",
    "multicrown_disorder": "👑",
    "root_rot": "🤎"
}

# -------------------------------------------------------------------------
# 3. Sidebar Configuration
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Go Ananas! 🍍")
    st.markdown("##### **Pineapple Disease Detector**")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Blue info card box
    st.markdown("""
        <div class="sidebar-info-card">
            Upload a photo of a pineapple leaf, crown, or fruit section to diagnose health conditions instantly.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Supported Conditions:**")
    st.markdown("<br>", unsafe_allow_html=True)
    
    for name in class_names:
        icon = CONDITION_ICONS.get(name, "•")
        formatted_name = name.replace('_', ' ').title()
        st.markdown(f"<div class='condition-item'>{icon} {formatted_name}</div>", unsafe_allow_html=True)
        
    st.markdown("<br>" * 4, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("⚙️ **Demo Controls**")

# -------------------------------------------------------------------------
# 4. Main Interface
# -------------------------------------------------------------------------
st.markdown('<div class="main-title">🍍 Go Ananas! Disease Diagnostic Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Welcome to your smart agricultural assistant. Upload an image below to run pattern recognition.</div>', unsafe_allow_html=True)
st.markdown("---")

if model is None:
    st.error(f"⚠️ Model file `{MODEL_PATH}` not found in directory. Please run `train_model.py` first to generate your trained model weights.")
else:
    # File Uploader with correct valid image extensions
    uploaded_file = st.file_uploader(
        "Choose a pineapple image (.jpg, .png, .jpeg)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1], gap="large")

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

            card_html = f"""
                <div class="prediction-card">
                    <h4 style="margin:0; color: #FFFFFF;">Diagnosis: <b>{predicted_class.replace('_', ' ').title()}</b></h4>
                    <p style="margin:8px 0 0 0; font-size: 1.1rem; color: #38BDF8;">Confidence Score: <b>{confidence:.2f}%</b></p>
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
            <div class="info-banner">
                👉 Upload an image using the box above or via the sidebar to begin analysis.
            </div>
        """, unsafe_allow_html=True)
