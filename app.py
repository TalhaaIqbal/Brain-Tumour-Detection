import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title="🧠 Brain Tumor Detection",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .result-section {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
    }
    .prediction-box {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #4B0082;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e6e6e6;
        height: 20px;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-fill {
        background-color: #4B0082;
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
    }
    .subtitle {
        color: #666;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# Class labels (customize if needed)
class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']

# App UI
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🧠 Brain Tumor Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an MRI scan to detect the presence of a brain tumor using our advanced deep learning model.</p>", unsafe_allow_html=True)

# File upload + display in styled white container
st.markdown("<div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"], help="Upload a brain MRI scan image (JPG, JPEG, or PNG format)")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Image', use_container_width=True)

    # Optional: add margin after image
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# Prediction Section
if uploaded_file is not None:
    # Preprocess the image
    img_resized = image.resize((128, 128))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)[0]
    print(prediction)
    predicted_class = class_names[np.argmax(prediction)]
    print(class_names)
    confidence = np.max(prediction) * 100

    # Results container
    st.markdown("<div class='result-section'>", unsafe_allow_html=True)

    st.markdown(f"""
        <div class='prediction-box'>
            <h3 style='text-align: center; color: #4B0082; margin-bottom: 1rem;'>Analysis Results</h3>
            <h4 style='text-align: center; color: #2E7D32;'>🧬 Prediction: <b>{predicted_class.upper()}</b></h4>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: center; margin-top: 1rem;'>Model Confidence</h4>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class='confidence-bar'>
            <div class='confidence-fill' style='width: {confidence}%;'></div>
        </div>
        <p style='text-align: center; font-size: 1.2rem;'><b>{confidence:.2f}%</b> confidence in prediction</p>
    """, unsafe_allow_html=True)

    if predicted_class == 'no tumor':
        st.success("✅ No tumor detected in the MRI scan. However, please consult with a medical professional for a complete diagnosis.")
    else:
        st.warning(f"⚠️ A {predicted_class} tumor has been detected. Please consult with a medical professional immediately for proper diagnosis and treatment.")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer note
st.markdown("""
    <div style='text-align: center; margin-top: 2rem; color: #666; font-size: 0.9rem;'>
        <p>Note: This is an AI-assisted tool and should not replace professional medical advice.</p>
    </div>
""", unsafe_allow_html=True)
