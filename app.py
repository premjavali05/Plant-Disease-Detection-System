import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import requests
from io import BytesIO
import tempfile

# Function to load model from GitHub/cloud storage
@st.cache_resource  # This will cache the model loading
def load_model():
    # Option 1: If using GitHub releases
    try:
        # Replace this URL with your actual model file URL
        model_url = "YOUR_MODEL_GITHUB_RELEASE_URL"
        response = requests.get(model_url)
        response.raise_for_status()  # Raises an HTTPError if the status is 4xx, 5xx
        
        # Save the model temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
            tmp_file.write(response.content)
            temp_model_path = tmp_file.name
        
        # Load the model
        model = tf.keras.models.load_model(temp_model_path)
        
        # Clean up
        os.unlink(temp_model_path)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to preprocess the image
def preprocess_image(image_path):
    H, W, C = 224, 224, 3
    img = cv2.imread(image_path)
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype="float32") / 255.0
    img = img.reshape(1, H, W, C)
    return img

# Function to predict the class of the plant disease
def model_predict(image_path):
    model = load_model()
    if model is None:
        return None, None
        
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    prediction_index = np.argmax(predictions, axis=-1)[0]
    confidence = np.max(predictions) * 100
    return prediction_index, confidence

# Class names for the diseases
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Streamlit sidebar
st.sidebar.title("Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Main Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)
    st.image("index.jpg", use_container_width=True)
    
elif app_mode == "DISEASE RECOGNITION":
    st.header("Upload an Image for Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image:
        # Save uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(test_image.name)[1]) as tmp_file:
            tmp_file.write(test_image.getbuffer())
            temp_file_path = tmp_file.name
            
        if st.button("Show Image"):
            st.image(test_image, caption="Uploaded Image", use_container_width=True)
            
        if st.button("Predict"):
            with st.spinner("Analyzing the image..."):
                try:
                    result_index, confidence = model_predict(temp_file_path)
                    if result_index is not None:
                        st.success(f"Prediction: {CLASS_NAMES[result_index]}")
                        st.info(f"Confidence: {confidence:.2f}%")
                except Exception as e:
                    st.error("Error during prediction. Please try again.")
                finally:
                    # Clean up temporary file
                    os.remove(temp_file_path)
    else:
        st.warning("Please upload an image to proceed.")
