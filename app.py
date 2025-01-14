import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import requests
import tempfile
from PIL import Image

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

@st.cache_resource
def load_model():
    """Load model from GitHub release"""
    try:
        # Replace this URL with your actual GitHub release URL
        MODEL_URL = "https://github.com/user-attachments/files/18413817/MobileNetV2_plantdiseases_model.zip"
        
        with st.spinner("Loading model... This may take a minute."):
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            
            # Save model to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            # Load model with custom configuration
            model = tf.keras.models.load_model(
                temp_file_path,
                compile=False,
                options=tf.saved_model.LoadOptions(
                    experimental_io_device='/job:localhost'
                )
            )
            
            # Clean up
            os.unlink(temp_file_path)
            
            # Basic compilation
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Convert to array and preprocess
        img_array = np.array(image)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# Class names
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

# Streamlit UI
st.sidebar.title("Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)
    st.image("index.jpg", use_container_width=True)

elif app_mode == "DISEASE RECOGNITION":
    st.header("Upload an Image for Disease Recognition")
    uploaded_file = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Make prediction
        with col2:
            if st.button("Predict"):
                with st.spinner("Analyzing..."):
                    # Load model if not already loaded
                    model = load_model()
                    
                    if model is not None:
                        # Preprocess image
                        processed_image = preprocess_image(image)
                        
                        if processed_image is not None:
                            try:
                                # Make prediction
                                predictions = model.predict(processed_image, verbose=0)
                                prediction_idx = np.argmax(predictions[0])
                                confidence = float(predictions[0][prediction_idx] * 100)
                                
                                # Display results
                                st.success(f"Prediction: {CLASS_NAMES[prediction_idx]}")
                                st.progress(confidence / 100)
                                st.info(f"Confidence: {confidence:.2f}%")
                            except Exception as e:
                                st.error("Error making prediction. Please try again.")
