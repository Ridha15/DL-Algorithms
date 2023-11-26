import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import cv2

from transformers import AutoModelForImageClassification, AutoImageFeatureExtractor

import requests


# Function to load the model
def load_model():
    model_name = "Ridha15/cnn_model"  # Replace with your Hugging Face repo name
    model = AutoModelForImageClassification.from_pretrained(model_name)
    feature_extractor = AutoImageFeatureExtractor.from_pretrained(model_name)
    return model, feature_extractor


model, feature_extractor = load_model()

# User input for image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if st.button("Predict"):
    if uploaded_image:
        # Load and preprocess the image
        image = Image.open(uploaded_image)
        inputs = feature_extractor(images=image, return_tensors="pt")
        
        # Make predictions
        outputs = model(**inputs)
        predicted_class = np.argmax(outputs.logits[0])
        
        # Display the result
        st.success(f"The predicted class is: {predicted_class}")
    else:
        st.warning("Please upload an image")

# Main content
st.title("Deep Learning Algorithms")

selected_option = st.radio("Choose an option", ["Tumor Detection", "Sentiment Classification"])

# Upload image if "Tumor Detection" is selected
if selected_option == "Tumor Detection":
    st.title("Tumor Detection using CNN")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Display the uploaded image
    if uploaded_image is not None:
        
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True,width=10)



    if st.button("Predict"):
        if uploaded_image:
            # Load and preprocess the image
            image = Image.open(uploaded_image)
            inputs = feature_extractor(images=image, return_tensors="pt")
            
            # Make predictions
            outputs = model(**inputs)
            predicted_class = np.argmax(outputs.logits[0])
            
            # Display the result
            st.success(f"The predicted class is: {predicted_class}")
        else:
            st.warning("Please upload an image")
