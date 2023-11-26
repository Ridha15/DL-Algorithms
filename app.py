import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO
import cv2
import requests

# Function to preprocess image for tumor detection
def preprocess_image(image_path, target_size=(180, 180)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img) / 255.0  # Normalize pixel values to between 0 and 1
    img = np.expand_dims(img, axis=0)
    return img


def load_model_from_google_drive(gdrive_file_id):
    gdrive_url = f"https://drive.google.com/uc?id={gdrive_file_id}"
    response = requests.get(gdrive_url)
    model_bytes = BytesIO(response.content)
    model = load_model(model_bytes)
    return model

# Google Drive File ID for your model file
model_gdrive_file_id = '1_mKNM-t6Do7fmXtzrsE9F5L9DliR1Yas'

# Load the model


# Load the model

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

    # Add a "Predict" button
    if st.button("Predict"):

        # Check if an image is uploaded before attempting to process it
        if uploaded_image is not None:
            model = load_model_from_google_drive(model_gdrive_file_id)
            img=cv2.imread(uploaded_image)
            img=Image.fromarray(img)
            img=img.resize((128,128))
            img=np.array(img)
            input_img = np.expand_dims(img, axis=0)
            res = model.predict(input_img)
            if res:
                print("Tumor Detected")
            else:
                print("No Tumor")
        else:
            st.warning("Please upload an image before clicking 'Predict'")
