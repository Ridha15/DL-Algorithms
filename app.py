import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained DNN model
model = load_model("models/dnn_model.h5")

# Streamlit UI
st.title("Iris Flower Classification App")

# Input form for user to input feature values
sepal_length = st.slider("Sepal Length", 0.0, 10.0, 5.0)
sepal_width = st.slider("Sepal Width", 0.0, 10.0, 5.0)
petal_length = st.slider("Petal Length", 0.0, 10.0, 5.0)
petal_width = st.slider("Petal Width", 0.0, 10.0, 5.0)

# Make a prediction when the user clicks the "Predict" button
if st.button("Predict"):
    # Prepare the input features
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make a prediction
    prediction = model.predict(input_features)

    # Display the predicted class
    predicted_class = np.argmax(prediction)
    st.write(f"Predicted Class: {predicted_class}")
