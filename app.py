import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

col1, col2, col3 = st.columns([1,1])

with col1:
    st.button('Sentiment Classification')
with col2:
    st.button('Tumor Detection')

tumor_detection_selected = st.button("Tumor detection")

# Button for Sentiment Classification
sentiment_classification_selected = st.button("Sentiment Classification")

# Check which button is selected
if tumor_detection_selected:
    # Add Tumor Detection functionality here
    pass
elif sentiment_classification_selected:
    model_type = st.radio("Select a Model", ["Perceptron", "Backpropagation", "DNN", "RNN", "LSTM"])

    # Load the selected model
    if model_type == "Perceptron":
        model_path = "models/perceptron_model.h5"  # Replace with your actual path
    elif model_type == "Backpropagation":
        model_path = "models/backpropagation_model.h5"  # Replace with your actual path
    elif model_type == "DNN":
        model_path = "models/dnn_model.h5"  # Replace with your actual path
    elif model_type == "RNN":
        model_path = "models/rnn_model.h5"  # Replace with your actual path
    elif model_type == "LSTM":
        model_path = "models/lstm_model.h5"  # Replace with your actual path

    # Load the selected model
    model = load_model(model_path)

    # Load the IMDB dataset and obtain the word index
    imdb_dataset = tf.keras.datasets.imdb
    (_, _), (x_test, _) = imdb_dataset.load_data()
    word_index = imdb_dataset.get_word_index()

    # Set the maximum review length
    max_review_length = 500

    # Reverse the word index to map indices back to words
    reverse_word_index = {value: key for key, value in word_index.items()}

    # Function to preprocess new reviews
    def preprocess_new_review(review_text):
        words = review_text.lower().split()
        indices = [word_index.get(word, 0) for word in words]
        padded_sequence = pad_sequences([indices], maxlen=max_review_length)
        return padded_sequence

    # Streamlit app title for sentiment classification
    st.title("Sentiment Classification")

    # User input for sentiment classification
    user_review = st.text_area("Enter your movie review:")

    if st.button("Predict Sentiment"):
        if user_review:
            # Preprocess the user input
            processed_input = preprocess_new_review(user_review)

            # Make predictions
            prediction = model.predict(processed_input)

            # Display the result
            sentiment = "Positive" if prediction > 0.5 else "Negative"
            st.success(f"The predicted sentiment is: {sentiment} (Confidence: {prediction[0][0]:.2f})")
        else:
            st.warning("Please enter a movie review")

