import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained LSTM model
model = load_model("models/lstm_model.h5")

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

# Streamlit app
st.title("Movie Review Sentiment Analysis")

# User input
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
        st.warning("Please enter a movie review.")

