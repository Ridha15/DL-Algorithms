
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# ...

# Function to preprocess text input
def preprocess_text(text, tokenizer, max_length):
    # Tokenize the text
    tokenized_text = tokenizer.texts_to_sequences([text])

    # Pad the sequences
    padded_text = sequence.pad_sequences(tokenized_text, maxlen=max_length)

    return padded_text

# ...

# Load the trained LSTM model
model_path = "models/lstm_model.h5"
lstm_model = load_model(model_path)

# Tokenize the input text during app initialization
top_words = 500
tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(["placeholder"])  # Placeholder text for initialization

# ...

# Get user input
user_input = st.text_area("Enter a movie review:")

if st.button("Predict"):
    if user_input:
        # Preprocess the input
        processed_input = preprocess_text(user_input, tokenizer, max_review_length)

        # Make a prediction
        prediction = lstm_model.predict(processed_input)[0, 0]

        # Display the result
        st.write(f"Sentiment: {'Positive' if prediction > 0.5 else 'Negative'}")
        st.write(f"Confidence: {prediction:.2f}")

# ...
