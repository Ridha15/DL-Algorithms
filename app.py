import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

st.title("Deep Learning Algorithms")

# Layout for buttons
button1 = st.button("Sentiment Classification")
button2 = st.button("Tumor Detection")

# Function to preprocess image for tumor detection
def make_prediction(img,model):
    img=cv2.imread(img)
    img=Image.fromarray(img)
    img=img.resize((180,180))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img)
    if res[0][0] < 0.5: 
        st.write("Tumor Detected")
    else:
        st.write("No Tumor")
# Main content based on button clicks
if button1:
    st.title("Sentiment Classification")
    # Add your code for sentiment classification here

if button2:
    st.title("Tumor Detection")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        model_cnn = load_model("models/cnn_model.h5")

        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        # Add debugging statements
        st.write("Image displayed successfully")

        # Add a "Predict" button
        if st.button("Predict"):
            make_prediction(uploaded_image,model_cnn)


if button1:
    st.title("Sentiment Classification")
    model_type = st.radio("Select a Model", ["Perceptron", "Backpropagation", "DNN", "RNN", "LSTM","GRU"])

    # Load the selected model
    if model_type == "Perceptron":
        model_path = "models/perceptron_model.h5"
        data = st.radio("What type of text you want to try on?",["SMS","Movie Review"])
        if data == "Movie Review":
            text = st.text_input("Enter your text")
            model = load_model(model_path)
            word_index = tf.keras.datasets.imdb.get_word_index()
            max_len = 100
            tokens = [word_index[word] if word in word_index and word_index[word] < 10000 else 0 for word in text.split()]
            padded_sequence = pad_sequences([tokens], maxlen=max_len)
            prediction = model.predict(padded_sequence)
            st.write(f"Prediction: {prediction}")

    elif model_type == "Backpropagation":
        model_path = "models/backprop_model.pkl"  # Replace with your actual path
    elif model_type == "DNN":
        model_path = "models/dnn_model.h5"
    elif model_type == "RNN":
        model_path = "models/rnn_model.h5"  # Replace with your actual path
        model = load_model(model_path)
        user_review = st.text_area("Enter your movie review:") 
        if st.button("Predict Sentiment"):
            if user_review:
                dataset = pd.read_csv('RNN\spam.csv')
                dataset['label'] = dataset['label'].map( {'spam': 1, 'ham': 0} )
                X = dataset['message'].values
                y = dataset['label'].values
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                tokeniser = tf.keras.preprocessing.text.Tokenizer()
                tokeniser.fit_on_texts(X_train)
                encoded_train = tokeniser.texts_to_sequences(X_train)
                encoded_test = tokeniser.texts_to_sequences(X_test)
                encoded_sms = tokeniser.texts_to_sequences(user_review)
                max_length = 10
                padded_sms = tf.keras.preprocessing.sequence.pad_sequences(encoded_sms, maxlen=max_length, padding='post')
                preds = (model.predict(padded_sms) > 0.5).astype("int32")
                st.success(preds)

    elif model_type == "LSTM":
        model_path = "models/lstm_model.h5"  # Replace with your actual path
    elif model_type == "GRU":
        model_path = "models/gru_model.h5"


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

