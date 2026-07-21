import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the full path to the model
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.keras")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Get IMDB word index
word_index = imdb.get_word_index()

# Reverse word index (word -> index)
word_to_index = {k: (v + 3) for k, v in word_index.items()}
word_to_index["<PAD>"] = 0
word_to_index["<START>"] = 1
word_to_index["<UNK>"] = 2
word_to_index["<UNUSED>"] = 3

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review below.")

review = st.text_area("Movie Review")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Convert text into sequence
        words = review.lower().split()
        sequence = [word_to_index.get(word, 2) for word in words]

        # Pad sequence
        padded = pad_sequences([sequence], maxlen=200)

        # Predict
        prediction = model.predict(padded, verbose=0)

        if prediction[0][0] >= 0.5:
            st.success("😊 Positive Review")
            st.write(f"Confidence: {prediction[0][0]:.2%}")
        else:
            st.error("😞 Negative Review")
            st.write(f"Confidence: {(1 - prediction[0][0]):.2%}")