# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('sentiment_rnn.keras')


# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st

## streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')
if st.button('Classify'):

    preprocessed_input = preprocess_text(user_input)

    ## MAke prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive ✅' if prediction[0][0] > 0.5 else 'Negative ❌'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    if sentiment == 'Negative ❌':
        st.write(f'Prediction Score: {round(100 - prediction[0][0] * 100, 1)}%')
        st.write('⚠️ Note: This model has been developed for learning purposes and may not always provide accurate results. It is primarily trained on lengthy reviews, which might affect its performance when analyzing shorter ones.')

    else:
        st.write(f'Prediction Score: {round(prediction[0][0] * 100, 1)}%')
        st.write('⚠️ Note: This model is made for learning and may not always provide accurate results. It is primarily trained on long reviews, which might affect its performance when analyzing shorter ones.')
else:
    st.write('Please enter a movie review.')
