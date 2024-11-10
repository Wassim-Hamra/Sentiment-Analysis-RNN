import numpy as np
import pandas as pd
import pickle as pkl
import streamlit as st
from tensorflow.keras.models import load_model

model = load_model("model.h5")
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder = pkl.load(file)
with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder = pkl.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pkl.load(file)

## Streamlit App
st.title("Customer Churn Prediction")

# User Input
geography = st.selectbox("Geography", onehot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Num Of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", ["No", "Yes"])
is_active = st.selectbox("Is Active Member", ["No", "Yes"])

input_df = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender],
    "Age": [age],
    "Balance": [balance],
    "EstimatedSalary": [estimated_salary],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [0 if has_cr_card == "No" else 1],
    "IsActiveMember": [0 if is_active == "No" else 1],
    "Tenure": [tenure],
})

geo_encoded = onehot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(["Geography"]))
input_df['Gender'] = label_encoder.transform(input_df['Gender'])
input_data = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)
input_data = input_data[
    ["CreditScore", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
     "EstimatedSalary", "Geography_France", "Geography_Germany", "Geography_Spain"]]
scaled_data = scaler.transform(input_data)

prediction = model.predict(scaled_data)
prediction_proba = prediction[0][0]
background_color = "background-color: #cdf5a2;" if prediction_proba > 0.5 else "background-color: #f5a9a2;"

st.markdown(
    """
    <style>
    .box {
        padding: 1em;
        margin: 1em 0;
        background-color:red;
        border-radius: 5px;
        border: 1px solid #d3d3d3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display text inside the box

if prediction_proba > 0.5:
    st.markdown('<div class="box"><h3>Client is likely to churn 😢</h3></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="box"><h3>Client is not likely to churn 😌</h3></div>', unsafe_allow_html=True)

st.write(f"Churn Probability: {round(prediction_proba * 100,1)}%")
