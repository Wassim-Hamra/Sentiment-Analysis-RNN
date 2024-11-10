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
import streamlit.components.v1 as components

particles_js_code = """
<div id="particles-js"></div>
<style>
    #particles-js {
        position: fixed;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        z-index: -1;  /* Ensure particles are behind other elements */
    }
</style>
<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
<script>
    particlesJS("particles-js", {
        "particles": {
            "number": {
                "value": 80,
                "density": {
                    "enable": true,
                    "value_area": 800
                }
            },
            "color": {
                "value": "#ffffff"
            },
            "shape": {
                "type": "circle",
                "stroke": {
                    "width": 0,
                    "color": "#000000"
                },
                "polygon": {
                    "nb_sides": 5
                },
                "image": {
                    "src": "img/github.svg",
                    "width": 100,
                    "height": 100
                }
            },
            "opacity": {
                "value": 0.5,
                "random": false,
                "anim": {
                    "enable": false,
                    "speed": 1,
                    "opacity_min": 0.1,
                    "sync": false
                }
            },
            "size": {
                "value": 3,
                "random": true,
                "anim": {
                    "enable": false,
                    "speed": 40,
                    "size_min": 0.1,
                    "sync": false
                }
            },
            "line_linked": {
                "enable": true,
                "distance": 150,
                "color": "#ffffff",
                "opacity": 0.4,
                "width": 1
            },
            "move": {
                "enable": true,
                "speed": 6,
                "direction": "none",
                "random": false,
                "straight": false,
                "out_mode": "out",
                "bounce": false,
                "attract": {
                    "enable": false,
                    "rotateX": 600,
                    "rotateY": 1200
                }
            }
        },
        "interactivity": {
            "detect_on": "canvas",
            "events": {
                "onhover": {
                    "enable": true,
                    "mode": "repulse"
                },
                "onclick": {
                    "enable": true,
                    "mode": "push"
                },
                "resize": true
            },
            "modes": {
                "grab": {
                    "distance": 400,
                    "line_linked": {
                        "opacity": 1
                    }
                },
                "bubble": {
                    "distance": 400,
                    "size": 40,
                    "duration": 2,
                    "opacity": 8,
                    "speed": 3
                },
                "repulse": {
                    "distance": 200,
                    "duration": 0.4
                },
                "push": {
                    "particles_nb": 4
                },
                "remove": {
                    "particles_nb": 2
                }
            }
        },
        "retina_detect": true
    });
</script>
"""

# Display particles.js background
components.html(particles_js_code, height=500, width=500)

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
background_color = "background-color:rgba(79, 227, 94, 0.8);" if prediction_proba > 0.5 else "background-color:rgba(250, 87, 87, 0.8)"

st.markdown(
    f"""
    <style>
    .box {{
        padding: 1em;
        margin: 1em 0;
        {background_color}
        border-radius: 5px;
        border: 1px solid #d3d3d3;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Display text inside the box

if prediction_proba > 0.5:
    st.markdown('<div class="box"><h4 style="color:#0d0f0c;">Client is likely to churn 😢</h4></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="box"><h4 style="color:#0d0f0c;">Client is not likely to churn 😌</h4></div>', unsafe_allow_html=True)

st.write(f"Churn Probability: {round(prediction_proba * 100,1)}%")

