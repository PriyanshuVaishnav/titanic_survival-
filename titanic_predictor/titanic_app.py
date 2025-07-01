import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🚢 Titanic Survival Predictor")

# User input
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)

# Encode input
sex = 1 if sex == "male" else 0
features = np.array([[pclass, sex, age, sibsp, parch, fare]])

# Predict
if st.button("Predict Survival"):
    result = model.predict(features)
    if result[0] == 1:
        st.success("🎉 The passenger would have **survived**.")
    else:
        st.error("💀 The passenger would have **not survived**.")
