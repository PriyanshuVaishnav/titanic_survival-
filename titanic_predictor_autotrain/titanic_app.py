import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "model.pkl"
DATA_PATH = "titanic.csv"

# Auto-train model if not exists
def train_model():
    df = pd.read_csv(DATA_PATH)
    df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].dropna()
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Training model now...")
    train_model()

# Load model
model = pickle.load(open(MODEL_PATH, "rb"))

st.title("🚢 Titanic Survival Predictor")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)

sex = 1 if sex == "male" else 0
features = np.array([[pclass, sex, age, sibsp, parch, fare]])

if st.button("Predict Survival"):
    result = model.predict(features)
    if result[0] == 1:
        st.success("🎉 The passenger would have **survived**.")
    else:
        st.error("💀 The passenger would have **not survived**.")
