import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

st.title("🍷 Wine Quality Prediction App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("winequality-red 2.csv")
    return df

df = load_data()

# Display data overview
st.subheader("Dataset Overview")
st.write(df.head())
st.bar_chart(df['quality'].value_counts().sort_index())

# Features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Split the dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
@st.cache_resource
def train_model():
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    return clf

model = train_model()

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Accuracy: **{acc:.2f}**")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# User Input Section
st.subheader("Predict Wine Quality")
user_input = {}

cols = st.columns(3)
for i, col in enumerate(X.columns):
    user_input[col] = cols[i % 3].slider(
        col,
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].mean())
    )

# Predict Button
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    quality_labels = model.classes_

    st.success(f"🎯 Predicted Wine Quality: **{prediction}**")

    # Show input values
    st.subheader("Your Input Values")
    st.write(input_df)

    # Show prediction probabilities
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({
        'Quality': quality_labels,
        'Probability': probabilities
    })
    st.bar_chart(prob_df.set_index("Quality"))
