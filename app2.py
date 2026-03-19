import streamlit as st
import numpy as np
import joblib

# -----------------------
# Load saved files
# -----------------------
model = joblib.load("log_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Risk Prediction System")

st.write(
"""
This ML system predicts **diabetes risk category** based on health and lifestyle factors.
Model built using **Logistic Regression with Hyperparameter Tuning**.
"""
)

st.divider()
import pandas as pd
feature_names = [
'age',
'gender',
'bmi',
'blood_pressure',
'fasting_glucose_level',
'insulin_level',
'HbA1c_level',
'cholesterol_level',
'triglycerides_level',
'physical_activity_level',
'daily_calorie_intake',
'sugar_intake_grams_per_day',
'sleep_hours',
'stress_level',
'family_history_diabetes',
'waist_circumference_cm'
]

importance = model.coef_[0]

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

importance_df = importance_df.sort_values(
    by='Importance',
    key=abs,
    ascending=False
)

# -----------------------
st.header("Enter Patient Information")

col1, col2 = st.columns(2)

# -----------------------
# Column 1 : Patient Details
# -----------------------

with col1:
    st.subheader("Patient Details")

    age = st.slider("Age",18,90,30)

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=15.0,
        max_value=50.0,
        value=25.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=80,
        max_value=200,
        value=120
    )

    fasting_glucose = st.number_input(
        "Fasting Glucose Level",
        min_value=70,
        max_value=250,
        value=100
    )

    insulin = st.number_input(
        "Insulin Level",
        min_value=2,
        max_value=300,
        value=80
    )

    hba1c = st.number_input(
        "HbA1c Level",
        min_value=3.0,
        max_value=15.0,
        value=5.5
    )

    cholesterol = st.number_input(
        "Cholesterol Level",
        min_value=100,
        max_value=350,
        value=180
    )

    triglycerides = st.number_input(
        "Triglycerides Level",
        min_value=50,
        max_value=500,
        value=150
    )

# -----------------------
# Column 2 : Lifestyle Factors
# -----------------------

with col2:
    st.subheader("Lifestyle Factors")

    physical_activity = st.selectbox(
        "Physical Activity Level",
        ["Low","Moderate","High"]
    )

    daily_calories = st.number_input(
        "Daily Calorie Intake",
        min_value=1200,
        max_value=5000,
        value=2000
    )

    sugar_intake = st.number_input(
        "Sugar Intake (grams/day)",
        min_value=0,
        max_value=200,
        value=50
    )

    sleep_hours = st.slider(
        "Sleep Hours",
        min_value=3,
        max_value=12,
        value=7
    )

    stress_level = st.slider(
        "Stress Level",
        min_value=1,
        max_value=10,
        value=5
    )

    family_history = st.selectbox(
        "Family History of Diabetes",
        ["Yes","No"]
    )

    waist = st.number_input(
        "Waist Circumference (cm)",
        min_value=60,
        max_value=150,
        value=90
    )

# -----------------------
# Encoding categorical features
# -----------------------
gender = label_encoders["gender"].transform([gender])[0]

physical_activity = label_encoders[
    "physical_activity_level"
].transform([physical_activity])[0]

family_history = label_encoders[
    "family_history_diabetes"
].transform([family_history])[0]

# -----------------------
# Create feature array
# -----------------------
features = np.array([[

age,
gender,
bmi,
blood_pressure,
fasting_glucose,
insulin,
hba1c,
cholesterol,
triglycerides,
physical_activity,
daily_calories,
sugar_intake,
sleep_hours,
stress_level,
family_history,
waist

]])

# scale features
features_scaled = scaler.transform(features)

# -----------------------
# Prediction Button
# -----------------------
if st.button("🔍 Predict Diabetes Risk"):

    # prediction
    prediction = model.predict(features_scaled)

    # probability
    probability = model.predict_proba(features_scaled)

    # decode prediction label
    result = target_encoder.inverse_transform(prediction)[0]

    st.subheader(f"Prediction Result: {result} Diabetes Risk")

    # get probability values
    prob = probability[0]

    classes = target_encoder.classes_

    st.subheader("Prediction Probability")

    for i in range(len(classes)):
        st.write(f"{classes[i]} Risk Probability: {round(prob[i]*100,2)} %")
    st.subheader("Feature Importance")

    st.bar_chart(
        importance_df.set_index("Feature")
    )
st.divider()

st.markdown("""
### Model Information

Algorithm: **Logistic Regression**

Accuracy: **94.5%**

Models Compared:
- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- XGBoost
- AdaBoost
- Gradient Boosting
- Gaussian Naive Bayes

Best model selected after **Hyperparameter Tuning**.
""")