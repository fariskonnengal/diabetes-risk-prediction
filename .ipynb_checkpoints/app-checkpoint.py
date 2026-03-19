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

# -----------------------
# Sidebar Inputs
# -----------------------
st.sidebar.header("Patient Health Information")

age = st.sidebar.slider("Age",18,90,30)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)

bmi = st.sidebar.number_input("BMI",15.0,50.0,25.0)

blood_pressure = st.sidebar.number_input(
    "Blood Pressure",80,200,120
)

fasting_glucose = st.sidebar.number_input(
    "Fasting Glucose Level",70,250,100
)

insulin = st.sidebar.number_input(
    "Insulin Level",2,300,80
)

hba1c = st.sidebar.number_input(
    "HbA1c Level",3.0,15.0,5.5
)

cholesterol = st.sidebar.number_input(
    "Cholesterol Level",100,350,180
)

triglycerides = st.sidebar.number_input(
    "Triglycerides Level",50,500,150
)

physical_activity = st.sidebar.selectbox(
    "Physical Activity Level",
    ["Low","Moderate","High"]
)

daily_calories = st.sidebar.number_input(
    "Daily Calorie Intake",1200,5000,2000
)

sugar_intake = st.sidebar.number_input(
    "Sugar Intake (grams/day)",0,200,50
)

sleep_hours = st.sidebar.slider(
    "Sleep Hours",3,12,7
)

stress_level = st.sidebar.slider(
    "Stress Level (1-10)",1,10,5
)

family_history = st.sidebar.selectbox(
    "Family History of Diabetes",
    ["Yes","No"]
)

waist = st.sidebar.number_input(
    "Waist Circumference (cm)",60,150,90
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


    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)

    result = target_encoder.inverse_transform(prediction)[0]

    st.subheader("Prediction Result")

    if result.lower() == "low":
        st.success("Low Diabetes Risk")

    elif result.lower() == "medium":
        st.warning("Medium Diabetes Risk")

    else:
        st.error("High Diabetes Risk")

    st.subheader("Prediction Probability")

    prob = probability[0]

    st.write("Low Risk Probability:", round(prob[0]*100,2), "%")
    st.write("Medium Risk Probability:", round(prob[1]*100,2), "%")
    st.write("High Risk Probability:", round(prob[2]*100,2), "%")

    st.bar_chart(prob)

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