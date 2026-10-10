import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configure page layout to be wider for side-by-side columns
st.set_page_config(layout="wide")

# Load artifacts using joblib
@st.cache_resource
def load_artifacts():
    lr_model = joblib.load("logistic_regression_model.joblib")
    rf_model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_names = joblib.load("feature_names.joblib")
    return lr_model, rf_model, scaler, feature_names

lr_model, rf_model, scaler, feature_names = load_artifacts()

# App UI
st.title("Telecommunications Customer Churn Predictor")
st.write("This application predicts the likelihood of a customer leaving based on their demographics and service usage.")

model_choice = st.selectbox("Select Prediction Model:", ["Logistic Regression", "Random Forest"])

st.markdown("---")

# ----------------- SECTION 1: Demographics -----------------
st.subheader("Customer Demographics")
demo_col1, demo_col2, demo_col3, demo_col4 = st.columns(4)

with demo_col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
with demo_col2:
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
with demo_col3:
    partner = st.selectbox("Partner", ["Yes", "No"])
with demo_col4:
    dependents = st.selectbox("Dependents", ["Yes", "No"])


# ----------------- SECTION 2: Account & Services -----------------
st.subheader("Customer Account & Services")
serv_col1, serv_col2, serv_col3 = st.columns(3)

with serv_col1:
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with serv_col2:
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

with serv_col3:
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])


# ----------------- SECTION 3: Billing -----------------
st.subheader("Billing Information")
bill_col1, bill_col2, bill_col3 = st.columns(3)

with bill_col1:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

with bill_col2:
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)

with bill_col3:
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)

st.markdown("---")

# ----------------- Prediction Logic -----------------
# Centered predict button
_, center_col, _ = st.columns([1, 1, 1])
with center_col:
    predict_button = st.button("Predict Churn", use_container_width=True)

if predict_button:
    # 1. Create a DataFrame for the input
    input_data = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet_service,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
    }])
    
    # 2. Encode to match training features
    input_encoded = pd.get_dummies(input_data)
    
    # 3. Align features with saved feature names
    input_aligned = input_encoded.reindex(columns=feature_names, fill_value=0)
    
    # 4. Predict
    if model_choice == "Logistic Regression":
        input_scaled = scaler.transform(input_aligned)
        prediction = lr_model.predict(input_scaled)[0]
        prob = lr_model.predict_proba(input_scaled)[0][1]
    else:
        prediction = rf_model.predict(input_aligned)[0]
        prob = rf_model.predict_proba(input_aligned)[0][1]
        
    st.markdown("---")
    if prediction == 1:
        st.error(f"Prediction: Customer is likely to CHURN.** (Probability: {prob:.2%})")
    else:
        st.success(f"Prediction: Customer is likely to REMAIN.** (Probability: {1-prob:.2%})")

st.markdown("---")
st.caption("Disclaimer: This application is for educational purposes and is not a guaranteed business decision or professional assessment.")