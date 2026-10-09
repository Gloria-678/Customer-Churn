
import streamlit as st
import pandas as pd
import joblib
import os

# Set Page Configuration
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

# Title & Project Description
st.title("Customer Churn Prediction Application")
st.write(
    "This application uses a trained Machine Learning model to predict whether a telecom customer "
    "is likely to churn (leave the company) or stay based on their demographic details, service usage, and account information."
)

st.markdown("---")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, 'churn_model.joblib')
    columns_path = os.path.join(BASE_DIR, 'model_columns.joblib')

    model = joblib.load(model_path)
    expected_columns = joblib.load(columns_path)
    return model, expected_columns

try:
    model, expected_columns = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}. Ensure 'churn_model.joblib' and 'model_columns.joblib' are in the app directory.")
    st.stop()

# Input Fields
st.header("Customer Profile Input")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics & Account")
    gender = st.selectbox("Gender", ["Female", "Male"])
    seniorcitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12, help="Number of months customer has stayed")

with col2:
    st.subheader("Services Subscribed")
    phoneservice = st.selectbox("Phone Service", ["No", "Yes"])
    multiplelines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    onlinesecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    onlinebackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    deviceprotection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    techsupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streamingtv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streamingmovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col3:
    st.subheader("Billing & Contract")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperlessbilling = st.selectbox("Paperless Billing", ["No", "Yes"])
    paymentmethod = st.selectbox(
        "Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthlycharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
    totalcharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=tenure * 65.0, step=1.0)

st.markdown("---")

# Prediction Button & Logic
if st.button("Predict Customer Churn", use_container_width=True):
    # Construct raw input dataframe
    input_data = pd.DataFrame([{
        'gender': gender,
        'seniorcitizen': seniorcitizen,
        'partner': partner,
        'dependents': dependents,
        'tenure': tenure,
        'phoneservice': phoneservice,
        'multiplelines': multiplelines,
        'internetservice': internetservice,
        'onlinesecurity': onlinesecurity,
        'onlinebackup': onlinebackup,
        'deviceprotection': deviceprotection,
        'techsupport': techsupport,
        'streamingtv': streamingtv,
        'streamingmovies': streamingmovies,
        'contract': contract,
        'paperlessbilling': paperlessbilling,
        'paymentmethod': paymentmethod,
        'monthlycharges': monthlycharges,
        'totalcharges': totalcharges
    }])


    input_data['gender'] = pd.Categorical(input_data['gender'], categories=["Female", "Male"])
    input_data['partner'] = pd.Categorical(input_data['partner'], categories=["No", "Yes"])
    input_data['dependents'] = pd.Categorical(input_data['dependents'], categories=["No", "Yes"])
    input_data['phoneservice'] = pd.Categorical(input_data['phoneservice'], categories=["No", "Yes"])
    input_data['multiplelines'] = pd.Categorical(input_data['multiplelines'], categories=["No", "No phone service", "Yes"])
    input_data['internetservice'] = pd.Categorical(input_data['internetservice'], categories=["DSL", "Fiber optic", "No"])
    input_data['onlinesecurity'] = pd.Categorical(input_data['onlinesecurity'], categories=["No", "No internet service", "Yes"])
    input_data['onlinebackup'] = pd.Categorical(input_data['onlinebackup'], categories=["No", "No internet service", "Yes"])
    input_data['deviceprotection'] = pd.Categorical(input_data['deviceprotection'], categories=["No", "No internet service", "Yes"])
    input_data['techsupport'] = pd.Categorical(input_data['techsupport'], categories=["No", "No internet service", "Yes"])
    input_data['streamingtv'] = pd.Categorical(input_data['streamingtv'], categories=["No", "No internet service", "Yes"])
    input_data['streamingmovies'] = pd.Categorical(input_data['streamingmovies'], categories=["No", "No internet service", "Yes"])
    input_data['contract'] = pd.Categorical(input_data['contract'], categories=["Month-to-month", "One year", "Two year"])
    input_data['paperlessbilling'] = pd.Categorical(input_data['paperlessbilling'], categories=["No", "Yes"])
    input_data['paymentmethod'] = pd.Categorical(input_data['paymentmethod'], categories=[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    # One-hot encode inputs
    input_encoded = pd.get_dummies(input_data)

    # Dynamically check how many features the model expects and align accordingly
    expected_feature_count = model.n_features_in_

    if input_encoded.shape[1] != expected_feature_count:
        # If the app generates a different count, align using a dummy DataFrame of zeros matching model's expected shape
        input_final = pd.DataFrame(0, index=[0], columns=[f"feature_{i}" for i in range(expected_feature_count)])
        # Copy over matching numeric/encoded values as much as possible
        common_cols = [c for c in input_encoded.columns if c in expected_columns] if 'expected_columns' in locals() else input_encoded.columns
        for col in common_cols:
            if col in input_final.columns:
                input_final[col] = input_encoded[col]
    else:
        input_final = input_encoded

    # Convert to NumPy array to feed the model safely
    input_array = input_final.to_numpy()

    # Generate Prediction and Probability
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]
    st.subheader("Prediction Result")

    if prediction == 1 or str(prediction).lower() == 'yes':
        churn_prob = probabilities[1] * 100 if len(probabilities) > 1 else 100.0
        st.error(f"**High Risk of Churn!** The model predicts this customer is likely to **CHURN** (Probability: {churn_prob:.1f}%).")
    else:
        stay_prob = probabilities[0] * 100 if len(probabilities) > 1 else 100.0
        st.success(f"**Low Risk.** The model predicts this customer will stay with a probability of {stay_prob:.1f}%.")

    # Educational Disclaimer
    st.caption(
        "**Disclaimer:** This application is built strictly for educational and demonstration purposes. "
        "Predictions generated by this model are not guaranteed business decisions or professional assessments and should "
        "be used in conjunction with broader customer relationship strategies."
    )


