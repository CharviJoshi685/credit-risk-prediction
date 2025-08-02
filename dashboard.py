import streamlit as st
import pandas as pd
import joblib
import numpy as np

# === Load Models & Scaler ===
log_model = joblib.load("logistic_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Financial Risk Prediction", page_icon="💳", layout="centered")

st.title("💳 Financial Risk Prediction & Credit Scoring")
st.write("Enter applicant details below to predict creditworthiness:")

# === User Input Form ===
with st.form("credit_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    job = st.selectbox("Job", [0, 1, 2, 3])
    housing = st.selectbox("Housing", ["own", "free", "rent"])
    saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich", "NA"])
    checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "NA"])
    credit_amount = st.number_input("Credit Amount (€)", min_value=100, value=5000)
    duration = st.number_input("Loan Duration (months)", min_value=1, value=12)
    purpose = st.selectbox("Purpose", ["radio/TV", "education", "furniture/equipment", "car", "business", "vacation/others", "repairs", "domestic appliances"])

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    # Prepare input for model
    input_dict = {
        "Age": age,
        "Sex_female": 1 if sex == "female" else 0,
        "Sex_male": 1 if sex == "male" else 0,
        "Job": job,
        "Housing_free": 1 if housing == "free" else 0,
        "Housing_own": 1 if housing == "own" else 0,
        "Housing_rent": 1 if housing == "rent" else 0,
        "Saving accounts_little": 1 if saving_accounts == "little" else 0,
        "Saving accounts_moderate": 1 if saving_accounts == "moderate" else 0,
        "Saving accounts_quite rich": 1 if saving_accounts == "quite rich" else 0,
        "Saving accounts_rich": 1 if saving_accounts == "rich" else 0,
        "Saving accounts_nan": 1 if saving_accounts == "NA" else 0,
        "Checking account_little": 1 if checking_account == "little" else 0,
        "Checking account_moderate": 1 if checking_account == "moderate" else 0,
        "Checking account_rich": 1 if checking_account == "rich" else 0,
        "Checking account_nan": 1 if checking_account == "NA" else 0,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose_business": 1 if purpose == "business" else 0,
        "Purpose_car": 1 if purpose == "car" else 0,
        "Purpose_domestic appliances": 1 if purpose == "domestic appliances" else 0,
        "Purpose_education": 1 if purpose == "education" else 0,
        "Purpose_furniture/equipment": 1 if purpose == "furniture/equipment" else 0,
        "Purpose_radio/TV": 1 if purpose == "radio/TV" else 0,
        "Purpose_repairs": 1 if purpose == "repairs" else 0,
        "Purpose_vacation/others": 1 if purpose == "vacation/others" else 0,
        "Purpose_nan": 0  # Assuming no NA for purpose
    }
    # Convert to DataFrame with all expected columns
    expected_columns = [
        "Age", "Sex_female", "Sex_male", "Job", "Housing_free", "Housing_own", "Housing_rent",
        "Saving accounts_little", "Saving accounts_moderate", "Saving accounts_quite rich", "Saving accounts_rich", "Saving accounts_nan",
        "Checking account_little", "Checking account_moderate", "Checking account_rich", "Checking account_nan",
        "Credit amount", "Duration",
        "Purpose_business", "Purpose_car", "Purpose_domestic appliances", "Purpose_education",
        "Purpose_furniture/equipment", "Purpose_radio/TV", "Purpose_repairs", "Purpose_vacation/others", "Purpose_nan"
    ]
    input_data = pd.DataFrame([{col: input_dict.get(col, 0) for col in expected_columns}])

    # Scale for Logistic Regression
    input_scaled = scaler.transform(input_data)

    # Predictions
    log_pred = log_model.predict(input_scaled)[0]
    rf_pred = rf_model.predict(input_data)[0]

    st.subheader("📊 Predictions")
    st.write("Logistic Regression Model:", "✅ Good" if log_pred == 1 else "❌ Bad")
    st.write("Random Forest Model:", "✅ Good" if rf_pred == 1 else "❌ Bad")

    # Probability Scores
    st.subheader("📈 Probability Scores")
    log_proba = log_model.predict_proba(input_scaled)[0][1] * 100
    rf_proba = rf_model.predict_proba(input_data)[0][1] * 100
    st.write(f"Logistic Regression: {log_proba:.2f}% chance of Good Credit")
    st.write(f"Random Forest: {rf_proba:.2f}% chance of Good Credit")
