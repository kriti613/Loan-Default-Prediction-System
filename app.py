import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model_path = 'gb_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# App title
st.title("Loan Default Prediction System")
st.write("Use this app to predict the likelihood of loan defaults based on user inputs.")

# Input fields for user data
st.sidebar.header("Enter Applicant Details")
def get_user_input():
    Age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    Income = st.sidebar.number_input("Income", min_value=1000, value=50000)
    LoanAmount = st.sidebar.number_input("Loan Amount", min_value=1000, value=20000)
    CreditScore = st.sidebar.slider("Credit Score", min_value=300, max_value=850, value=700)
    MonthsEmployed = st.sidebar.number_input("Months Employed", min_value=0, value=36)
    NumCreditLines = st.sidebar.number_input("Number of Credit Lines", min_value=0, value=5)
    InterestRate = st.sidebar.slider("Interest Rate (in %)", min_value=1.0, max_value=30.0, value=5.0)
    LoanTerm = st.sidebar.number_input("Loan Term (in months)", min_value=6, max_value=360, value=60)
    DTIRatio = st.sidebar.slider("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3)
    HasMortgage = st.sidebar.selectbox("Has Mortgage?", [0, 1])
    HasDependents = st.sidebar.selectbox("Has Dependents?", [0, 1])
    HasCoSigner = st.sidebar.selectbox("Has Co-Signer?", [0, 1])
    
    # One-hot encoded categorical inputs
    Education = st.sidebar.selectbox("Education", ["High School", "Master's", "PhD"])
    EmploymentType = st.sidebar.selectbox("Employment Type", ["Part-time", "Self-employed", "Unemployed"])
    MaritalStatus = st.sidebar.selectbox("Marital Status", ["Married", "Single"])
    LoanPurpose = st.sidebar.selectbox("Loan Purpose", ["Business", "Education", "Home", "Other"])

    # Convert to one-hot encoding for model compatibility
    Education_High_School = 1 if Education == "High School" else 0
    Education_Master = 1 if Education == "Master's" else 0
    Education_PhD = 1 if Education == "PhD" else 0

    EmploymentType_Part_time = 1 if EmploymentType == "Part-time" else 0
    EmploymentType_Self_employed = 1 if EmploymentType == "Self-employed" else 0
    EmploymentType_Unemployed = 1 if EmploymentType == "Unemployed" else 0

    MaritalStatus_Married = 1 if MaritalStatus == "Married" else 0
    MaritalStatus_Single = 1 if MaritalStatus == "Single" else 0

    LoanPurpose_Business = 1 if LoanPurpose == "Business" else 0
    LoanPurpose_Education = 1 if LoanPurpose == "Education" else 0
    LoanPurpose_Home = 1 if LoanPurpose == "Home" else 0
    LoanPurpose_Other = 1 if LoanPurpose == "Other" else 0

    # Calculate derived fields
    LoanToIncomeRatio = LoanAmount / Income
    CreditUtilizationRate = LoanAmount / (NumCreditLines * 10000)  # Assuming $10,000 credit limit per line

    data = {
        "Age": Age,
        "Income": Income,
        "LoanAmount": LoanAmount,
        "CreditScore": CreditScore,
        "MonthsEmployed": MonthsEmployed,
        "NumCreditLines": NumCreditLines,
        "InterestRate": InterestRate,
        "LoanTerm": LoanTerm,
        "DTIRatio": DTIRatio,
        "HasMortgage": HasMortgage,
        "HasDependents": HasDependents,
        "HasCoSigner": HasCoSigner,
        "LoanToIncomeRatio": LoanToIncomeRatio,
        "CreditUtilizationRate": CreditUtilizationRate,
        "Education_High School": Education_High_School,
        "Education_Master's": Education_Master,
        "Education_PhD": Education_PhD,
        "EmploymentType_Part-time": EmploymentType_Part_time,
        "EmploymentType_Self-employed": EmploymentType_Self_employed,
        "EmploymentType_Unemployed": EmploymentType_Unemployed,
        "MaritalStatus_Married": MaritalStatus_Married,
        "MaritalStatus_Single": MaritalStatus_Single,
        "LoanPurpose_Business": LoanPurpose_Business,
        "LoanPurpose_Education": LoanPurpose_Education,
        "LoanPurpose_Home": LoanPurpose_Home,
        "LoanPurpose_Other": LoanPurpose_Other
    }
    return pd.DataFrame(data, index=[0])

# Get user input
data = get_user_input()

# Display user input
st.subheader("Applicant Details")
st.write(data)

# Make predictions
if st.button("Predict Loan Default"):
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    
    if prediction == 1:
        st.error(f"High Risk: The applicant is likely to default with a probability of {probability:.2f}.")
    else:
        st.success(f"Low Risk: The applicant is unlikely to default with a probability of {1 - probability:.2f}.")
