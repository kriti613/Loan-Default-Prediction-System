# Loan Default Prediction System

## Overview
The **Loan Default Prediction System** is a machine learning project that aims to predict the likelihood of loan default using various borrower and loan-related attributes. The project involves data preprocessing, exploratory data analysis, feature engineering, model building, and deployment via a Streamlit application.

## Dataset Details
After cleaning and preprocessing, the dataset includes:
- **Entries:** 255,347
- **Columns:** 27
- **No missing values or duplicates**

### Data Columns:
1. **Age** (numerical)
2. **Income** (numerical)
3. **LoanAmount** (numerical)
4. **CreditScore** (numerical)
5. **MonthsEmployed** (numerical)
6. **NumCreditLines** (numerical)
7. **InterestRate** (numerical)
8. **LoanTerm** (numerical)
9. **DTIRatio** (numerical)
10. **HasMortgage** (binary)
11. **HasDependents** (binary)
12. **HasCoSigner** (binary)
13. **Default** (target variable)
14. **LoanToIncomeRatio** (engineered feature)
15. **CreditUtilizationRate** (engineered feature)
16. One-hot encoded categorical columns:
    - Education: `Education_High School`, `Education_Master's`, `Education_PhD`
    - EmploymentType: `EmploymentType_Part-time`, `EmploymentType_Self-employed`, `EmploymentType_Unemployed`
    - MaritalStatus: `MaritalStatus_Married`, `MaritalStatus_Single`
    - LoanPurpose: `LoanPurpose_Business`, `LoanPurpose_Education`, `LoanPurpose_Home`, `LoanPurpose_Other`

### Memory Usage:
52.6 MB

## Exploratory Data Analysis (EDA)
1. **Alerts:**
   - `CreditUtilizationRate` and `LoanAmount` are highly correlated.
   - `MaritalStatus_Married` and `MaritalStatus_Single` are highly correlated.
   <img src="https://github.com/kriti613/Loan-Default-Prediction-System/blob/main/correlation_HeatMap.png" style="height: 80%; width: 80%">

   - `LoanToIncomeRatio` is highly skewed (γ1 = 228.9997178).

   - `Age` has 4,884 (1.9%) zeros.
   
3. **Box Plots:**
   - No outliers in `Income` and `LoanAmount`.
   <img src="https://github.com/kriti613/Loan-Default-Prediction-System/blob/main/incomeloanamountBoxPlot.png" style="height: 80%; width: 80%">
   
   - `LoanToIncomeRatio` has many outliers.
   - `CreditUtilizationRate` has no outliers.
    <img src="https://github.com/kriti613/Loan-Default-Prediction-System/blob/main/income_LoanAmount_boxplot.png" style="height: 80%; width: 80%">

## Feature Engineering
1. **Binary Columns:**
   - `HasMortgage`, `HasDependents`, `HasCoSigner`

2. **One-hot Encoding:**
   - Categorical Columns: `Education`, `EmploymentType`, `MaritalStatus`, `LoanPurpose`

3. **Engineered Features:**
   ```python
   df['LoanToIncomeRatio'] = df['LoanAmount'] / df['Income']
   df['CreditUtilizationRate'] = df['LoanAmount'] / df['CreditScore']
   ```

4. **Numerical Features Scaling:**
   - Columns scaled using `MinMaxScaler`:
     - `Age`, `Income`, `LoanAmount`, `CreditScore`, `MonthsEmployed`, `NumCreditLines`, `InterestRate`, `LoanTerm`, `DTIRatio`, `LoanToIncomeRatio`, `CreditUtilizationRate`

## Models and Accuracy
Several models were evaluated and compared based on their accuracy:
| Model              | Accuracy  |
|--------------------|-----------|
| Gradient Boosting  | 0.887781  |
| CatBoost           | 0.886881  |
| Random Forest      | 0.886626  |
| AdaBoost           | 0.886313  |
| XGBoost            | 0.886176  |
| Extra Trees        | 0.885667  |
| Decision Tree      | 0.804327  |

**Conclusion:** Gradient Boosting is the most accurate model for this dataset.

## Deployment
The trained Gradient Boosting model was saved as a `.pkl` file and deployed using a Streamlit application for user-friendly predictions.

<img src="https://github.com/kriti613/Loan-Default-Prediction-System/blob/main/LoanDefaultPredictionApp.png" style="height: 80%; width: 80%">
