# Loan Default Prediction System

This project aims to predict loan defaults using machine learning techniques. The dataset contains various features such as income, loan amount, credit score, and demographic information. The project involves data preprocessing, exploratory data analysis (EDA), feature engineering, model building, and deployment using Streamlit.

## Table of Contents
- [Dataset Overview](#dataset-overview)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Results and Insights](#results-and-insights)
- [Deployment](#deployment)
- [Installation and Usage](#installation-and-usage)

## Dataset Overview
- **Number of Variables:** 18
- **Number of Observations (Rows):** 255,347
- **Missing Cells:** 0 (0.0%)
- **Duplicate Rows:** 0 (0.0%)
- **Total Dataset Size in Memory:** 52.6 MiB
- **Average Record Size in Memory:** 216.0 Bytes

## Exploratory Data Analysis (EDA)
1. **No Missing Values:** The dataset is complete with no missing values.
2. **No Duplicate Rows:** The dataset contains no duplicate records.
3. **Box Plot Analysis:**
   - Columns such as `Income` and `LoanAmount` have **no outliers**, as shown in their box plots.
   - These observations indicate a clean dataset ready for feature engineering and modeling.

## Feature Engineering
New features were created to enhance the predictive power of the dataset:
1. **Loan-to-Income Ratio:**
   ```python
   df['LoanToIncomeRatio'] = df['LoanAmount'] / df['Income']
   ```
   - Box plot analysis revealed **many outliers** in this feature.

2. **Credit Utilization Rate:**
   ```python
   df['CreditUtilizationRate'] = df['LoanAmount'] / df['CreditScore']
   ```
   - Box plot analysis showed **no outliers** in this feature.

### Label Encoding
- **Binary Columns:** Label encoding was applied to binary categorical features.
- **Categorical Columns:** One-hot encoding was used for multi-class categorical features.

### Feature Scaling
- **Numerical Features Scaling:** MinMaxScaler was applied to normalize numerical features to a common scale.

## Model Development
The following machine learning models were implemented and evaluated:
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting**
- **XGBoost**
- **AdaBoost**
- **Extra Trees**
- **CatBoost**

### Accuracy Comparison
| Model               | Accuracy   |
|---------------------|------------|
| Gradient Boosting   | **0.887781** |
| CatBoost            | 0.886881   |
| Random Forest       | 0.886626   |
| AdaBoost            | 0.886313   |
| XGBoost             | 0.886176   |
| Extra Trees         | 0.885667   |
| Decision Tree       | 0.804327   |

### Best Model
- **Gradient Boosting** was identified as the most accurate model with an accuracy of **88.78%**.

### Model Saving
- The Gradient Boosting model was saved as a `.pkl` file for deployment.

## Deployment
The model was deployed using a Streamlit application to provide an interactive interface for loan default prediction.
