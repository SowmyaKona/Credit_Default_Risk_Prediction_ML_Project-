# 💳 Credit Risk Default Prediction

> An end-to-end machine learning project that predicts whether a credit card customer will default next month — trained on 30,000 real customers and deployed as an interactive Streamlit web application.

---

## Problem Statement

Credit card defaults pose a significant financial risk to lending institutions. Early identification of high-risk customers allows banks to take preventive action before a default occurs.

This project builds a **supervised binary classification model** trained on 30,000 real credit card customers to predict the probability of default in the following month. The model leverages six months of repayment behaviour, billing statements, payment amounts, and customer demographics — combining raw features with engineered signals — to produce an interpretable risk score.

---

## 🚀 Live Demo

👉 **[Open the Streamlit App] (https://vejktlcgvheavagxr9yjiq.streamlit.app/)**

Enter a customer's 6-month credit history and get an instant default probability prediction.

| Low Risk Customer | High Risk Customer |
|---|---|
| ✅ Not Likely to Default | ⚠️ Likely to Default |
| Probability < 50% | Probability ≥ 50% |

---

## Project Structure

```
credit-risk-default-prediction/
│
├── Credit_Default_risk_Prediction-ML.ipynb   # Full notebook — EDA, training, evaluation
├── app.py                                     # Streamlit web application
├── model.pkl                                  # Trained XGBoost pipeline (pickle)
├── feature_order.pkl                          # Saved feature column order
├── default of credit card clients.xls        # Raw dataset (UCI)
└── requirements.txt                           # Python dependencies
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | UCI Machine Learning Repository |
| **Customers** | 30,000 credit card holders (Taiwan) |
| **Time Period** | 6 months of repayment history |
| **Target** | `default payment next month` — 1 = Default, 0 = No Default |
| **Class Split** | 78% No Default · 22% Default (imbalanced) |

### Feature Groups

| Group | Columns | Description |
|---|---|---|
| Demographics | `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Customer profile |
| Repayment Status | `PAY_0` → `PAY_6` | Monthly payment delay (-2 to 9) |
| Bill Amounts | `BILL_AMT1` → `BILL_AMT6` | Monthly statement amounts |
| Payment Amounts | `PAY_AMT1` → `PAY_AMT6` | Actual payments made |

**PAY scale:** `-2` = No consumption · `-1` = Paid in full · `0` = Minimum paid · `1–8` = Months of delay

---

## 🔧 Project Pipeline

```
Raw Data → Data Cleaning → EDA → Feature Engineering
       → Train-Test Split → Outlier Handling → Preprocessing Pipeline
       → Model Training (7 Models + GridSearchCV) → Evaluation → Deployment
```

### Steps at a Glance

1. **Data Cleaning** — Fixed undocumented categories in EDUCATION and MARRIAGE, dropped ID column, removed duplicates
2. **EDA** — 8 visualizations revealing PAY_0 as the strongest predictor and the 78/22 class imbalance
3. **Feature Engineering** — Created 9 new features from the raw data
4. **Outlier Handling** — IQR clipping on training data only (no data leakage)
5. **Preprocessing Pipeline** — StandardScaler for numeric, OneHotEncoder for categorical, using ColumnTransformer
6. **Model Training** — 7 models trained with GridSearchCV + StratifiedKFold (5 folds), scored on ROC-AUC
7. **Class Imbalance** — SMOTE inside CV folds for most models, `scale_pos_weight` for XGBoost
8. **Evaluation** — Confusion matrix, ROC curve, Precision-Recall curve, threshold tuning
9. **Feature Importance** — XGBoost importance plot to interpret model decisions

---

##  Feature Engineering

| Feature | Formula | What It Captures |
|---|---|---|
| `total_bill` | Sum of BILL_AMT1–6 | Total 6-month debt load |
| `avg_bill` | Mean of BILL_AMT1–6 | Average monthly bill |
| `bill_trend` | BILL_AMT1 − BILL_AMT6 | Is debt growing? (positive = growing) |
| `util_rate` | BILL_AMT1 / (LIMIT_BAL + 1) | Credit utilization ratio |
| `total_pay` | Sum of PAY_AMT1–6 | Total payments made |
| `avg_pay` | Mean of PAY_AMT1–6 | Average monthly payment |
| `pay_ratio` | PAY_AMT1 / (BILL_AMT1 + 1) | How much of last bill was paid |
| `avg_pay_delay` | Mean of PAY_0, PAY_2–6 | Consistent delay behaviour |
| `max_pay_delay` | Max of PAY_0, PAY_2–6 | Worst-case delay in 6 months |

---

##  Model Training

### Models Compared

| Model | Class Imbalance Handling |
|---|---|
| Logistic Regression | SMOTE inside CV + `class_weight='balanced'` |
| KNN | Scaled data |
| Naive Bayes | SMOTE inside CV |
| SVC | `class_weight='balanced'` |
| Decision Tree | SMOTE inside CV + `class_weight='balanced'` |
| Random Forest | SMOTE inside CV + `class_weight='balanced'` |
| **XGBoost** ✅ | `scale_pos_weight = 3.5` |

- **Cross-validation:** StratifiedKFold (5 folds) — preserves 78/22 ratio in every fold
- **Scoring metric:** ROC-AUC — robust to class imbalance, unlike raw accuracy
- **Hyperparameter tuning:** GridSearchCV across all models

---

## 📈 Results

### Model Comparison (Test Set)

| Model | ROC-AUC | F1 (Default) | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| **XGBoost** ✅ | **0.7800** | 0.5276 | 0.4443 | **0.6493** | 0.7427 |
| Random Forest | 0.7709 | **0.5305** | 0.5055 | 0.5581 | 0.7814 |
| Gradient Boosting | 0.7700 | 0.5142 | **0.5645** | 0.4721 | **0.8026** |
| Logistic Regression | 0.7547 | 0.5174 | 0.4447 | 0.6184 | 0.7447 |
| Decision Tree | 0.7511 | 0.5072 | 0.4741 | 0.5452 | 0.7656 |

### Why XGBoost?

> In credit risk, **Recall is the priority**. Missing a defaulter (False Negative) costs the bank real money. XGBoost achieves the **highest Recall (0.65)** — it correctly identifies 65 out of every 100 actual defaulters — while also leading in ROC-AUC (0.78).

### Top Feature Importances (XGBoost)

```
PAY_0          ████████████████████  Most important
PAY_2          ████████████
avg_pay_delay  ██████████
max_pay_delay  █████████
LIMIT_BAL      ████████
pay_ratio      ███████
PAY_3          ██████
util_rate      █████
AGE            ██
SEX            █
EDUCATION      █                     Least important
```

---

##  Streamlit App — How It Works

The app takes **full 6-month customer history** as input across 4 sections:

1. **Customer Profile** — Credit limit, gender, education, marital status, age
2. **Repayment Status** — PAY status for each of the last 6 months
3. **Bill Amounts** — Monthly bill statements for the last 6 months
4. **Payment Amounts** — Actual payments made for the last 6 months

On clicking **Predict Default Risk**, the app:
- Builds a feature row from all inputs
- Computes all 9 engineered features using the same formulas as the notebook
- Reorders columns using `feature_order.pkl` to match training exactly
- Passes through the loaded XGBoost pipeline (`model.pkl`)
- Outputs binary prediction: **Not Likely to Default** or **Likely to Default**

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/credit-risk-default-prediction.git
cd credit-risk-default-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

> **Note:** `model.pkl` and `feature_order.pkl` must be in the same directory as `app.py`. To retrain the model, run all cells in `Credit_Default_risk_Prediction-ML.ipynb`.

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
streamlit
matplotlib
seaborn
openpyxl
xlrd
pickle5
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🔍 Key Decisions & Reasoning

| Challenge | Decision | Why |
|---|---|---|
| Class imbalance (78/22) | SMOTE inside CV + `scale_pos_weight` | Prevents synthetic samples leaking into validation |
| Skewed features | No log transform applied | Mild skew (0.9–1.2), tree models don't require normality |
| Multicollinearity in BILL_AMTs | Kept all columns | XGBoost handles multicollinearity via feature selection at each split |
| Scoring metric | ROC-AUC instead of Accuracy | Accuracy misleading on imbalanced data |
| Threshold | 0.50 (binary dataset) | Model trained on 0/1 binary target — 50% is the natural boundary |
| Outlier handling | IQR clipping on train only | Prevents data leakage to test set |

---

## 🚧 Limitations & Future Work

- App uses full 6-month input — a simplified version with fewer fields could improve usability
- SHAP values could be added for per-customer prediction explanation
- A cost-sensitive threshold (weighting False Negatives 5–10× higher than False Positives) would better reflect real banking risk
- Could be deployed as a REST API using FastAPI for integration with banking systems
- Model performance could improve with more recent data or additional features like income

---

**Tech Stack:** Python · Pandas · NumPy · Scikit-learn · XGBoost · Imbalanced-learn · Streamlit · Matplotlib · Seaborn

---

*If you found this project helpful, please ⭐ the repository!*
