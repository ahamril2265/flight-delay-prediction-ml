# ✈️ Flight Delay Prediction System

Predict whether a flight is likely to be delayed **before departure** using a machine-learning model trained on real US airline data.  
The system provides **both predictions and explanations** using SHAP, making the model transparent and interpretable.

---

## 🚀 Project Overview

Airline delays cause operational inefficiencies and poor passenger dissatisfaction.  
This project builds an **end-to-end ML system** that:

- Predicts flight delays using historical airline data
- Handles real-world data issues (class imbalance, leakage, high cardinality)
- Explains predictions using **SHAP (Explainable AI)**
- Deploys predictions via an interactive **Streamlit web app**

---

## 🧠 Machine Learning Approach

### Problem Type
- **Binary Classification**
- Target: `IS_DELAYED`
  - `1` → Departure delay > 15 minutes  
  - `0` → On-time or early departure  

### Model Pipeline
- **Preprocessing**
  - One-Hot Encoding for categorical features
  - Feature scaling for numeric features
  - High-cardinality handling (top airlines & routes)
- **Models**
  - Logistic Regression (baseline)
  - Random Forest (final tuned model)
- **Evaluation**
  - ROC-AUC (primary metric)
  - Precision, Recall, F1-Score
- **Explainability**
  - SHAP global & local explanations

---

## 🗂️ Project Structure

flight-delay-ml/
│
├── data/
│ └── processed/
│ └── flights_ml.csv
│
├── models/
│ └── best_model.joblib
│
├── notebooks/
---

## 📊 Features Used

| Feature | Description |
|------|------------|
| AIRLINE | Operating airline |
| ORIGIN_AIRPORT | Departure airport |
| DESTINATION_AIRPORT | Arrival airport |
| ROUTE | Origin → Destination |
| DAY_OF_WEEK | Day of week |
| DEP_HOUR | Scheduled departure hour |
| IS_PEAK_HOUR | Peak traffic indicator |
| IS_WEEKEND | Weekend flag |
| DISTANCE | Flight distance (miles) |

---

## 🧪 Dataset

- **Source:** US Airline On-Time Performance Data (BTS / Kaggle)
- **Year Used:** 2018
- **Processing Highlights:**
  - Cancelled flights removed
  - Target leakage prevented
  - Date-derived features engineered
  - Sampled for efficient training

---

## 🔍 Explainable AI (SHAP)

The system provides:
- **Global explanations** → What generally causes delays
- **Local explanations** → Why a specific flight was predicted as delayed

This makes the model suitable for **decision support**, not just prediction.

---

## 🖥️ Running the App Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/flight-delay-ml.git
cd flight-delay-ml

│ ├── 01_eda.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_model_training.ipynb
│ └── 04_explainability.ipynb
│
├── app/
│ ├── app.py
│ └── shap_utils.py
│
├── requirements.txt
└── README.md

