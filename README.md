# ✈️ Flight Delay Prediction & Explainable AI Platform

A machine learning system that predicts airline departure delays before takeoff and explains prediction decisions using SHAP-based explainable AI.

---

## Overview

Flight delays create operational disruptions, passenger dissatisfaction, and financial losses.

This project develops an end-to-end machine learning workflow that:

- Processes airline operational data
- Predicts departure delays
- Evaluates model performance
- Explains predictions using SHAP
- Serves predictions through a Streamlit application

---

## Business Problem

Can we predict whether a flight will be delayed before departure using only information available before takeoff?

Target:

```text
IS_DELAYED

1 = Delay > 15 minutes
0 = On-Time
```

---

## Machine Learning Pipeline

```text
Raw Flight Data
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Target Creation
        │
        ▼
Model Training
        │
        ▼
Evaluation
        │
        ▼
SHAP Explainability
        │
        ▼
Streamlit Deployment
```

---

## Features

| Feature | Description |
|----------|------------|
| AIRLINE | Airline Code |
| ORIGIN_AIRPORT | Departure Airport |
| DESTINATION_AIRPORT | Arrival Airport |
| ROUTE | Flight Route |
| DAY_OF_WEEK | Day Indicator |
| DEP_HOUR | Scheduled Departure Hour |
| IS_PEAK_HOUR | Peak Traffic Flag |
| IS_WEEKEND | Weekend Indicator |
| DISTANCE | Flight Distance |

---

## Models Evaluated

### Logistic Regression

Baseline model.

### Random Forest

Final selected model.

Reasons:

- Handles nonlinear relationships
- Robust to feature interactions
- Interpretable with SHAP

---

## Explainable AI

The project uses SHAP to provide:

### Global Explanations

Which factors generally influence flight delays?

### Local Explanations

Why was a specific flight predicted as delayed?

---

## Tech Stack

| Category | Technology |
|-----------|-----------|
| Language | Python |
| Data Processing | Pandas |
| Machine Learning | Scikit-Learn |
| Explainability | SHAP |
| Visualization | Matplotlib |
| Deployment | Streamlit |

---

## Project Structure

```text
flight-delay-prediction/
│
├── data/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_explainability.ipynb
│
├── src/
│   ├── preprocess.py
│   └── label_data.py
│
├── app/
│   ├── app.py
│   └── shap_utils.py
│
└── README.md
```

---

## Future Improvements

- XGBoost Benchmarking
- MLflow Experiment Tracking
- FastAPI Prediction Service
- Docker Deployment
- Model Monitoring
- Real-Time Flight Data Integration

---

## Author

Ahamed Rilwan

GitHub: https://github.com/ahamril2265

LinkedIn: https://linkedin.com/in/ahamedrilwan
