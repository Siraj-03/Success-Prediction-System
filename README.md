<div align="center">

# Startup Success Prediction

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square"/>
</p>

<p>An end-to-end machine learning system that predicts whether a startup will be <strong>acquired</strong> or <strong>shut down</strong> — built across 7 modular pipelines covering classical ML, deep learning, and explainable AI, with a live interactive frontend.</p>

**Mohammed Hasnaine** &nbsp;·&nbsp; B.Tech CSE, 4th Year &nbsp;·&nbsp; Major Internship Project

</div>

---

## Overview

This project tackles binary classification on real-world startup data — predicting success (acquired) or failure (closed) using a full machine learning pipeline. It goes beyond model training to include rigorous preprocessing, domain-specific feature engineering, a deep learning suite, and production-grade model explainability through SHAP and LIME.

**Dataset:** 923 startups · 49 raw features · cleaned to 778 rows × 42 features · sourced from Kaggle

**Target:** `acquired` → 1 (Success) &nbsp;|&nbsp; `closed` → 0 (Failure)

---

## Project Modules

| # | Module | Description |
|---|---|---|
| 1 | **Preprocessing** | Missing value imputation, feature engineering, SMOTE, StandardScaler, train/test split |
| 2 | **EDA** | 11 visualizations covering distributions, correlations, geographic and sector analysis |
| 3 | **Classical ML** | Logistic Regression, SVM, Random Forest, Gradient Boosting, XGBoost with GridSearchCV |
| 4 | **ANN** | Deep feedforward network with BatchNorm, Dropout, and early stopping (47 epochs) |
| 5 | **Deep Learning** | 1D-CNN, LSTM, and hybrid CNN-LSTM architectures on tabular data |
| 6 | **Explainability** | SHAP (global + per-prediction) and LIME explanations on the best model |
| 7 | **Frontend** | Interactive web app — enter startup details, get a live prediction with explanation |

---

## Feature Engineering

Six domain-specific features were derived from raw funding data to improve predictive signal:

| Feature | Definition |
|---|---|
| `funding_efficiency` | Total funding ÷ (funding rounds + 1) |
| `investor_strength` | Weighted score: VC×3 + Angel×2 + Round A/B/C/D |
| `milestone_speed` | Milestones ÷ (age at first milestone + 1) |
| `network_funding_ratio` | Relationships ÷ log(funding + 1) |
| `is_top_state` | Binary flag — California, New York, or Massachusetts |
| `log_funding` | log1p transform of total funding to reduce skew |

All six features appear in the top-15 SHAP global importance rankings, validating their predictive value.

---

## Results

### Model Leaderboard

| Model | Type | Accuracy | ROC-AUC |
|---|---|---|---|
| 🏆 **Random Forest** | Classical ML | **81.5%** | **0.909** |
| Gradient Boosting | Classical ML | 81.4% | 0.901 |
| XGBoost | Classical ML | 77.6% | 0.882 |
| **ANN** | Neural Network | **78.5%** | **0.880** |
| SVM | Classical ML | 74.8% | 0.844 |
| **CNN** | Deep Learning | **74.3%** | **0.840** |
| Logistic Regression | Classical ML | 71.9% | 0.812 |
| LSTM | Deep Learning | 71.7% | 0.778 |
| CNN-LSTM | Deep Learning | 52.4% | 0.727 |

**Best model: Random Forest** — ROC-AUC 0.909, CV accuracy 79.1%, saved as `best_classical_model.pkl`

**Best deep model: CNN** — ROC-AUC 0.840, saved as `CNN_best.keras`

### Top Predictive Features (SHAP)

| Rank | Feature | Mean SHAP |
|---|---|---|
| 1 | `relationships` | 0.0616 |
| 2 | `network_funding_ratio` | 0.0554 |
| 3 | `milestones` | 0.0413 |
| 4 | `is_top500` | 0.0381 |
| 5 | `log_funding` | 0.0248 |

### Key Findings

- **Massachusetts** leads in startup success rate at **74.3%**; California dominates in volume
- **Enterprise** software has the highest sector success rate(~75%); 
-   E-Commerce the  lowest        (~40%)
- Reaching **Series B** funding adds a **+22.3%** lift in success probability
- Having VC funding alone (without subsequent rounds) shows a **−7.5%** lift — a counterintuitive risk signal
- Startups with **3+ milestones** survive at dramatically higher rates than those with fewer

---

## Tech Stack

| Category | Tools |
|---|---|
| Data & ML | Pandas, NumPy, Scikit-learn, XGBoost |
| Deep Learning | TensorFlow 2.x, Keras |
| Explainability | SHAP, LIME |
| Visualization | Matplotlib, Seaborn, Plotly |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Frontend | Streamlit |
| Persistence | Joblib, Keras `.keras` format |

---

## Project Structure

```
StartupSuccessPrediction/
├── dataset/
│   └── startup_data.csv
├── notebooks/
│   ├── Module1_Preprocessing.ipynb
│   ├── Module2_EDA.ipynb
│   ├── Module3_Classical_ML.ipynb
│   ├── Module4_ANN.ipynb
│   ├── Module5_DeepLearning.ipynb
│   └── Module6_Explainability.ipynb
├── models/
│   ├── best_classical_model.pkl      ← Random Forest (AUC 0.909)
│   ├── ann_best_model.keras          ← ANN (AUC 0.880)
│   ├── CNN_best.keras                ← CNN (AUC 0.840)
│   ├── LSTM_best.keras
│   ├── CNN-LSTM_best.keras
│   ├── scaler.pkl
│   └── feature_names.pkl
├── outputs/
│   ├── clean_dataset.csv
│   ├── plots/                        ← 27 saved visualizations
│   └── shap_feature_summary.csv
└── app.py                            ← Frontend web application
```

---

## Setup & Usage

```bash
# Clone the repository
git clone https://github.com/MohammedHasnaine/startup-success-prediction.git
cd startup-success-prediction

# Install dependencies
pip install pandas numpy scikit-learn xgboost tensorflow keras shap lime \
            imbalanced-learn matplotlib seaborn plotly streamlit joblib

# Run notebooks in order: Module1 → Module2 → ... → Module6
jupyter notebook

# Launch the frontend
streamlit run app.py
```

> All modules are independent after Module 1 completes. Each notebook loads its inputs from `outputs/` and saves its artifacts back — no re-running required.

---

## Acknowledgements

Dataset by [Manish KC on Kaggle](https://www.kaggle.com/datasets/manishkc06/startup-success-prediction) &nbsp;·&nbsp;
SHAP by Lundberg & Lee (NeurIPS 2017) &nbsp;·&nbsp;
LIME by Ribeiro et al. (KDD 2016)

---

<div align="center">
<sub>Mohammed Hasnaine &nbsp;·&nbsp; B.Tech Computer Science & Engineering, 4th Year &nbsp;·&nbsp; Major Internship Project</sub>
</div>
