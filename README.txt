Group08 — HR Predictive Pipelines
==================================

Deliverables
------------
G8_pipeline_regression.pkl     — Trained regression pipeline (RandomForestRegressor)
G8_pipeline_classification.pkl — Trained classification pipeline (GradientBoostingClassifier)
Group08_notebook.ipynb          — Complete notebook with all 8 mandatory sections
Group08_report.docx             — Written project report

External Libraries (beyond Python stdlib)
------------------------------------------
scikit-learn==1.8.0
numpy==2.4.4
pandas==3.0.2
matplotlib==3.10.3
seaborn==0.13.2

All pipelines use only standard scikit-learn components.
No custom classes or external ML libraries (XGBoost, LightGBM, etc.) are used.

Usage
-----
import pickle, pandas as pd

with open('G8_pipeline_regression.pkl', 'rb') as f:
    reg = pickle.load(f)

with open('G8_pipeline_classification.pkl', 'rb') as f:
    clf = pickle.load(f)

# Accepts raw DataFrame rows (same columns as employee_data.csv, minus target)
reg_preds = reg.predict(X_raw)   # returns float MonthlyIncome values
clf_preds = clf.predict(X_raw)   # returns 'Yes'/'No' strings

Metrics on evaluator dataset
-----------------------------
Regression  R²           = 0.9564
Classification Macro F1  = 0.8745
Score Combinado          = 0.9154
