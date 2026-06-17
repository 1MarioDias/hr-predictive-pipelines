# HR Predictive Pipelines — Group08

End-to-end machine learning project for an AI course. Two independent Scikit-learn pipelines trained on the same HR employee dataset: one predicting monthly salary and one predicting employee attrition.

---

## Evaluator Results

| Task | Metric | Score |
|---|---|---|
| Regression | R² | **0.9564** |
| Classification | Macro F1 | **0.8745** |
| Combined Score | (R² + Macro F1) / 2 | **0.9154** |

---

## Dataset

`employee_data/employee_data.csv` — 1249 rows × 35 columns, no missing values. The two targets are `MonthlyIncome` (continuous) and `Attrition` ("Yes"/"No" strings). Attrition is imbalanced: 84% "No" and 16% "Yes".

![Target distributions](assets/Distribui%C3%A7%C3%A3o%20de%20MonthlyIncome%20%28alvo%20%E2%80%94%20regress%C3%A3o%29%20e%20Distribui%C3%A7%C3%A3o%20de%20Attrition%20%28alvo%20%E2%80%94%20classifica%C3%A7%C3%A3o%29.png)

The correlation analysis showed `JobLevel` has ~0.95 correlation with `MonthlyIncome` — the strongest single predictor for regression. `TotalWorkingYears` and `Age` also contribute significantly.

![Numeric feature correlation heatmap](assets/Correla%C3%A7%C3%A3o%20entre%20vari%C3%A1veis%20num%C3%A9ricas.png)

Among categorical features, `OverTime` is the strongest attrition signal: employees with overtime have a 29% exit rate versus 11% for those without. `BusinessTravel` follows the same pattern.

![Attrition by OverTime and BusinessTravel](assets/Attrition%20por%20OverTime%20%28%25%29%20e%20Attrition%20por%20BusinessTravel%20%28%25%29.png)

---

## Pipeline Architecture

Both exported pipelines use only native scikit-learn components — no custom classes:

```
Pipeline
 ├── preprocessor  (ColumnTransformer)
 │    ├── num: SimpleImputer(median) → StandardScaler        [22 features]
 │    └── cat: SimpleImputer(most_frequent) → OneHotEncoder  [7 → 28 columns]
 ├── selector      (SelectKBest, k tuned)
 └── model         (estimator)
```

`handle_unknown="ignore"` on the encoder ensures unseen categories at inference time produce a zero vector instead of an error. The `SimpleImputer` handles missing values even though the training file has none, making the pipeline robust to the blind test.

---

## Regression — MonthlyIncome

Three models were compared. `LinearRegression` is the baseline; ensemble models capture non-linear interactions between features like `JobLevel`, `TotalWorkingYears`, and `Age`.

| Model | R² | RMSE | MAE |
|---|---|---|---|
| LinearRegression (baseline) | 0.9277 | 1134.48 | 882.22 |
| RandomForestRegressor | 0.9399 | 1034.38 | 769.58 |
| GradientBoostingRegressor | 0.9398 | 1034.88 | 770.19 |

![Residuals — LinearRegression baseline](assets/Res%C3%ADduos%20-%20LinearRegression%20baseline.png)

The baseline residual plot shows a fan pattern (heteroscedasticity) at higher salary values, which justifies using ensemble models that do not assume linearity.

![Residuals by model — Regression](assets/Res%C3%ADduos%20por%20modelo%20%E2%80%94%20Regress%C3%A3o.png)

`RandomForestRegressor` was tuned with `RandomizedSearchCV` (20 iterations, cv=5, `scoring='r2'`). Best parameters: `n_estimators=200`, `max_depth=5`, `min_samples_split=5`. Final R² on local test set: **0.9417**.

---

## Classification — Attrition

The evaluation metric is **Macro F1**: the arithmetic mean of the F1 score for each class ("Yes" and "No"). With only 16% "Yes", accuracy is misleading — a model that always predicts "No" gets 84% accuracy but a Macro F1 near 0.42. All model comparisons were made on Macro F1.

| Model | Macro F1 | Accuracy | ROC-AUC |
|---|---|---|---|
| LogisticRegression (baseline) | 0.6507 | 0.740 | 0.8018 |
| GradientBoostingClassifier | 0.7088 | 0.880 | 0.8119 |
| RandomForestClassifier | 0.6371 | 0.864 | 0.8009 |

![Confusion matrix — LogisticRegression baseline](assets/Matriz%20de%20Confus%C3%A3o%20%E2%80%94%20LogisticRegression%20baseline.png)

![Confusion matrices — all models](assets/Matrizes%20de%20Confus%C3%A3o%20%E2%80%94%20Classifica%C3%A7%C3%A3o.png)

`GradientBoostingClassifier` was tuned with `RandomizedSearchCV` (50 iterations, `StratifiedKFold(5)`, `scoring='f1_macro'`). `StratifiedKFold` preserves the 16% "Yes" proportion across folds, making cross-validation scores more stable. Best parameters: `n_estimators=300`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`. Final Macro F1 on local test set: **0.7106**.

---

## Deliverables

| File | Description |
|---|---|
| `G8_pipeline_regression.pkl` | Trained regression pipeline |
| `G8_pipeline_classification.pkl` | Trained classification pipeline |
| `Group08_notebook.ipynb` | Complete notebook with all 8 mandatory sections |
| `Group08_report.docx` | Written project report |

---

## Repository Structure

```
hr-predictive-pipelines/
├── G8_pipeline_regression.pkl       # DELIVERABLE
├── G8_pipeline_classification.pkl   # DELIVERABLE
├── Group08_notebook.ipynb           # DELIVERABLE
├── Group08_report.docx              # DELIVERABLE
├── README.txt                       # External libraries declaration
├── employee_data/
│   ├── employee_data.csv
│   └── employee_data.txt
├── assets/                          # Images exported from notebook
├── notebooks/                       # Working/EDA notebooks
├── reference/                       # Assignment PDF + course reference notebooks
├── docs/
│   ├── decisions.md                 # Decision log — source for the report
│   └── eda-techniques-guide.md
└── project ranking/                 # Teacher's evaluator
    ├── avaliador.py
    ├── dataset.csv
    └── modelos/                     # Pipeline copies for the evaluator
```

---

## Usage

```python
import pickle, pandas as pd

with open('G8_pipeline_regression.pkl', 'rb') as f:
    reg = pickle.load(f)

with open('G8_pipeline_classification.pkl', 'rb') as f:
    clf = pickle.load(f)

# Accepts raw DataFrame with the original employee_data.csv columns (minus the target)
reg_preds = reg.predict(X_raw)   # float — predicted MonthlyIncome
clf_preds = clf.predict(X_raw)   # strings 'Yes' or 'No'
```

---

## Deadline

Submission: **June 18, 2026 at 23:59** | Defence: **June 19, 2026 from 09:00**
