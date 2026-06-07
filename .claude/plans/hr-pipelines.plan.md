# Plan: HR Predictive Pipelines — End-to-End Implementation

**Source**: `AI-Project-2526.pdf` + `CLAUDE.md`
**Complexity**: Medium
**Deadline**: June 18, 2026 at 23:59

## Summary

Build two independent Scikit-learn Pipelines — one predicting `MonthlyIncome` (regression, R²/RMSE) and one predicting `Attrition` (classification, F1-score) — over `employee_data/employee_data.csv`. Everything (imputation, encoding, scaling, model) must live inside the pipeline object so the exported `.pkl` works as a black box on raw data.

## Files to Create / Modify

| File | Action | Why |
|------|--------|-----|
| `Group08_notebook.ipynb` | CREATE | Main deliverable — full 8-section pipeline notebook |
| `Group08_pipeline_regression.pkl` | CREATE | Exported trained regression pipeline |
| `Group08_pipeline_classification.pkl` | CREATE | Exported trained classification pipeline |
| `notebooks/employee_eda.ipynb` | EXTEND | Add EDA sections 1.12–1.20 (nulls, corr heatmap, target plots) |

> Replace `Group08` with the actual group name throughout.

---

## Phase 1 — Complete EDA (`notebooks/employee_eda.ipynb`)

The notebook already has 13 cells covering structure inspection (Section 1.1–1.11). Add:

### Task 1.1 — Null analysis
- `df.isnull().sum().sort_values(ascending=False)` + comment
- Dataset is clean, but document it; add synthetic NaN injection to prove `SimpleImputer` works

### Task 1.2 — Correlation heatmap
```python
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=False, cmap='coolwarm')
```
- Comment: `JobLevel` and `MonthlyIncome` correlation is the strongest signal

### Task 1.3 — Target distributions
- Regression target: `sns.histplot(df['MonthlyIncome'], kde=True)` — comment skewness
- Classification target: `df['Attrition'].value_counts(normalize=True)` — document imbalance

### Task 1.4 — Categorical plots vs Attrition
```python
for col in ['OverTime', 'BusinessTravel', 'Department', 'MaritalStatus']:
    sns.countplot(data=df, x=col, hue='Attrition')
```

### Task 1.5 — GroupBy analysis
- `df.groupby('Department')['MonthlyIncome'].mean().sort_values()`
- `df.groupby('OverTime')['Attrition'].value_counts(normalize=True)`

---

## Phase 2 — Build Main Notebook (`Group08_notebook.ipynb`)

Create a single notebook that consolidates EDA + full pipeline training for both tasks. This is the submission artifact.

### Notebook header (Markdown cell 1)
```markdown
# HR Predictive Pipelines
**Dataset**: employee_data/employee_data.csv (1249 rows x 35 cols)
**Targets**: MonthlyIncome (regression) | Attrition (classification)
**Group**: Group08
```

### Global imports and constants (Code cell 2)
```python
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                              mean_squared_error, mean_absolute_error, r2_score, roc_auc_score)

_random_state = 24
DATA_PATH = Path("../employee_data/employee_data.csv")
df = pd.read_csv(DATA_PATH)
```

---

## Phase 3 — Section: Análises do Relatório de Dados

Copy and condense the best cells from `employee_eda.ipynb`. Must include:

- `head()` — confirm raw format
- `info()` — types + non-null counts
- `describe()` numerics + categoricals
- `isnull().sum()` — confirm/document nulls
- `corr()` + heatmap
- Target analysis: `MonthlyIncome` distribution, `Attrition` class balance
- Key categorical plots (`OverTime`, `Department` vs Attrition)
- Commentary Markdown after each group of commands

---

## Phase 4 — Section: Pré-processamento

Define feature lists and build the `ColumnTransformer`. No actual `.fit()` here — just design and document.

```python
COLS_TO_DROP = ['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18']

# Regression setup
X_reg = df.drop(columns=COLS_TO_DROP + ['MonthlyIncome', 'Attrition'])
y_reg = df['MonthlyIncome'].astype(float)

# Classification setup
y_clf = (df['Attrition'] == 'Yes').astype(int)
X_clf = df.drop(columns=COLS_TO_DROP + ['Attrition', 'MonthlyIncome'])

NUMERIC_FEATURES = X_reg.select_dtypes(include='number').columns.tolist()
CATEGORICAL_FEATURES = X_reg.select_dtypes(include='object').columns.tolist()
```

Build preprocessor (shared between both tasks):
```python
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])
```

Add Markdown: explain why each column was dropped, why `handle_unknown="ignore"` is mandatory for the blind test.

---

## Phase 5 — Section: Train/Test Split

```python
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=_random_state
)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=_random_state, stratify=y_clf
)
```

Add Markdown: explain `stratify=y_clf` for the imbalanced Attrition class (~16% Yes).

---

## Phase 6 — Section: Scaling

This section is minimal — scaling lives inside the pipeline. Document:
- Why `StandardScaler` is needed (different feature scales)
- Why it goes inside the pipeline (prevents data leakage from the test set)

---

## Phase 7 — Section: Treino

Define helper functions and train initial models:

```python
def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }

def classification_metrics(y_true, y_pred, y_prob=None):
    report = classification_report(y_true, y_pred, output_dict=True)
    result = {"F1": report['1']['f1-score'], "Accuracy": report['accuracy']}
    if y_prob is not None:
        result["ROC-AUC"] = roc_auc_score(y_true, y_prob)
    return result
```

Train baseline models:
- Regression: `LinearRegression` inside pipeline
- Classification: `LogisticRegression(class_weight='balanced', max_iter=1000)` inside pipeline

---

## Phase 8 — Section: Modelos

Train and compare at least 3 models per task. Build a results table.

### Regression models to compare
| Model | Role |
|-------|------|
| `LinearRegression` | Baseline |
| `RandomForestRegressor` | Main |
| `GradientBoostingRegressor` | Strong |

### Classification models to compare
| Model | Role |
|-------|------|
| `LogisticRegression(class_weight='balanced')` | Baseline |
| `RandomForestClassifier(class_weight='balanced')` | Main |
| `GradientBoostingClassifier` | Strong |

Add `ConfusionMatrixDisplay` for each classification model. Add residuals scatter plot for regression.

---

## Phase 9 — Section: Otimização

### Regression optimization
```python
param_grid_reg = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5],
}
search_reg = RandomizedSearchCV(
    Pipeline([("preprocessor", preprocessor),
              ("model", RandomForestRegressor(random_state=_random_state))]),
    param_distributions=param_grid_reg,
    n_iter=20, cv=5, scoring='r2',
    random_state=_random_state, n_jobs=-1,
)
search_reg.fit(X_reg_train, y_reg_train)
best_reg_pipeline = search_reg.best_estimator_
```

### Classification optimization
```python
param_grid_clf = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10],
    "model__class_weight": ['balanced'],
}
search_clf = RandomizedSearchCV(
    Pipeline([("preprocessor", preprocessor),
              ("model", RandomForestClassifier(random_state=_random_state))]),
    param_distributions=param_grid_clf,
    n_iter=20, cv=5, scoring='f1',
    random_state=_random_state, n_jobs=-1,
)
search_clf.fit(X_clf_train, y_clf_train)
best_clf_pipeline = search_clf.best_estimator_
```

Report: `search.best_params_`, `search.best_score_`, compare with baseline metrics.

---

## Phase 10 — Section: Deploy

### Serialize both pipelines
```python
GROUP = "Group08"

reg_path = Path(f"{GROUP}_pipeline_regression.pkl")
clf_path = Path(f"{GROUP}_pipeline_classification.pkl")

with open(reg_path, "wb") as f:
    pickle.dump(best_reg_pipeline, f)

with open(clf_path, "wb") as f:
    pickle.dump(best_clf_pipeline, f)
```

### Validate — load and predict on raw data
```python
with open(reg_path, "rb") as f:
    loaded_reg = pickle.load(f)

with open(clf_path, "rb") as f:
    loaded_clf = pickle.load(f)

# No preprocessing — pass raw X directly
reg_preds = loaded_reg.predict(X_reg_test)
clf_preds = loaded_clf.predict(X_clf_test)

print("Regression R2:", r2_score(y_reg_test, reg_preds))
print("Classification F1:", classification_report(y_clf_test, clf_preds, output_dict=True)['1']['f1-score'])
print("Pipeline files are valid — ready for blind test.")
```

---

## Phase 11 — Final Validation Before Submission

```python
import os
print(os.path.getsize(f"{GROUP}_pipeline_regression.pkl"), "bytes")
print(os.path.getsize(f"{GROUP}_pipeline_classification.pkl"), "bytes")
# Expected: several MB each, not 0 bytes
```

Also test with synthetic NaN data to confirm `SimpleImputer` handles missing values:
```python
X_test_with_nans = X_reg_test.copy()
X_test_with_nans.iloc[0, 0] = np.nan
_ = loaded_reg.predict(X_test_with_nans)  # must not raise
print("NaN handling: OK")
```

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `OneHotEncoder` fails on unseen categories in blind test | HIGH | `handle_unknown="ignore"` in pipeline |
| Attrition imbalance tanks F1 | MEDIUM | `class_weight='balanced'` + stratified split |
| Pipeline leaks test data through scaler | MEDIUM | Scaler is inside pipeline, fit only on train |
| `sparse_output` deprecation warning (sklearn >= 1.2) | LOW | Use `sparse_output=False` explicitly |
| pkl incompatible across sklearn versions | LOW | Note sklearn version in report |

---

## Acceptance

- [ ] All 8 notebook sections present with exact Markdown headers
- [ ] Both pipelines trained and predict on `X_test` without manual preprocessing
- [ ] Both `.pkl` files load from disk and predict on raw DataFrame
- [ ] RandomizedSearchCV completed and best params documented
- [ ] Results table comparing at least 2 models per task
- [ ] Confusion matrix shown for classification
- [ ] Residuals plot shown for regression
- [ ] `_random_state = 24` used everywhere
- [ ] Group name `Group08` replaced with actual group name in file names
- [ ] NaN injection test passes

---

## Implementation Order

```
Phase 2 (header + imports)
  -> Phase 3 (EDA section)
    -> Phase 4 (Preprocessing design)
      -> Phase 5 (Train/Test Split)
        -> Phase 6 (Scaling docs)
          -> Phase 7 (Treino)
            -> Phase 8 (Modelos)
              -> Phase 9 (Otimização)
                -> Phase 10 (Deploy + validation)
                  -> Phase 11 (Final validation)
                    -> Phase 1 (Extend employee_eda.ipynb if needed)
```
