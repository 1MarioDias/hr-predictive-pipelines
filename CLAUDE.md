# CLAUDE.md — HR Predictive Pipelines

This file guides Claude Code when working in `hr-predictive-pipelines/`.

## Project at a Glance

End-to-end ML project for a university AI course. Two independent Scikit-learn Pipelines over the same HR dataset:

| Task | Target | Metric | Output file |
|------|--------|--------|-------------|
| Regression | `MonthlyIncome` (continuous) | R² + RMSE | `Group08_pipeline_regression.pkl` |
| Classification | `Attrition` (Yes/No) | **F1-score** | `Group08_pipeline_classification.pkl` |

**Deadline**: June 18, 2026 at 23:59. Defense: June 19, 2026 from 09:00.

The evaluation is a **blind test**: each `.pkl` must accept raw DataFrame rows (same columns as `employee_data.csv`, minus the target) and return predictions without any manual preprocessing.

## Repository Layout

```
hr-predictive-pipelines/
├── Group08_notebook.ipynb              # DELIVERABLE — main notebook (all 8 sections)
├── Group08_pipeline_regression.pkl     # DELIVERABLE — trained regression pipeline
├── Group08_pipeline_classification.pkl # DELIVERABLE — trained classification pipeline
├── Group08_report.docx                 # DELIVERABLE — written manually by the group
├── employee_data/
│   ├── employee_data.csv               # 1249 rows × 35 columns — main dataset
│   └── employee_data.txt               # Ordinal label legend (Education, JobSatisfaction, etc.)
├── notebooks/
│   └── employee_eda.ipynb              # Working EDA scratch notebook (not a deliverable)
├── reference/
│   ├── AI-Project-2526.pdf             # Assignment specification
│   └── class_notebooks/                # Course reference notebooks (read-only)
│       ├── 260306 - DT iris.ipynb
│       ├── 260313 - PandasPrimer IRIS.ipynb
│       ├── 260327 - RF titanic_3_short version.ipynb
│       ├── 260410_XGBoost_TelecomChurn.ipynb
│       ├── 260417_LinearRegression_2_HousePrice_v3.ipynb
│       └── 260424 - Clustering_IRIS.ipynb
├── docs/
│   ├── decisions.md                    # Decision log — source material for the report
│   └── eda-techniques-guide.md         # EDA technique reference mapped to this dataset
├── .claude/plans/
│   └── hr-pipelines.plan.md            # Implementation plan (step-by-step)
└── README.md
```

## Dataset — Key Facts

**File**: `employee_data/employee_data.csv` (path from notebooks: `../employee_data/employee_data.csv`)

**Shape**: 1249 rows × 35 columns. No nulls in the raw file, but pipelines must handle nulls for the blind test (use `SimpleImputer` inside the pipeline).

### Columns to DROP (before defining X)

| Column | Reason |
|--------|--------|
| `EmployeeNumber` | ID — no predictive value |
| `EmployeeCount` | Constant = 1 |
| `StandardHours` | Constant = 80 |
| `Over18` | Constant = 'Y' |

Also drop the **other target** when building each pipeline:
- Regression X: drop `Attrition` (and the 4 constants above)
- Classification X: drop `MonthlyIncome` (and the 4 constants above)

### Feature Types (after drops)

**Numeric** (int/float — use `SimpleImputer(median)` + `StandardScaler`):
`Age`, `DailyRate`, `DistanceFromHome`, `Education`, `EnvironmentSatisfaction`,
`HourlyRate`, `JobInvolvement`, `JobLevel`, `JobSatisfaction`, `MonthlyRate`,
`NumCompaniesWorked`, `PercentSalaryHike`, `PerformanceRating`,
`RelationshipSatisfaction`, `StockOptionLevel`, `TotalWorkingYears`,
`TrainingTimesLastYear`, `WorkLifeBalance`, `YearsAtCompany`,
`YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`

**Categorical** (object — use `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`):
`BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`

### Target Encoding

- **MonthlyIncome**: already numeric — use directly as `y` for regression.
- **Attrition**: `'Yes'` / `'No'` string — encode **outside** the pipeline:
  ```python
  y_clf = (df['Attrition'] == 'Yes').astype(int)  # 1=Yes, 0=No
  ```
  Class distribution: ~16% Yes (imbalanced) → use `class_weight='balanced'` in the classifier and `stratify=y_clf` in `train_test_split`.

### Key EDA Insights (already confirmed in employee_eda.ipynb)

- `JobLevel` ↔ `MonthlyIncome` correlation ≈ 0.95 (strongest predictor for regression)
- `TotalWorkingYears` and `Age` also correlate with `MonthlyIncome`
- `OverTime == 'Yes'` is a strong Attrition signal
- `BusinessTravel` correlates with Attrition
- Attrition is ~16% Yes → F1 (not accuracy) is the evaluation metric

## Mandatory Pipeline Architecture

Every exported `.pkl` must be a single `sklearn.pipeline.Pipeline` that handles everything internally:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    # optional: ("selector", SelectKBest(...)),
    ("model", SomeEstimator()),
])
```

The pipeline must accept a raw DataFrame with the original column names and produce predictions without any manual step.

## Serialization (mandatory format)

```python
import pickle

with open("Group08_pipeline_regression.pkl", "wb") as f:
    pickle.dump(regression_pipeline, f)

with open("Group08_pipeline_classification.pkl", "wb") as f:
    pickle.dump(classification_pipeline, f)
```

Validate before submission:

```python
with open("Group08_pipeline_regression.pkl", "rb") as f:
    reg_pipeline = pickle.load(f)
preds = reg_pipeline.predict(X_test_raw)  # raw DataFrame, no preprocessing
```

## Notebook Structure (mandatory section order)

The final deliverable notebook `[GroupX]_notebook.ipynb` must follow the ML pipeline structure with exactly these Markdown section headers:

1. `## Análises do Relatório de Dados`
2. `## Pré-processamento`
3. `## Train/Test Split`
4. `## Scaling`
5. `## Treino`
6. `## Modelos`
7. `## Otimização`
8. `## Deploy`

After each EDA command add a Markdown cell explaining what was found and what decision it drives.

## Key Conventions

- `_random_state = 24` — fixed constant, defined at the top of every notebook
- Never invent column names — always detect features from `df.columns`
- Use `pathlib.Path` for all file paths. `Group08_notebook.ipynb` is at the repo root, so the dataset path is `Path("employee_data/employee_data.csv")` (no `../`). Notebooks inside `notebooks/` use `Path("../employee_data/employee_data.csv")`.
- All preprocessing inside the `Pipeline` object — no manual transforms before `fit`/`predict`
- Ordinal columns (`Education`, `JobSatisfaction`, etc.) stay as numeric integers (already 1–4/5 scale)
- Regression metrics: R², RMSE, MAE
- Classification metrics: `classification_report`, `confusion_matrix`, ROC-AUC, **F1-score** (blind test metric)

## Models to Compare

### Regression (MonthlyIncome)
| Model | Role | Notes |
|-------|------|-------|
| `LinearRegression` | Baseline | Fast, interpretable |
| `RandomForestRegressor` | Main | Handles non-linearity well |
| `GradientBoostingRegressor` | Strong | Best expected R² |

### Classification (Attrition)
| Model | Role | Notes |
|-------|------|-------|
| `LogisticRegression(class_weight='balanced')` | Baseline | Interpretable |
| `RandomForestClassifier(class_weight='balanced')` | Main | Handles imbalance natively |
| `GradientBoostingClassifier` | Strong | Best expected F1 |

## Optimization

Use `GridSearchCV` or `RandomizedSearchCV` with `cv=5`. For classification, `scoring='f1'`; for regression, `scoring='r2'`.

Pipeline parameters are prefixed with the step name: `model__n_estimators`, `preprocessor__num__imputer__strategy`, etc.

## Deliverables Checklist

- [ ] `Group08_notebook.ipynb` — complete notebook with all 8 sections
- [ ] `Group08_pipeline_regression.pkl` — serialized regression pipeline (pickle)
- [ ] `Group08_pipeline_classification.pkl` — serialized classification pipeline (pickle)
- [ ] `Group08_report.docx` — project report (written manually by the group)

## What's Already Done

- `notebooks/employee_eda.ipynb` — EDA Section 1 (structure inspection: head, shape, columns, info, describe, nunique, dtypes, iloc, sample, rename) — 13 cells
- `docs/eda-techniques-guide.md` — comprehensive EDA technique reference guide mapped to this dataset
- `docs/decisions.md` — full decision log covering every analytical and architectural choice (source material for the report)
- `Group08_notebook.ipynb` — Phase 2 complete: header cell + global imports + `_random_state` + data loading
- Repo restructured: deliverables at root, reference material under `reference/`, working notebooks under `notebooks/`
- Group name confirmed: **Group08**

## What Remains

See `.claude/plans/hr-pipelines.plan.md` for the full step-by-step plan. Next: Phase 3 (`## Análises do Relatório de Dados` section in `Group08_notebook.ipynb`).
