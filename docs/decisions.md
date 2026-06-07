# Decision Log — Group08 HR Predictive Pipelines

This file records every analytical and architectural decision made during the project, with the reasoning behind each. It is the primary source for writing `Group08_report.docx`.

---

## 1. Dataset

**File**: `employee_data/employee_data.csv` — 1249 rows × 35 columns.  
**No nulls** in the provided file. `SimpleImputer` is still placed inside each pipeline because the blind-test evaluator may send rows with missing values.

---

## 2. Columns Dropped from X

| Column | Reason |
|---|---|
| `EmployeeNumber` | Unique ID per row — no predictive signal, would cause overfitting |
| `EmployeeCount` | Constant value = 1 across all rows — zero variance |
| `StandardHours` | Constant value = 80 across all rows — zero variance |
| `Over18` | Constant value = 'Y' across all rows — zero variance |

When building each pipeline, the **other target** is also excluded from X:
- Regression X: additionally drops `Attrition`
- Classification X: additionally drops `MonthlyIncome`

---

## 3. Feature Types

Determined by `df.dtypes` after the column drops above.

**Numeric** (`int64` / `float64`) — processed with `SimpleImputer(median)` + `StandardScaler`:
`Age`, `DailyRate`, `DistanceFromHome`, `Education`, `EnvironmentSatisfaction`,
`HourlyRate`, `JobInvolvement`, `JobLevel`, `JobSatisfaction`, `MonthlyRate`,
`NumCompaniesWorked`, `PercentSalaryHike`, `PerformanceRating`,
`RelationshipSatisfaction`, `StockOptionLevel`, `TotalWorkingYears`,
`TrainingTimesLastYear`, `WorkLifeBalance`, `YearsAtCompany`,
`YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`

**Categorical** (`object`) — processed with `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`:
`BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`

**Note on ordinal columns** (`Education`, `JobSatisfaction`, `EnvironmentSatisfaction`, etc.): stored as integers on a 1–4/5 scale, treated as numeric. Tree-based models (the primary models here) are invariant to monotonic feature transformations, so the ordinal structure is preserved without a dedicated encoder.

---

## 4. Preprocessing Architecture

Every pipeline follows a three-step structure:

```
Pipeline
 ├── preprocessor  (ColumnTransformer)
 │    ├── num: SimpleImputer(median) → StandardScaler
 │    └── cat: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
 ├── selector      (SelectKBest — k=20)          ← variable selection
 └── model         (estimator)
```

The `ColumnTransformer` used as `preprocessor`:
```
preprocessor = ColumnTransformer([
    ("num", Pipeline([SimpleImputer(median), StandardScaler()]), NUMERIC_FEATURES),
    ("cat", Pipeline([SimpleImputer(most_frequent), OneHotEncoder(handle_unknown="ignore")]), CATEGORICAL_FEATURES),
])
```

**Why `handle_unknown="ignore"`**: the blind test may include employees with categorical values not seen during training (e.g. a new `JobRole`). Without this flag the pipeline raises a `ValueError` at inference time. With it, unseen categories produce a zero vector — a safe, neutral representation.

**Why `sparse_output=False`**: avoids deprecation warnings in scikit-learn ≥ 1.2 and simplifies downstream debugging.

**Why scaling inside the pipeline**: fitting `StandardScaler` on the full dataset before the train/test split leaks test-set statistics into training. Placing the scaler inside the pipeline ensures it is fit only on `X_train` and applied to `X_test`.

---

## 5. Target Encoding

### Regression — `MonthlyIncome`
Already numeric (`int64`). Used directly as `y_reg = df['MonthlyIncome'].astype(float)`.

### Classification — `Attrition`
String categories `'Yes'` / `'No'`. Encoded **outside** the pipeline:
```python
y_clf = (df['Attrition'] == 'Yes').astype(int)  # 1 = churned, 0 = stayed
```
Encoding outside the pipeline is correct because `y` is never passed through `predict()` — only `X` flows through the pipeline, so there is no leakage risk.

---

## 6. Train / Test Split

- `test_size=0.2`, `random_state=24`
- Classification split uses `stratify=y_clf` because Attrition is imbalanced (~16% Yes). Without stratification the test set could contain a very different class ratio than training, making evaluation metrics unreliable.

---

## 7. Class Imbalance — Attrition (~16% Yes)

Attrition = 'Yes' is ~16% of rows (≈ 201 out of 1249). Two decisions address this:

1. **`class_weight='balanced'`** on classifiers that support it (`LogisticRegression`, `RandomForestClassifier`). Adjusts loss weights inversely proportional to class frequencies, penalising minority-class misclassification more heavily.
2. **Evaluation metric = F1-score**. Accuracy is misleading on imbalanced data — a model that always predicts "No" achieves ~84% accuracy while being useless. F1 balances precision and recall for the positive (churn) class, which is what the blind test scores on.

---

## 8. Models Evaluated

### Regression (MonthlyIncome)

| Model | Role | Rationale |
|---|---|---|
| `LinearRegression` | Baseline | Fast, interpretable, establishes a lower-bound metric |
| `RandomForestRegressor` | Main | Captures non-linear interactions (e.g. `JobLevel` × `TotalWorkingYears`); robust to outliers |
| `GradientBoostingRegressor` | Strong | Sequential error correction; typically best R² on tabular data |

`JobLevel` ↔ `MonthlyIncome` correlation ≈ 0.95 (confirmed via EDA). Even the linear baseline is expected to perform well.

### Classification (Attrition)

| Model | Role | Rationale |
|---|---|---|
| `LogisticRegression(class_weight='balanced', max_iter=1000)` | Baseline | Interpretable; `max_iter=1000` avoids convergence warnings after scaling |
| `RandomForestClassifier(class_weight='balanced')` | Main | Native class weighting; ensemble reduces variance |
| `GradientBoostingClassifier` | Strong | Best expected F1; handles imbalance well |

---

## 9. Hyperparameter Optimisation

**Method**: `RandomizedSearchCV`, `cv=5`, `n_iter=20`, `n_jobs=-1`.  
**Why Randomized over Grid**: exhaustive grid search over the parameter space would take hours. Randomized search finds near-optimal configurations in a fraction of the time, which is well-supported in the literature for tabular ML.

**Scoring**: `'r2'` for regression, `'f1'` for classification.

**Search space (regression — RandomForestRegressor)**:
- `model__n_estimators`: [100, 200, 300]
- `model__max_depth`: [None, 5, 10]
- `model__min_samples_split`: [2, 5]

**Search space (classification — RandomForestClassifier)**:
- `model__n_estimators`: [100, 200, 300]
- `model__max_depth`: [None, 5, 10]
- `model__class_weight`: ['balanced']

---

## 10. Serialization

Both final pipelines are serialized with Python's native `pickle`. Validated by loading from disk and calling `.predict()` on raw `X_test` with no manual preprocessing. A NaN-injection test also confirms `SimpleImputer` handles missing values at inference time.

**sklearn version**: will be added here once confirmed. `.pkl` files are only guaranteed compatible within the same major.minor sklearn version.

---

## 11. Random State

`_random_state = 24` is used for all `train_test_split`, model constructors, and `RandomizedSearchCV` calls to ensure full reproducibility.

---

## 12. Implementation Progress

| Phase | Section in notebook | Status |
|---|---|---|
| Phase 2 | Header + imports + data load | ✅ Done |
| Phase 3 | `## Análises do Relatório de Dados` | ✅ Done |
| Phase 4 | `## Pré-processamento` | ✅ Done |
| Phase 5 | `## Train/Test Split` | ✅ Done |
| Phase 6 | `## Scaling` | ✅ Done |
| Phase 7 | `## Treino` | ✅ Done |
| Phase 8 | `## Modelos` | ✅ Done |
| Fix: SelectKBest added to all pipelines | Variable selection step (k=20) | ✅ Done |
| Fix: Treino/Modelos redundancy | Modelos loop seeded with Treino baselines | ✅ Done |
| Phase 9 | `## Otimização` | ⬜ Pending (colleagues) |
| Phase 10 | `## Deploy` | ⬜ Pending |
| Phase 11 | Final validation + pkl files | ⬜ Pending |

---

## 13. Preprocessing — Additional Decisions (Phase 4)

### `PerformanceRating` — kept despite low variance
EDA confirmed std=0.36 with only values 3 and 4. Not fully constant, so it is not dropped. Tree-based models select features by information gain at each split — a near-constant feature simply never wins a split and is effectively ignored. Removing it manually would be premature optimisation with no measurable benefit.

### `remainder="drop"` in ColumnTransformer
Added explicitly so that any unexpected column arriving in the blind test is silently discarded instead of causing a shape mismatch error.

### Feature lists auto-detected from dtypes
`NUMERIC_FEATURES` and `CATEGORICAL_FEATURES` are derived from `X_reg.select_dtypes(...)` at runtime. Both tasks share the same lists because `X_reg` and `X_clf` have identical columns after dropping their respective targets.

### `SimpleImputer(strategy="median")` for numerics — not mean
`MonthlyIncome` has a right-skewed distribution (confirmed in EDA). Mean imputation would pull values toward the high-salary tail; median is more representative of the typical employee. Applied to all numeric features for consistency.

### `preprocessor` fit only on `X_train` inside each pipeline
The sanity-check cell in the notebook fits on full `X_reg` only to verify the output shape. In both final pipelines the `ColumnTransformer` is re-fit strictly on `X_train` via `pipeline.fit(X_train, y_train)` — no test-set information is ever seen during fit.

---

## 14. Phase 8 — Modelos

### Loop-based training pattern
All models are trained inside a `for` loop over a list of `(name, estimator)` tuples. Each iteration builds a fresh `Pipeline([clone(preprocessor), model])`, fits on `X_train`, predicts on `X_test`, and stores results in `reg_pipes` / `clf_pipes` dicts. This keeps Phase 8 self-contained and makes it trivial to add or swap models.

### Regression — model selection rationale

| Model | Key property | Expected behaviour |
|---|---|---|
| `LinearRegression` | Assumes linear relationship | High R² expected given JobLevel↔MonthlyIncome ≈ 0.95, but residuals will show heteroscedasticity for high earners |
| `RandomForestRegressor(n_estimators=100)` | Bagged decision trees | Captures non-linear interactions; robust to outliers |
| `GradientBoostingRegressor(n_estimators=100)` | Sequential residual correction | Typically best R² on tabular data; slower to train |

Default `n_estimators=100` used at this stage — tuned in Phase 9.

### Classification — `GradientBoostingClassifier` has no `class_weight`
`GradientBoostingClassifier` does not accept `class_weight`. The boosting algorithm implicitly down-weights well-classified examples in each iteration, which partially compensates for imbalance — but less directly than `class_weight='balanced'`. If `GradientBoosting` wins on F1 despite this, it proceeds to optimisation; otherwise `RandomForest` (which has explicit weighting) is preferred.

### Comparison tables sorted by primary metric
- Regression: sorted by R² descending
- Classification: sorted by F1 descending

The model at the top of each table is the candidate for `RandomizedSearchCV` in Phase 9.

### Confusion matrix title includes F1
Each confusion matrix subplot shows the model name and its F1 score in the title, making it easy to correlate the visual distribution of errors with the scalar metric.

---

## 15. Variable Selection — SelectKBest

The project statement explicitly lists *"variable selection"* as a required pipeline step alongside imputation, encoding, and scaling. After the `ColumnTransformer` produces 50 features (22 numeric + 28 OHE columns), a `SelectKBest` selector reduces that to the 20 most statistically relevant.

**Score functions chosen by task:**
- Regression: `f_regression` — computes the F-statistic for the linear relationship between each feature and `MonthlyIncome`. Appropriate for a continuous target.
- Classification: `f_classif` — computes the ANOVA F-statistic between each feature and the binary `Attrition` target. Standard choice for classification with a categorical target.

**Why k=20 (out of 50 post-OHE features):**
- Retains all features that showed meaningful correlation or EDA signal (e.g. `JobLevel`, `TotalWorkingYears`, `OverTime`, `BusinessTravel`).
- Discards OHE columns for low-signal categories (e.g. infrequent `EducationField` or `Department` splits) and near-constant numerics like `PerformanceRating`.
- Conservative enough not to hurt performance on a 50-feature space; aggressive enough to demonstrate variable selection awareness.

**How it integrates with the pipeline:**
```python
REG_SELECTOR = SelectKBest(f_regression, k=20)
CLF_SELECTOR = SelectKBest(f_classif, k=20)

# Each pipeline:
Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("selector", clone(REG_SELECTOR)),   # or CLF_SELECTOR
    ("model", estimator),
])
```

`clone()` is used for both `preprocessor` and `selector` so every pipeline instance has its own unfitted copies — no shared state across the comparison loop.

**GridSearchCV / RandomizedSearchCV compatibility:** the selector step is accessible via `selector__k` in the parameter grid, making it straightforward to also search over `k` during optimisation if desired.

---

## 16. Treino / Modelos Redundancy Resolution

The `## Treino` section trained two baseline models (`pipe_lr`, `pipe_logreg`) in isolation as a demonstration of the training workflow. The `## Modelos` section then re-trained those same models inside a comparison loop — creating redundancy and two different fitted objects for the same model.

**Resolution chosen (Option A — minimal change):** The `## Modelos` comparison loop is seeded with the already-trained baseline pipes from `## Treino` before iterating over the remaining models:

```python
# Seed with baselines from ## Treino — no redundant refit
reg_pipes = {"LinearRegression": (pipe_lr, y_reg_pred_lr)}
clf_pipes = {"LogisticRegression": (pipe_logreg, y_clf_pred_logreg, y_clf_prob_logreg)}

# Loop only over non-baseline models
for name, estimator in [(...)]:   # RandomForest, GradientBoosting
    ...
```

**Why Option A over Option B:** minimises the diff to the colleague's existing cells in `## Treino`, which already have output cached. Option B (removing Treino training cells entirely) would invalidate cached outputs and break the narrative of introducing the workflow before the full comparison.

---

## 17. Phases 5–7 Decisions

### Phase 5 — Train/Test Split

**80/20 split** with `test_size=0.2`, `random_state=24`. 999 training samples / 250 test samples.

**`stratify=y_clf` for classification only.** With ~16% minority class, a random split risks distributing the 201 Yes examples unevenly between train and test. Stratification ensures both sets maintain the original 16% ratio, making the F1 evaluation on the test set a fair reflection of real-world performance. Regression target (`MonthlyIncome`) is continuous so stratification does not apply.

### Phase 6 — Scaling

Scaling is entirely inside the pipeline — no standalone scaling step exists in the notebook. This is an explicit architectural decision: `StandardScaler.fit()` called on full data before the split would leak test-set statistics (mean, std) into training, constituting data leakage. Placing it inside the `Pipeline` ensures fit happens only on `X_train`.

`StandardScaler` is critical for `LinearRegression` and `LogisticRegression` (scale-sensitive models). For tree-based models it has no effect on predictions but does not hurt, keeping the pipeline consistent across all model types.

### Phase 7 — Treino (Baseline Models)

**`clone(preprocessor)` for each pipeline.** `sklearn.Pipeline` does not deep-copy its steps — passing the same `preprocessor` object to multiple pipelines would share fitted state across them. `clone()` creates an unfitted copy with the same hyperparameters, giving each pipeline its own independent preprocessor. This avoids subtle bugs when fitting multiple pipelines in the same session.

**Helper functions `regression_metrics` and `classification_metrics`** defined once and reused across Phases 7, 8, and 9 to ensure consistent metric reporting. `classification_metrics` accepts an optional `y_prob` argument to report ROC-AUC alongside F1.

**Regression baseline — `LinearRegression`:**
Expected to perform well given `JobLevel`↔`MonthlyIncome` ≈ 0.95 correlation. Residuals plot included to detect heteroscedasticity or non-linearity that would justify the ensemble models in Phase 8.

**Classification baseline — `LogisticRegression(class_weight='balanced', max_iter=1000)`:**
`class_weight='balanced'` is mandatory for the imbalanced Attrition target. `max_iter=1000` prevents non-convergence warnings after StandardScaler (scaled features converge faster but the default 100 iterations is sometimes still insufficient). Confusion matrix shown to quantify false negatives (churned employees misclassified as staying) — the most costly error type from a business perspective.

---

## 18. GenAI Usage (for report section)

Claude Code (claude-sonnet-4-6) was used as a development assistant. Key uses:
- Structuring the EDA notebook and commentary templates
- Designing the `ColumnTransformer` + `Pipeline` architecture
- Drafting this decision log
- Implementing notebook sections

All analytical decisions (model selection, evaluation metric, class imbalance handling, drop strategy) were reviewed and validated by the group.
