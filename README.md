# HR Predictive Pipelines

This repository contains a reproducible end-to-end machine learning solution for an employee predictive modeling project. It includes two independent Scikit-learn Pipelines built on the same dataset: one for monthly salary prediction and one for employee attrition prediction.

## Overview

In modern web systems, AI models should not depend on manual preprocessing steps or the original development environment. For that reason, this project uses Scikit-learn Pipelines to encapsulate the full workflow, from raw input data to final predictions.

The goal is to deliver portable and reproducible models that can be evaluated in a blind test scenario.

## Project Goals

This project includes two predictive tasks:

- **Regression model**: predict `MonthlyIncome`
- **Classification model**: predict `Attrition`

Both models must be built using the same dataset and exported as complete pipelines.

## Technical Requirements

To ensure full portability, each model must be exported as a single serialized Scikit-learn Pipeline.

Each pipeline should include the full preprocessing and modeling flow, such as:

- Missing value imputation
- Categorical encoding
- Feature scaling
- Feature selection
- Final estimator

The final exported file should behave like a black box: it must accept raw input data in its original format and generate predictions without manual intervention.

## Deliverables

The following files must be delivered:

- `[GroupX]_pipeline_regression.pkl` — trained pipeline for the regression problem
- `[GroupX]_pipeline_classification.pkl` — trained pipeline for the classification problem
- `[GroupX]_notebook.ipynb` — Jupyter Notebook with EDA, validation, model comparison, hyperparameter tuning, and theoretical notes
- `[GroupX]_report.docx` — detailed report describing the project execution, decisions taken, tested models, reasoning, and Generative AI prompts used

## Pipeline Design

The use of Scikit-learn `Pipeline` is mandatory. Models and preprocessing steps should not be exported as separate files.

The expected pipeline may include:

- `SimpleImputer` for missing values
- `OneHotEncoder(handle_unknown="ignore")` for categorical features
- `StandardScaler` or `MinMaxScaler` for numerical features
- Optional feature selection
- Final regression or classification algorithm

This approach ensures that the exported pipeline can be used safely in the blind test without requiring manual preprocessing.

## Suggested Workflow

1. Perform Exploratory Data Analysis (EDA)
2. Clean and prepare the dataset
3. Define preprocessing pipelines for numerical and categorical variables
4. Build complete regression and classification pipelines
5. Compare multiple algorithms
6. Optimize hyperparameters using techniques such as:
   - `GridSearchCV`
   - `RandomizedSearchCV`
7. Validate performance on the test dataset
8. Serialize the final pipelines using `pickle`

## Evaluation Criteria

The project will be evaluated according to the following criteria:

### 1. Data Engineering and Pipeline Quality
- Robust missing value handling
- Appropriate encoding strategy
- Correct feature selection
- Safe handling of unknown categories

### 2. Exploration and Scientific Methodology
- Quality and depth of EDA
- Comparison of multiple algorithms
- Proper hyperparameter optimization

### 3. Blind Test Performance
The evaluation metrics are:

- **Regression**: `R2` and `RMSE`
- **Classification**: `F1-score`

### 4. Code Organization and Documentation
- Correct file naming
- Readable and well-structured notebook
- Clear documentation and report

### 5. Oral Defense
- Ability to justify preprocessing choices
- Ability to explain algorithm selection
- Understanding of hyperparameters and validation strategy

## Serialization

All final models must be serialized using Python's native `pickle` library.

Example:

```python
import pickle

with open("[GroupX]_pipeline_regression.pkl", "wb") as f:
    pickle.dump(regression_pipeline, f)
```

## Validation

Before submission, both pipelines should be tested with the provided test dataset to ensure that:

- The file loads correctly
- Raw data is accepted without manual preprocessing
- Predictions are generated successfully
- No runtime errors occur

## Generative AI Use

The use of Generative AI tools is allowed as a support mechanism for code optimization and development assistance. However, all analytical decisions must remain under full team control.

Any prompts used with Generative AI tools should be documented in the final report.

## Notes

- Do not export preprocessing steps separately from the model
- Do not rely on notebook-only variables or manual transformations
- Keep the project reproducible and organized
- Make sure the final exported pipelines are presentation-ready for the blind test

## Deadline

- **Submission deadline**: June 18, 2026, at 11:59 PM
- **Presentation and defense**: June 19, 2026, from 9:00 AM

## Repository Description

Reproducible end-to-end Scikit-learn pipelines for employee salary regression and attrition classification. Designed for blind-test evaluation with robust preprocessing, model validation, and portable serialized pipelines.