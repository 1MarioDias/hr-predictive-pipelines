Group08 — Pipelines Preditivas de RH
=====================================

Entregáveis
-----------
G8_pipeline_regression.pkl     — Pipeline de regressão treinada (RandomForestRegressor)
G8_pipeline_classification.pkl — Pipeline de classificação treinada (GradientBoostingClassifier)
Group08_notebook.ipynb          — Notebook completo
Group08_report.docx             — Relatório escrito do projeto

Bibliotecas Externas (além da stdlib do Python)
----------------------------------------------
scikit-learn==1.8.0
numpy==2.4.4
pandas==3.0.2
matplotlib==3.10.3
seaborn==0.13.2

Todas as pipelines usam apenas componentes standard do scikit-learn.
Não são usadas classes personalizadas nem bibliotecas externas de ML (XGBoost, LightGBM, etc.).

Utilização
----------
import pickle, pandas as pd

with open('G8_pipeline_regression.pkl', 'rb') as f:
    reg = pickle.load(f)

with open('G8_pipeline_classification.pkl', 'rb') as f:
    clf = pickle.load(f)

# Aceita linhas brutas de DataFrame (as mesmas colunas de employee_data.csv, sem o alvo)
reg_preds = reg.predict(X_raw)   # devolve valores float de MonthlyIncome
clf_preds = clf.predict(X_raw)   # devolve strings 'Yes'/'No'

Métricas no conjunto do avaliador
---------------------------------
Regressão     R²           = 0.9564
Classificação  Macro F1     = 0.8745
Score Combinado             = 0.9154
