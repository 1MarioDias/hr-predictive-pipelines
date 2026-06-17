import os
import pickle
import pandas as pd
import numpy as np
import requests
from sklearn.metrics import r2_score, f1_score
import time

class MockRegressao:
    def predict(self, X):
        return np.random.normal(5000, 1000, len(X))

class MockClassificacao:
    def predict(self, X):
        return np.random.choice(['Yes', 'No'], len(X), p=[0.2, 0.8])
    
dados_teste = pd.read_csv('dataset.csv')
target_class = 'Attrition'
target_reg = 'MonthlyIncome'

X_class = dados_teste.drop([target_class], axis=1)
y_real_class = dados_teste[target_class]

X_reg = dados_teste.drop([target_reg], axis=1)
y_real_reg = dados_teste[target_reg]

grupos_pipelines = {
    'Grupo 0': {
        'reg': 'modelos/G0_pipeline_regression.pkl', 
        'class': 'modelos/G0_pipeline_classification.pkl'
    },
     'Grupo 1': {
         'reg': 'modelos/G1_pipeline_regression.pkl', 
         'class': 'modelos/G1_pipeline_classification.pkl'
    },
     'Grupo 2': {
         'reg': 'modelos/G2_pipeline_regression.pkl', 
         'class': 'modelos/G2_pipeline_classification.pkl'
    },
     'Grupo 3': {
         'reg': 'modelos/G3_pipeline_regression.pkl', 
         'class': 'modelos/G3_pipeline_classification.pkl'
    },
     'Grupo 4': {
         'reg': 'modelos/G4_pipeline_regression.pkl', 
         'class': 'modelos/G4_pipeline_classification.pkl'
    },
     'Grupo 5': {
         'reg': 'modelos/G5_pipeline_regression.pkl', 
         'class': 'modelos/G5_pipeline_classification.pkl'
    },
     'Grupo 6': {
         'reg': 'modelos/G6_pipeline_regression.pkl', 
         'class': 'modelos/G6_pipeline_classification.pkl'
    },
     'Grupo 7': {
         'reg': 'modelos/G7_pipeline_regression.pkl', 
         'class': 'modelos/G7_pipeline_classification.pkl'
    },
     'Grupo 8': {
         'reg': 'modelos/G8_pipeline_regression.pkl', 
         'class': 'modelos/G8_pipeline_classification.pkl'
    },
     'Grupo 9': {
         'reg': 'modelos/G9_pipeline_regression.pkl', 
         'class': 'modelos/G9_pipeline_classification.pkl'
    },
     'Grupo 10': {
         'reg': 'modelos/G10_pipeline_regression.pkl', 
         'class': 'modelos/G10_pipeline_classification.pkl'
    },
     'Grupo 11': {
         'reg': 'modelos/G11_pipeline_regression.pkl', 
         'class': 'modelos/G11_pipeline_classification.pkl'
    },
     'Grupo 12': {
         'reg': 'modelos/G12_pipeline_regression.pkl', 
         'class': 'modelos/G12_pipeline_classification.pkl'
    },
     'Grupo 13': {
         'reg': 'modelos/G13_pipeline_regression.pkl', 
         'class': 'modelos/G13_pipeline_classification.pkl'
    },
     'Grupo 14': {
         'reg': 'modelos/G14_pipeline_regression.pkl', 
         'class': 'modelos/G14_pipeline_classification.pkl'
    },
     'Grupo 15': {
         'reg': 'modelos/G15_pipeline_regression.pkl', 
         'class': 'modelos/G15_pipeline_classification.pkl'
    },
     'Grupo 16': {
         'reg': 'modelos/G16_pipeline_regression.pkl', 
         'class': 'modelos/G16_pipeline_classification.pkl'
    },
     'Grupo 17': {
         'reg': 'modelos/G17_pipeline_regression.pkl', 
         'class': 'modelos/G17_pipeline_classification.pkl'
    }
}

URL_LEADERBOARD = "http://localhost:5000/submeter"

for grupo, ficheiros in grupos_pipelines.items():
    try:
        print(f"\nAvaliação do {grupo}...")
        
        # --- REGRESSÃO ---
        with open(ficheiros['reg'], 'rb') as f_reg:
            pipeline_reg = pickle.load(f_reg)
        pred_reg = pipeline_reg.predict(X_reg)
        score_reg = r2_score(y_real_reg, pred_reg)
        
        # --- CLASSIFICAÇÃO ---
        with open(ficheiros['class'], 'rb') as f_class:
            pipeline_class = pickle.load(f_class)
        pred_class = pipeline_class.predict(X_class)
        score_class = f1_score(y_real_class, pred_class, average='macro') 
        
        # Métrica combinada para o Leaderboard
        # Média aritmética simples para o ranking geral
        score_final = (score_reg + score_class) / 2
        
        print(f"-> Regressão (R2): {score_reg:.4f} | Classificação (F1): {score_class:.4f}")
        print(f"-> Score Combinado Final: {score_final:.4f}")
        
        # Enviar dados para a API do Leaderboard
        payload = {
            'grupo': grupo, 
            'score': round(score_final, 4),
            'score_reg': round(score_reg, 4),
            'score_class': round(score_class, 4)
        }
        requests.post(URL_LEADERBOARD, json=payload)
        
    except Exception as e:
        print(f"Erro crítico ao avaliar o {grupo}: {e}")
    time.sleep(1)