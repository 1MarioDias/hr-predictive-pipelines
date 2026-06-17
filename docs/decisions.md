# Registo de Decisões — Group08 HR Predictive Pipelines

Este documento acompanha o desenvolvimento do projeto e serve de base para a redação do `Group08_report.docx`. Cada secção corresponde a uma fase do trabalho e explica as escolhas feitas, com os valores reais obtidos durante a execução.

---

## 1. Análise do Dataset

O ficheiro `employee_data.csv` contém 1249 registos e 35 colunas, sem nenhum valor nulo. Os dois alvos estão presentes: `MonthlyIncome` (inteiro contínuo, alvo da regressão) e `Attrition` (string "Yes"/"No", alvo da classificação). A distribuição de `Attrition` é bastante desequilibrada: 1048 registos com "No" e 201 com "Yes", o que representa aproximadamente 16% de saídas.

> **[Figura 1 — Notebook, secção "Análises do Relatório de Dados"]**
> Histograma de `MonthlyIncome` com curva KDE e gráfico de barras com contagens e percentagens de `Attrition`. Inserir aqui no relatório.

A análise de correlação entre variáveis numéricas revelou que `JobLevel` tem uma correlação muito elevada com `MonthlyIncome` (aproximadamente 0.95), o que é o sinal mais forte para a regressão. `TotalWorkingYears` e `Age` também contribuem de forma significativa.

> **[Figura 2 — Notebook, secção "Análises do Relatório de Dados"]**
> Heatmap de correlação entre todas as variáveis numéricas. Inserir aqui no relatório.

Nas variáveis categóricas, o cruzamento de `OverTime` com `Attrition` mostrou que funcionários com horas extra têm uma taxa de saída de 29%, contra 11% nos restantes. `BusinessTravel` apresenta um padrão semelhante, com quem viaja frequentemente a sair mais do que quem não viaja.

> **[Figura 3 — Notebook, secção "Análises do Relatório de Dados"]**
> Gráficos de barras empilhadas com a percentagem de Attrition por `OverTime` e por `BusinessTravel`. Inserir aqui no relatório.

Quatro colunas foram removidas antes da construção das features: `EmployeeNumber` é um identificador único sem qualquer sinal preditivo; `EmployeeCount` e `StandardHours` têm um único valor em todas as linhas e portanto variância zero; `Over18` contém sempre "Y". Em cada pipeline, o alvo da outra tarefa também é excluído de X para evitar fuga de informação.

---

## 2. Pré-processamento e Arquitectura dos Pipelines

Após os drops, ficaram 29 colunas de features. A deteção automática dos tipos por `df.dtypes` produziu 22 variáveis numéricas e 7 categóricas.

**Numéricas (22):** Age, DailyRate, DistanceFromHome, Education, EnvironmentSatisfaction, HourlyRate, JobInvolvement, JobLevel, JobSatisfaction, MonthlyRate, NumCompaniesWorked, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager.

**Categóricas (7):** BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime.

As variáveis ordinais como `Education` e `JobSatisfaction` estão codificadas como inteiros de 1 a 4 ou 5 e são tratadas como numéricas. Os modelos de árvore não assumem distância entre categorias, por isso não foi necessário aplicar um encoder ordinal dedicado.

O pré-processamento foi encapsulado num `ColumnTransformer` com dois ramos. Para as variáveis numéricas: `SimpleImputer(strategy="median")` seguido de `StandardScaler`. Para as categóricas: `SimpleImputer(strategy="most_frequent")` seguido de `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. O uso de mediana (em vez de média) nos imputadores numéricos torna o pipeline robusto a valores extremos, que são comuns em dados salariais. O parâmetro `handle_unknown="ignore"` no encoder é obrigatório para o blind test: se o conjunto de avaliação contiver uma categoria não vista em treino (por exemplo, um novo `JobRole`), o encoder produz um vetor de zeros em vez de lançar um erro.

Depois do `ColumnTransformer`, cada pipeline inclui um `SelectKBest` com k=20 como passo de seleção de features. O valor de k é pesquisável no `RandomizedSearchCV`, o que permite que a otimização encontre o número ideal de features para cada modelo. O `StandardScaler` está dentro do pipeline e é ajustado apenas sobre `X_train`, o que evita que a informação do conjunto de teste contamine o treino.

A estrutura final de cada pipeline exportável é:

```
Pipeline
 ├── preprocessor  (ColumnTransformer)
 │    ├── num: SimpleImputer(median) → StandardScaler        [22 features]
 │    └── cat: SimpleImputer(most_frequent) → OneHotEncoder  [7 → 28 colunas]
 ├── selector      (SelectKBest, k configurável)
 └── model         (estimador)
```

Os pipelines não contêm nenhuma classe externa ao scikit-learn. Esta decisão foi tomada para garantir compatibilidade com o avaliador, que carrega os ficheiros com `pickle.load()` num ambiente onde classes personalizadas não estão disponíveis.

---

## 3. Divisão Treino/Teste

A divisão foi feita com `test_size=0.2` e `random_state=24`. Para a classificação, o parâmetro `stratify=y_clf` garante que a proporção de 16% "Yes" é mantida em ambos os conjuntos. O treino ficou com 999 amostras e o teste com 250, com 161 "Yes" no treino e 40 no teste.

---

## 4. Regressão — Comparação de Modelos

Foram treinados três modelos com a mesma estrutura de pipeline. O `LinearRegression` serve como baseline; o `RandomForestRegressor` e o `GradientBoostingRegressor` capturam relações não lineares que o modelo linear não consegue representar.

| Modelo | R² | RMSE | MAE |
|---|---|---|---|
| LinearRegression | 0.9277 | 1134.48 | 882.22 |
| RandomForestRegressor | 0.9399 | 1034.38 | 769.58 |
| GradientBoostingRegressor | 0.9398 | 1034.88 | 770.19 |

O `LinearRegression` já obtém um R² de 0.93, o que é explicado pela correlação quase linear entre `JobLevel` e `MonthlyIncome`. Os modelos de ensemble melhoram ligeiramente ao capturar interações entre variáveis como `TotalWorkingYears`, `Age` e `JobRole`. O `RandomForestRegressor` ficou em primeiro lugar e foi selecionado para a otimização de hiperparâmetros.

> **[Figura 4 — Notebook, secção "Modelos"]**
> Três gráficos de resíduos lado a lado (previsão vs. resíduo) para os três modelos de regressão. Inserir aqui no relatório.

---

## 5. Classificação — Comparação de Modelos

Para a classificação, o alvo `y_clf` preserva os rótulos originais "Yes" e "No" em formato de string. Esta escolha é necessária porque o avaliador calcula `f1_score(y_real, pred, average='macro')` com strings, e um pipeline treinado com rótulos de string produz predições de string directamente, sem conversão manual.

A métrica de avaliação é o Macro F1, que calcula o F1 de cada classe separadamente e faz a média aritmética. Com 16% de "Yes", um modelo que apenas preveja "No" obteria um F1 alto na classe maioritária mas zero na minoritária, resultando num Macro F1 baixo. Esta métrica penaliza esse comportamento.

Foram treinados três modelos:

| Modelo | Macro F1 | Accuracy | ROC-AUC |
|---|---|---|---|
| LogisticRegression | 0.6507 | 0.740 | 0.8018 |
| GradientBoostingClassifier | 0.7088 | 0.880 | 0.8119 |
| RandomForestClassifier | 0.6371 | 0.864 | 0.8009 |

O `GradientBoostingClassifier` obteve o melhor Macro F1 e foi selecionado para otimização. O `LogisticRegression` com `class_weight='balanced'` serve de baseline: este parâmetro ajusta os pesos das classes inversamente à sua frequência, o que evita que o modelo ignore completamente a classe minoritária. O `RandomForestClassifier` usa o mesmo parâmetro. O `GradientBoostingClassifier` não suporta `class_weight`, mas o algoritmo de boosting corrige iterativamente os erros cometidos nas iterações anteriores, o que tem um efeito parcialmente equivalente.

> **[Figura 5 — Notebook, secção "Modelos"]**
> Matrizes de confusão dos três modelos de classificação com o Macro F1 de cada um no título. Inserir aqui no relatório.

---

## 6. Otimização de Hiperparâmetros

### Regressão

Foi usada a `RandomizedSearchCV` com 20 iterações, validação cruzada de 5 folds e `scoring='r2'`. O espaço de pesquisa cobriu `n_estimators`, `max_depth` e `min_samples_split`.

Melhores hiperparâmetros encontrados:

```
n_estimators   = 200
max_depth      = 5
min_samples_split = 5
```

Melhor R² em validação cruzada: **0.9507**. No conjunto de teste: **R² = 0.9417**, RMSE = 1018.26, MAE = 742.03.

### Classificação

Foi usada a `RandomizedSearchCV` com 50 iterações, `StratifiedKFold(n_splits=5, shuffle=True, random_state=24)` e `scoring='f1_macro'`. O `StratifiedKFold` é necessário porque com apenas 16% de "Yes", um split aleatório pode produzir folds com proporções muito diferentes, tornando os scores de validação cruzada pouco estáveis. O parâmetro `selector__k` foi incluído no espaço de pesquisa para que o número de features selecionadas também pudesse ser otimizado.

Melhores hiperparâmetros encontrados:

```
selector__k        = all (todas as 50 features após o ColumnTransformer)
n_estimators       = 300
max_depth          = 3
learning_rate      = 0.05
subsample          = 0.8
min_samples_leaf   = 1
```

Melhor Macro F1 em validação cruzada: resultado obtido com `scoring='f1_macro'`. No conjunto de teste: **Macro F1 = 0.7106**, F1(Yes) = 0.4918, F1(No) = 0.9294, ROC-AUC = 0.7849.

A evolução ao longo do projeto foi a seguinte:

| Fase | Modelo | Macro F1 (conjunto de teste local) |
|---|---|---|
| Baseline | LogisticRegression | 0.6507 |
| Melhor inicial (sem otimização) | GradientBoosting | 0.7088 |
| Após RandomizedSearchCV (scoring='f1_macro') | GradientBoosting otimizado | 0.7106 |

---

## 7. Serialização e Validação

Os dois pipelines finais foram serializados com `pickle` para os ficheiros `G8_pipeline_regression.pkl` e `G8_pipeline_classification.pkl`. A nomenclatura segue o formato esperado pelo avaliador (`modelos/G8_pipeline_*.pkl`).

A validação foi feita carregando os ficheiros do disco e chamando `.predict()` directamente sobre os dados crus, sem qualquer pré-processamento externo. Os resultados coincidem com os da fase de otimização. Foi também testado o comportamento com valores nulos injectados artificialmente: o `SimpleImputer` dentro do pipeline trata-os sem erros.

No conjunto de avaliação do professor (`dataset.csv`), os resultados foram:

| Tarefa | Métrica | Valor |
|---|---|---|
| Regressão | R² | **0.9564** |
| Classificação | Macro F1 | **0.8745** |
| Combinado | (R² + Macro F1) / 2 | **0.9154** |

---

## 8. Utilização de IA Generativa

Claude Code (claude-sonnet-4-6) foi usado como assistente de desenvolvimento ao longo de todo o projeto. As contribuições principais foram a estrutura do notebook e os comentários de EDA, a arquitectura do `ColumnTransformer` + `Pipeline`, a configuração do `RandomizedSearchCV` com `StratifiedKFold` e `scoring='f1_macro'`, e a redação deste registo de decisões. Todas as decisões analíticas foram revistas e validadas pelo grupo antes de serem incorporadas no trabalho final.
