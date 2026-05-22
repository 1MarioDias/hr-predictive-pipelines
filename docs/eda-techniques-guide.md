# Guia de técnicas de EDA — Aulas → Projeto HR

Referência das técnicas vistas nos notebooks das aulas (`Notebooks aulas-20260522/`), mapeadas para o dataset `employee_data/employee_data.csv` e para os requisitos do projeto ([AI-Project-2526.pdf](../AI-Project-2526.pdf)).

---

## 1. Inspeção e estrutura (todos os notebooks)

| Técnica | Onde (aula) | Para o HR |
|---------|-------------|-----------|
| `head()` / `display()` | Todos | Primeiras linhas de `employee_data.csv` |
| `shape` | PandasPrimer | Dimensão (1249 × 35) |
| `columns` | HousePrice | Listar nomes das colunas |
| `info()` | Todos | Tipos, non-null count, memória |
| `describe()` | Todos | Estatísticas numéricas (média, std, quartis, min, max) |
| `describe(exclude=np.number)` | TelecomChurn | Estatísticas de colunas categóricas |
| `dtypes` | Titanic | Confirmar int vs object |
| `nunique()` | Titanic, Telecom | Cardinalidade por coluna (detectar IDs, constantes) |
| Seleção por índice `iloc[[0,50,100]]` | DT iris | Ver uma linha por “classe” (ex.: Attrition Yes/No) |
| `sample()` / `sample(frac=1)` | PandasPrimer, DT iris | Amostra aleatória ou embaralhar antes do split |
| Renomear colunas `rename()` | HousePrice | Nomes mais curtos (`MonthlyIncome` → `income`) |

---

## 2. Valores em falta

| Técnica | Onde (aula) | Para o HR |
|---------|-------------|-----------|
| `isnull().sum()` | Todos | Contagem por coluna (no CSV está limpo, mas documentar) |
| `isnull().value_counts()` | Titanic (Cabin) | % missing por coluna |
| Injetar NaNs artificialmente | PandasPrimer | Provar que o pipeline com `SimpleImputer` funciona |
| `fillna(média)` por coluna | PandasPrimer | Imputação simples (substituir por `SimpleImputer` no pipeline) |
| `dropna(subset=[...])` | Titanic (Embarked) | Remover linhas com missing em colunas críticas |
| Imputação condicional `loc` + média por grupo | Titanic (Age por Pclass) | Ex.: idade média por `Department` ou `JobRole` |
| `groupby().mean()` para imputar | Titanic | Estratégia segmentada |
| Decisão documentada: drop vs fill | Titanic (Cabin, Name) | `EmployeeNumber` drop; constantes drop |

---

## 3. Cardinalidade, balanceamento e classes

| Técnica | Onde (aula) | Para o HR |
|---------|-------------|-----------|
| `unique()` | PandasPrimer, Titanic | Valores distintos (`Attrition`: Yes/No) |
| `value_counts()` | PandasPrimer, Titanic | Contagem por categoria |
| `value_counts(normalize=True)` | PandasPrimer | Proporção — essencial para Attrition (~16% Yes) |
| Verificar balanceamento | PandasPrimer | Justificar F1 e `class_weight` |
| `nunique()` global | Titanic, Telecom | Colunas com 1 valor → drop (`EmployeeCount`, `StandardHours`) |

---

## 4. Filtragem, subset e comparação lógica

| Técnica | Onde (aula) | Para o HR |
|---------|-------------|-----------|
| Filtro uma condição `df[df.col > x]` | PandasPrimer | Ex.: `OverTime == 'Yes'` |
| Filtro múltiplas condições `&` | PandasPrimer | Ex.: alto `JobLevel` e `Attrition == 'Yes'` |
| Subset por alvo `df[df.Survived==1]` | Titanic | `df[df.Attrition=='Yes']` vs `No` |
| Separar dataframes Churn / noChurn | TelecomChurn | `attrition_yes` / `attrition_no` para comparar distribuições |
| `drop()` de colunas / target | Todos | Separar X e y |
| `reset_index(drop=True)` | Titanic | Após concatenação de features |

---

## 5. Estatísticas por grupo (análise segmentada)

| Técnica | Onde (aula) | Para o HR |
|---------|-------------|-----------|
| `groupby("Pclass").mean()` | Titanic | Média de `MonthlyIncome` por `Department` |
| Média condicional por filtro | Titanic | Idade média mulheres vs homens → por `Gender` e `Attrition` |
| `value_counts()` em subset | Titanic | Sobreviventes por porto → attrition por `BusinessTravel` |
| Contagem cruzada alvo × categoria | Telecom | `countplot` com `hue='churn'` |

---

## 6. Correlação e relações lineares

| Técnica | Onde (aula) | Para o HR |
|---------|-------------|-----------|
| `df.corr()` / `df_numeric.corr()` | PandasPrimer, HousePrice, Titanic, Telecom | Matriz numérica |
| `sns.heatmap(corr, annot=True)` | HousePrice, Titanic, Telecom | Mapa de calor com valores |
| `corr(numeric_only=True)` | HousePrice, Titanic | Ignorar strings |
| Interpretação textual | HousePrice | Ex.: `JobLevel` ↔ `MonthlyIncome` (~0.95) |
| `pairplot` (pesado) | DT iris, HousePrice, Telecom | Relações par a par (usar subset de colunas no HR) |

---

## 7. Distribuições de variáveis numéricas

| Técnica | Onde (aula) | Para o HR |
|---------|-------------|-----------|
| `sns.histplot` com bins definidos | Titanic (Age 0–80) | `Age`, `TotalWorkingYears`, `MonthlyIncome` |
| `sns.displot` + `kde=True` | HousePrice, Telecom | Distribuição suavizada do alvo `MonthlyIncome` |
| `sns.boxplot` | Titanic | Outliers em `DailyRate`, `MonthlyRate`, `DistanceFromHome` |
| Leitura de quartis via `describe()` | Titanic, HousePrice | Comentar caudas longas |
| Análise de outliers explícita | Titanic (Fare) | Valores extremos e impacto no modelo |

---

## 8. Variáveis categóricas (frequências e comparação)

| Técnica | Onde (aula) | Para o HR |
|---------|-------------|-----------|
| `value_counts()` por coluna | Titanic (Embarked) | `Department`, `JobRole`, `MaritalStatus` |
| `sns.countplot` | Telecom | Barras de categorias |
| `countplot(x=..., hue='churn')` | Telecom | Categórica vs Attrition (ex.: `OverTime`, `Department`) |
| Histogramas com `hue` e `multiple='dodge'` | Telecom | Comportamento por Attrition em variáveis numéricas |

---

## 9. Visualização bivariada e multivariada

| Técnica | Onde (aula) | Para o HR |
|---------|-------------|-----------|
| `sns.scatterplot` com `hue` | DT iris | Duas features + cor da classe |
| `sns.pairplot` com `hue` e `markers` | DT iris | Visão global (4–6 features no HR) |
| `plt.scatter` 2D simples | Clustering | Ex.: `Age` vs `TotalWorkingYears` |
| Scatter real vs previsto | HousePrice | `y_test` vs `y_pred` (regressão salário) |
| Distribuição de resíduos `y_test - pred` | HousePrice | Qualidade do modelo de regressão |
| `plot_tree` | DT iris | Interpretabilidade (opcional no relatório) |

---

## 10. Pré-processamento exploratório (antes do Pipeline)

| Técnica | Onde (aula) | Para o HR |
|---------|-------------|-----------|
| `StandardScaler` manual | HousePrice | Nas aulas é manual; no projeto deve ir **dentro** do `Pipeline` |
| `LabelEncoder` | Telecom | Target binário (`Attrition` Yes/No → 0/1) |
| `OneHotEncoder` + `pd.concat` | Titanic, Telecom | Encapsular em `ColumnTransformer` |
| Drop de colunas inúteis | Titanic, Telecom | IDs, constantes, texto sem valor |
| Drop de colunas redundantes | Telecom (`columns2drop`) | Features altamente correlacionadas |
| Verificar linhas com NaN `df[df.isna().any(axis=1)]` | Titanic | Após limpeza |

---

## 11. Validação e otimização (metodologia científica)

| Técnica | Onde (aula) | Para o HR |
|---------|-------------|-----------|
| Split manual train/test | DT iris | Preferir `train_test_split(..., stratify=y)` |
| `train_test_split(test_size=0.3)` | HousePrice, Telecom | Holdout 70/30 |
| `cross_val_score(..., cv=10)` | HousePrice | Comparar modelos de regressão |
| Tabela comparativa `results_df` | HousePrice, Telecom | MAE, MSE, RMSE, R² ou Accuracy, Precision, Recall, F1 |
| `GridSearchCV` | HousePrice, Titanic, Telecom | Hiperparâmetros — **obrigatório** no projeto |
| Múltiplos modelos + confusion matrix | Telecom (vários XGB) | Comparar versões tunadas |

---

## 12. Métricas e diagnóstico pós-modelo

| Técnica | Onde (aula) | Para o HR |
|---------|-------------|-----------|
| `accuracy_score` | DT iris | Menos importante com classes desbalanceadas |
| `confusion_matrix` + `ConfusionMatrixDisplay` | DT iris, Titanic, Telecom | Classificação `Attrition` |
| `sns.heatmap` da matriz de confusão | Titanic, Telecom | Visualização de erros |
| MAE, MSE, RMSE, R² | HousePrice | Regressão `MonthlyIncome` |
| Precision, Recall, F1 | Telecom | **F1** é a métrica do blind test para attrition |

---

## 13. Análise não supervisionada (extra)

| Técnica | Onde (aula) | Para o HR |
|---------|-------------|-----------|
| Scatter sem labels | Clustering | Explorar estrutura natural dos funcionários |
| Método do cotovelo (WCSS vs k) | Clustering | Opcional: quantos “perfis” de employee existem |
| K-Means + scatter por cluster | Clustering | Segmentação para EDA (não substitui pipeline supervisionado) |
| Comparar `cluster == target` | Clustering | Ver se clusters alinham com `Attrition` |

---

## Checklist EDA — `employee_data` (síntese)

Aplicar ao dataset do projeto:

- [ ] **Estrutura:** `head`, `info`, `describe`, `shape`, `nunique`, `dtypes`
- [ ] **Qualidade:** `isnull().sum()`, constantes, `EmployeeNumber` como ID
- [ ] **Alvo classificação:** `Attrition` → `value_counts(normalize=True)`
- [ ] **Alvo regressão:** `MonthlyIncome` → `displot` + `boxplot`
- [ ] **Numéricas:** histogramas/boxplots de `Age`, `TotalWorkingYears`, `YearsAtCompany`, `DistanceFromHome`
- [ ] **Categóricas:** `countplot` com `hue='Attrition'` para `OverTime`, `Department`, `JobRole`, `BusinessTravel`
- [ ] **Por grupo:** `groupby('Department')['MonthlyIncome'].mean()`, idade média por género, attrition rate por `OverTime`
- [ ] **Correlação:** heatmap numérico; comentar `JobLevel`–`MonthlyIncome`
- [ ] **Multivariada:** `pairplot` em 4–5 features chave (não nas 33 de uma vez)
- [ ] **Ordinal:** usar `employee_data.txt` para rotular gráficos de `Education`, `JobSatisfaction`, etc.
- [ ] **Robustez (projeto):** teste com NaNs sintéticos + categorias desconhecidas no pipeline
- [ ] **Modelagem (notebook):** tabela de modelos + `GridSearchCV` + confusion matrix / resíduos

---

## O que as aulas não fazem (mas o projeto exige)

| Requisito do enunciado | Notas |
|------------------------|--------|
| `sklearn.pipeline.Pipeline` + `ColumnTransformer` end-to-end | Nas aulas o pré-processamento é manual |
| `SimpleImputer` / `OneHotEncoder(handle_unknown="ignore")` dentro do pipeline | Obrigatório para o blind test |
| `SelectKBest` / feature selection formal | Critério de avaliação (3 pts data engineering) |
| `StratifiedKFold` explícito | Recomendado para `Attrition` desbalanceado |
| `RandomizedSearchCV` | Nas aulas aparece sobretudo `GridSearchCV` |

As aulas cobrem a **análise exploratória**; o notebook de entrega (`[GroupX]_notebook.ipynb`) deve unir esta EDA com pipelines “caixa preta” serializados em `.pkl`.

---

## Ficheiros relacionados no repo

| Ficheiro | Descrição |
|----------|-----------|
| [notebooks/employee_eda.ipynb](../notebooks/employee_eda.ipynb) | EDA em curso (Secção 1 implementada) |
| [employee_data/employee_data.csv](../employee_data/employee_data.csv) | Dataset principal |
| [employee_data/employee_data.txt](../employee_data/employee_data.txt) | Legenda de variáveis ordinais |
| [Notebooks aulas-20260522/](../Notebooks%20aulas-20260522/) | Notebooks das aulas de referência |
