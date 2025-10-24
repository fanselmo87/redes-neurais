# Modelos Computacionais — Notebook: Aula4-redes-neurais_modcom.ipynb

Visão geral
- Este repositório contém notebooks da disciplina *Modelos Computacionais*. O notebook principal descrito aqui é [c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb).
- Objetivo: demonstrar pré-processamento de dados (house prices), ajuste de modelos (Regressão Linear, Random Forest, XGBoost) e redes neurais (incluindo tuning com Keras Tuner), além de PCA e clusterização sobre o dataset Iris.

Estrutura do notebook ([c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb))
1. Setup e download de dados
   - Instala dependências (ex.: `!pip install kagglehub[pandas-datasets]`).
   - Carregamento via KaggleHub:
     - função principal: [`kagglehub.load_dataset`](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb)
     - variável do dataframe: [`df`](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb)
2. Pré-processamento
   - Remoção de colunas (Id, Alley, MasVnrType, ...).
   - Preenchimento de NA: `df.fillna(df.median(numeric_only=True), inplace=True)` e `df.fillna('None', inplace=True)`.
   - Codificação de categóricas com [`LabelEncoder`](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb).
   - Split: [`X_train`, `X_test`, `y_train`, `y_test`](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb).
3. Modelos clássicos
   - Regressão Linear: objeto [`model = LinearRegression()`](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb).
   - Random Forest: [`rf = RandomForestRegressor(...)`](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb).
   - XGBoost: [`xgb = XGBRegressor(...)`](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb).
   - Em cada modelo: treino, predição (`y_pred`) e métricas.
4. Rede Neural (TensorFlow / Keras)
   - Construção do `model` sequencial com Dense layers (entrada → 37, 19, 10, 5 → saída).
   - Treino: `model.fit(...)` e análise do histórico (`history`).
   - Hyperparameter tuning com Keras Tuner:
     - função de construção: [`build_model`](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb)
     - `tuner = kt.RandomSearch(...)` e chamada `tuner.search(...)`
     - melhor modelo: `best_model = tuner.get_best_models(num_models=1)[0]`
5. PCA & Clustering (Iris)
   - Escalonamento (`StandardScaler`) → PCA (`PCA(n_components=4)`) → KMeans e DBSCAN.
6. Visualizações e artefatos
   - Gráficos salvos: `gráficos.tiff`, `gráficos_xgboost.tiff`, `PCA.png`, `cotovelo.tiff`, `pontuação_de_silhueta.tiff`.
   - Plots: real vs previsto, histogramas de resíduos, importância de variáveis, curvas de loss/val_loss.

Métricas (fórmulas)
- Mean Absolute Error (MAE) — implementado por `mean_absolute_error`.
- Mean Squared Error (MSE) — implementado por `mean_squared_error`.
- Root Mean Squared Error (RMSE):
$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}
$$
- R² — `r2_score`.

Como executar (resumo rápido)
1. Abrir o notebook [c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb) em um kernel Python (recomendado: 3.8+).
2. Instalar dependências:
   - kagglehub, xgboost, tensorflow, keras-tuner (ex.: `!pip install kagglehub xgboost tensorflow keras-tuner`).
3. Executar as células na ordem apresentada. Atenção a tokens/credenciais do Kaggle se exigidos pelo `kagglehub`.

Principais símbolos / pontos de edição (links para o notebook)
- [`df`](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb) — dataframe carregado.
- [`X_train`, `X_test`, `y_train`, `y_test`](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb) — splits de treino/teste.
- [`model` (Linear Regression)](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb)
- [`rf` (RandomForestRegressor)](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb)
- [`xgb` (XGBRegressor)](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb)
- [`build_model`](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb) — função usada pelo Keras Tuner.
- [`tuner`](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb) — instância de RandomSearch do Keras Tuner.
- [`best_model`](c:/Users/ferna/Downloads/Aula4-redes-neurais_modcom.ipynb) — modelo final obtido pelo tuner.

Arquivos relacionados no workspace
- [aula2-regressao.ipynb](aula2-regressao.ipynb) — pipeline de regressão e métricas.
- [aula3-modelos-computacionais.ipynb](aula3-modelos-computacionais.ipynb) — PCA e clusterização (Iris).
- [Aula4-redes_neurais.ipynb](Aula4-redes_neurais.ipynb) — versão alternativa do notebook de redes neurais.

Notas rápidas / gotchas
- Corrija erros de digitação em argumentos de Keras Tuner (ex.: `sampling='log'` ao invés de `smapling='LOG'`).
- Verifique shapes antes de calcular métricas (ex.: `y_pred = y_pred.ravel()` se necessário).
- Configurar seed (`np.random.seed`, `tf.random.set_seed`) para reprodutibilidade.

Contato / autoria
- Notebook criado para as aulas de Modelos Computacionais; edite as células de setup para apontar local correto dos dados e credenciais.
