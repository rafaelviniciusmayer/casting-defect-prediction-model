# Prompt para o Cursor — Ajustes solicitados pela banca

## Contexto do projeto (cole isso no início da conversa com o Cursor)

Estou trabalhando em um pipeline de Machine Learning para predição de defeitos em fundição sob pressão (die casting) de peças de alumínio, usando um dataset sintético com:

- 25.000 registros, 15 variáveis de processo originais, expandidas para 115 features via engenharia de atributos (ratios, produtos, diferenças, distâncias de faixas ideais, features binárias de faixa, agregações estatísticas, transformações matemáticas).
- 28 tipos de defeito, problema de classificação multi-label binária (um por defeito).
- Split estratificado 80/20 (treino/desenvolvimento vs. teste final), com validação cruzada estratificada em 5 folds usada apenas na rede neural.
- Balanceamento de classes via SMOTE (aplicado só no treino, razão-alvo 1.5:1 negativo:positivo por defeito) + aprendizado sensível ao custo (loss ponderada na rede neural, `class_weight='balanced'` na Random Forest).
- Três arquiteturas comparadas: Rede Neural (PyTorch, MLP 115→128→64→32→28), XGBoost (`MultiOutputClassifier`, one-vs-rest) e Random Forest (`MultiOutputClassifier`).
- Otimização de threshold por defeito via grid search (0.10 a 0.91, passo 0.01), maximizando recall com F1 como critério de desempate.
- Resultado atual: Rede Neural selecionada para deploy (recall 97,63%, inferência 0,33 ms/100 amostras); Random Forest teve o maior recall (98,38%) mas inferência de 1.232 ms; XGBoost teve o melhor F1-micro/macro mas recall mais baixo (92,78%).

Preciso implementar os ajustes abaixo, apontados pela banca na apresentação do trabalho. As tarefas estão organizadas em **fases sequenciais**, na ordem em que devem ser executadas — cada fase depende do resultado da anterior, então siga a ordem para não ter que retrabalhar análises já feitas com pipeline desatualizado. Mantenha compatibilidade com o pipeline existente (mesmos dados, mesmo split, mesmas features) para que os resultados sejam comparáveis entre si e com os já reportados no artigo.

---

## Ordem de execução

`Fase 1 → Fase 2 → Fase 3`, tarefas dentro de cada fase na ordem numerada.

---

## FASE 1 — Infraestrutura de validação (base para todo o resto)

> Objetivo desta fase: parar de treinar XGBoost/Random Forest "direto" e passar todos os modelos para o mesmo esquema de cross-validation, e já aproveitar essa estrutura para investigar overfitting e a real contribuição do SMOTE. Só depois disso vale a pena adicionar novos modelos ou gerar gráficos de threshold/ROC — senão eles têm que ser refeitos.

### 1.1 Padronizar treino/validação nos 3 modelos escolhidos *(era item 5)*
- Hoje só a Rede Neural usa validação cruzada estratificada em 5 folds; XGBoost e Random Forest treinam direto no conjunto de desenvolvimento completo (80%). Isso já é citado como limitação no artigo (Seção 5.3).
- Ajustar para que **todos os modelos comparados usem exatamente o mesmo esquema de validação cruzada estratificada em 5 folds** sobre o conjunto de desenvolvimento (80%), com SMOTE aplicado dentro de cada fold de treino (nunca no fold de validação, para evitar vazamento de dados).
- Reportar média e desvio-padrão das métricas (recall, precisão, F1) entre os folds para os 3 modelos, permitindo comparação estatisticamente mais justa.
- O conjunto de teste final (20%) permanece isolado e usado apenas para avaliação final, como já é feito.
- **Entrega desta etapa**: pipeline unificado de CV reutilizável (ex. `unified_cv_pipeline.py`) que os itens seguintes vão chamar, em vez de reimplementar treino/validação.

### 1.2 Análise formal de overfitting via 80/20 + cross-validation *(era item 9)*
- Reaproveitando o pipeline de CV do item 1.1, para cada modelo:
  1. Calcular métricas de **treino** e de **validação** em cada um dos 5 folds.
  2. Comparar a média de métricas de treino (dentro dos folds) com a média de métricas de validação — um gap grande aqui indica overfitting durante o treinamento.
  3. Avaliar o modelo final (treinado nos 80% completos) nos **20% de teste isolado**, comparando com o desempenho médio de validação da CV. Um gap grande entre validação (CV) e teste final também é sinal de overfitting ou de distribuição diferente entre teste e treino/validação.
- Consolidar tudo em uma tabela única: Recall/Precisão/F1 em Treino (CV), Validação (CV) e Teste Final, por modelo.

### 1.3 Resultados com e sem SMOTE *(era item 7)*
- Usando o mesmo pipeline de CV (item 1.1), rodar cada modelo em duas condições:
  - (a) **sem SMOTE**, usando apenas o aprendizado sensível ao custo (class weights / loss ponderada);
  - (b) **com SMOTE**, como já é feito atualmente (SMOTE + cost-sensitive learning combinados).
- Comparar recall, precisão, F1-micro, F1-macro e tempo de treino, para os 3 modelos, nas duas condições.
- Isso responde diretamente à pergunta da banca sobre a real contribuição do SMOTE (o artigo já afirma que a combinação SMOTE + cost-sensitive é melhor que qualquer técnica isolada — Seção 5.1 — mas isso ainda não está demonstrado empiricamente lado a lado; essa tarefa gera essa evidência).
- **Decisão a tomar aqui**: se o SMOTE não trouxer ganho relevante (ou piorar overfitting), considerar manter a configuração final apenas com cost-sensitive learning — isso muda a versão "oficial" do pipeline usada nas fases seguintes.

### 1.4 Verificar overfitting causado pelo SMOTE *(era item 8)*
- Complementa o item 1.3: comparar métricas de desempenho no **conjunto de treino (após SMOTE)** vs. **conjunto de teste (nunca visto, sem SMOTE)**, para os 3 modelos.
- Um gap grande entre treino e teste é sinal de overfitting, possivelmente inflado pelas amostras sintéticas do SMOTE (que podem ser muito parecidas entre si em espaço de 115 dimensões).
- Complementar com curvas de aprendizado (learning curves): desempenho (recall/F1) em função do tamanho do conjunto de treino, para verificar se o modelo está memorizando padrões sintéticos.
- Reportar lado a lado com a tabela do item 1.2, já que ambos abordam overfitting a partir de ângulos complementares.

---

## FASE 2 — Expandir os modelos comparados

> Só começa depois da Fase 1 fechada, para não treinar o mesmo modelo duas vezes com pipelines diferentes.

### 2.1 Comparar com modelos estatísticos e de regressão *(era item 1)*
Adicionar ao comparativo de modelos (além de Rede Neural, XGBoost e Random Forest) pelo menos:
- **Regressão Logística** (com `class_weight='balanced'`), como baseline linear interpretável, treinada com `MultiOutputClassifier` da mesma forma que XGBoost/RF.
- Um modelo estatístico clássico adicional, se possível: **Regressão Logística Regularizada (L1/L2 - Lasso/Ridge)** para servir de baseline "estatístico" formal, ou um teste de **regressão linear/logística simples por variável** como benchmark de referência (não apenas ML "caixa-preta").
- Usar exatamente o **mesmo pipeline de CV, SMOTE (ou não, conforme decidido em 1.3) e otimização de threshold** definido na Fase 1, para manter a comparação justa.
- Adicionar os resultados à tabela comparativa (equivalente à Tabela 8 do artigo): recall, precisão, F1-micro, F1-macro, tempo de treino, tempo de inferência.

---

## FASE 3 — Diagnósticos e interpretação (usam os modelos finais)

> Esta fase consome os modelos já treinados e validados nas fases anteriores. A ordem aqui evita recalcular probabilidades/thresholds mais de uma vez.

### 3.1 Análise de importância de features (ex.: etapa de injeção) *(era item 6)*
- Implementar análise de importância de features para os 4 modelos finais (NN, XGBoost, RF, Regressão Logística):
  - **Random Forest / XGBoost**: usar `feature_importances_` nativo e complementar com **SHAP values**.
  - **Rede Neural**: usar **SHAP (DeepExplainer ou KernelExplainer)** ou **permutation importance**.
  - **Regressão Logística**: usar os coeficientes padronizados como medida direta de importância/direção do efeito.
- Agrupar a importância das features por **fase do processo** (injeção, intensificação, resfriamento, configuração/manutenção) e por **categoria de feature engineering** (ratio, produto, diferença, distância, binária de faixa, agregação estatística, específica de domínio, transformação matemática), para responder diretamente à pergunta: "as variáveis da etapa de injeção são as que mais impactam?"
- Gerar um gráfico de barras (top 20-30 features mais importantes) e uma tabela agregada por fase/categoria.
- Não depende de threshold, por isso pode vir antes dos itens 3.2/3.3.

### 3.2 Balancear Precision x Recall (reduzir excesso de foco em recall) *(era item 2)*
- Revisar a lógica de otimização de threshold: hoje ela **maximiza recall com F1 como desempate**. Ajustar para permitir comparação entre estratégias:
  - (a) otimização atual (recall puro),
  - (b) otimização por **F1-score** (equilíbrio),
  - (c) otimização por **F-beta com beta ajustável** (ex.: beta=2 prioriza recall moderadamente, beta=1 equilibra, beta=0.5 prioriza precisão).
- Gerar uma tabela/gráfico mostrando como precisão, recall e F1 mudam para cada defeito conforme a estratégia de threshold muda. Isso dá material concreto para discutir o trade-off com a banca (Leonardo mencionou ter material sobre isso — vou complementar com uma seção no relatório justificando a escolha final).
- **Definir aqui a estratégia de threshold "oficial"** que será usada nos itens 3.3 e 3.4 a seguir.

### 3.3 Adicionar curva ROC e discutir AUC (Leonardo questionou o threshold) *(era item 3)*
- Reaproveitar as probabilidades já calculadas no item 3.2 (evita recomputar previsões).
- Implementar o cálculo e plot da **curva ROC** por defeito (pelo menos para os defeitos mais frequentes: gas porosity, cold shut, incomplete fill) e uma versão agregada (micro-average ROC).
- Calcular o **AUC-ROC** por defeito e reportar em tabela.
- Marcar no gráfico da curva ROC o ponto correspondente ao threshold ótimo escolhido em 3.2, deixando visualmente clara a relação entre o threshold e o trade-off TPR/FPR.

### 3.4 Gráfico de previsto x realizado (metodologia de validação) *(era item 4)*
- Implementar um gráfico de **previsto x realizado** para cada modelo, usando o threshold final definido em 3.2, útil tanto para visualizar calibração de probabilidades quanto para inspecionar erros sistemáticos.
- Como o problema é classificação binária multi-label (não regressão contínua), adaptar como:
  - **Gráfico de calibração** (reliability diagram): probabilidade prevista (bins) vs. frequência real observada de defeito, por defeito ou agregado.
  - **Matriz de confusão normalizada** por defeito (complementar à Tabela 12 já existente), como forma visual de "previsto x realizado".
- Se a intenção da banca for algo mais específico (ex. um método com nome que eu não identifiquei corretamente na fala — "SCOD BRIK"), vou confirmar com o professor Leonardo o nome exato do método antes de implementar algo diferente do que descrevi acima.
- É o diagnóstico de "acabamento": serve como validação visual final de tudo que foi construído nas fases anteriores.

---

## Instruções gerais para o Cursor

- **Siga a ordem das fases (1 → 2 → 3).** Não pule para a Fase 2 ou 3 antes de fechar a Fase 1 — os itens seguintes reaproveitam o pipeline de CV e as decisões (SMOTE sim/não, threshold oficial) tomadas nas etapas anteriores.
- Implemente cada tarefa como uma função ou módulo separado, seguindo esta sequência de arquivos:
  1. `unified_cv_pipeline.py` (item 1.1 — base de tudo)
  2. `overfitting_analysis.py` (item 1.2)
  3. `smote_ablation.py` (item 1.3)
  4. `smote_overfitting_check.py` (item 1.4)
  5. `compare_statistical_models.py` (item 2.1)
  6. `feature_importance_analysis.py` (item 3.1)
  7. `threshold_tradeoff_analysis.py` (item 3.2)
  8. `roc_curve_analysis.py` (item 3.3)
  9. `predicted_vs_actual.py` (item 3.4)
- Cada módulo novo deve **importar e reutilizar** o pipeline de pré-processamento, CV e (quando aplicável) threshold definidos nos módulos anteriores — evite reimplementar treino/avaliação do zero em cada arquivo.
- Sempre que possível, gere as saídas em dois formatos: tabelas (CSV ou DataFrame, prontas para virar tabelas do artigo) e gráficos (PNG/SVG, prontos para virar figuras do artigo).
- Documente claramente, em comentários ou docstrings, qual pergunta da banca cada script/análise está respondendo, para facilitar a redação da seção de resultados depois.
- Ao final de cada fase, gere um checkpoint (tabela resumo ou markdown curto) com as decisões tomadas naquela fase (ex.: "SMOTE mantido/removido", "threshold oficial = F1" etc.), para que as fases seguintes partam de uma referência clara.
- Ao final de tudo, gere um notebook ou script de sumário que rode as três fases em sequência e produza um relatório consolidado (markdown ou HTML) com todas as tabelas e gráficos, organizados na mesma ordem das fases.

Aqui está meu código atual: [cole aqui os arquivos/pastas relevantes do seu projeto, ou aponte o Cursor para o repositório]
