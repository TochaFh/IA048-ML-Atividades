# IA048 - Atividade 1 - Regressão Linear

Prof. Levy Boccato

**Aluno:** Rafael Campideli Hoyos (175100).

---

## Introdução

Para a realização da atividade foram usadas ferramentas python, destacando o uso das biblioteca `Scikit-Learn` [[2]](#referências) (para treino dos modelos e obtenção de métricas) e `Plotly` [[3]](#referências) (para plotagens de gráficos e tabelas). Os códigos e demais arquivos utilizados/confeccionados neste trabalho foram organizados em uma estrutura simplificada da *Cookiecutter Data Science* [[4]](#referências)  e estão disponíveis no *GitHub* no repositório [**TochaFh/IA048-ML-Atividades**](https://github.com/TochaFh/IA048-ML-Atividades) [[1]](#referências).
**Nota sobre uso de IA:** Foi empregado o auxílio do *Gemini* para esclarecimento de alguns conceitos e confecção de alguns trechos dos scripts e há um relatório detalhando o uso da LLM nesta atividade, encontrado na pasta `Atividade 2/Docs`.

---

## Avaliação de Desempenho

Antes do treinamento dos classificadores, foi implementada uma função de avaliação centralizada para a análise e comparação sistemática dos mesmos, no módulo `evaluation.py`. Esta função foi desenhada para extrair métricas globais adaptadas ao cenário multi-classe, incluindo a acurácia global e a acurácia balanceada, que é fundamental caso exista desbalanceamento entre as classes do problema. A rotina também calcula a precisão, a sensibilidade (*recall*) e o *F-score*, que combina as informações de completude e qualidade do modelo. Por fim, para permitir a observação das classes em que os algoritmos apresentam maior dificuldade de classificação, a função constrói uma matriz de confusão interativa utilizando a biblioteca Plotly.

---

## Pré-processamento dos Dados

Os métodos de tratamento dos dados foram agrupados no módulo `preprocessing.py` e executados via `preprocess_data.ipynb`, gerando os arquivos `.npy` salvos na pasta `data/preprocessed/`. O conjunto de dados foi organizado em duas estruturas bidimensionais: a versão de 561 atributos extraídos do UCI HAR e a versão de sinais brutos. Para os dados brutos, utilizaram-se os 3 eixos de aceleração total (total_acc) e os 3 eixos de velocidade angular (body_gyro), preservando o sinal temporal coletado pelo hardware nas janelas de 128 leituras (6 * 128 = 768 atributos).   A padronização estatística (z-score) foi aplicada para alinhar as escalas das diferentes grandezas físicas (g e rad/s), permitindo a convergência adequada dos modelos kNN e Regressão Logística. Para evitar vazamento de dados (data leakage), a média (mu) e o desvio padrão (sigma) foram extraídos apenas do conjunto de treino e aplicados de forma fixa sobre os dados de teste.

---

## Regressão Logística

Foi escolhida a função **Softmax** para mapear as combinações lineares dos atributos em uma distribuição de probabilidade sobre as classes. Para contornar a alta dimensionalidade dos dados e encontrar o melhor compromisso entre capacidade de generalização e complexidade do modelo, adotou-se uma pipeline de busca sistemática de hiperparâmetros.

### Validação Cruzada K-Fold
O ajuste e a validação intermediária dos hiperparâmetros foram conduzidos através da técnica de **K-Fold Estratificado (K=5)** sobre o conjunto de treinamento. 

* **Limitação e Cuidado Metodológico:** A divisão por K-Fold tradicional garante que a proporção das seis classes de atividades permaneça equilibrada em cada partição de treino e validação. No entanto, o esquema convencional ignora a identidade dos voluntários (Subject ID) do dataset HAR. Como janelas temporais contíguas pertencentes ao mesmo indivíduo compartilham padrões anatômicos e dinâmicas de movimento muito parecidas, a partição aleatória de amostras pode se enquadrar como **vazamento de dados (*data leakage*)** entre os folds de treino e validação. A abordagem metodológica ideal para este cenário seria o uso do `GroupKFold` agrupado por voluntário, assegurando que amostras de um mesmo sujeito nunca estivessem presentes simultaneamente no conjunto de treinamento e de validação, mas, por questões de simplificação do processamento dos dados, optou-se pela versão de estratificação.

### Regularização ElasticNet
Dado que o dataset conta com centenas de atributos extraídos dos sensores (561 features), a escolha da técnica de regularização é fundamental para evitar *overfitting* e tratar possíveis colinearidades. Em vez de restringir o modelo estritamente à penalidade L1 ou L2, optou-se pela formulação **ElasticNet**, que combina linearmente ambos os termos de regularização:

$$\mathcal{L}_{ElasticNet} = \frac{1 - \text{l1\_ratio}}{2} \Vert{}\mathbf{w}\Vert{}_2^2 + \text{l1\_ratio} \Vert{}\mathbf{w}\Vert{}_1$$

Essa escolha fundamenta-se nas seguintes propriedades:
* **Ridge, `l1_ratio` = 0:** Suaviza os pesos atribuídos aos atributos correlacionados, distribuindo a importância entre eles e conferindo estabilidade numérica aos estimadores.
* **Lasso, `l1_ratio` = 1:** Promove a esparsidade do modelo ao zerar os coeficientes de *features* irrelevantes, funcionando como um mecanismo integrado de seleção de atributos.
* **ElasticNet (0 < `l1_ratio` < 1):** Permite avaliar um comportamento misto.


### Avaliação da Grade de Hiperparâmetros e Métrica de Desempenho

Para determinar o modelo de menor complexidade com máximo poder preditivo, realizou-se uma busca em grade avaliando a combinação dos parâmetros de regularização ElasticNet: o inverso da força de regularização ($C$) e a razão da penalidade L1 (`l1_ratio`). 

O hiperparâmetro $C$ controla a intensidade da penalização aplicada aos pesos do modelo de forma inversamente proporcional ($C = 1/\lambda$). Valores pequenos de $C$ aplicam uma regularização forte, restringindo a magnitude dos coeficientes e prevenindo *overfitting*, enquanto valores elevados de $C$ relaxam a penalização, permitindo que o modelo se ajuste mais livremente aos dados de treino.

A qualidade de cada combinação ($C$, `l1_ratio`) foi aferida através do **F1-Score Macro médio** obtido nos 5 *folds* de validação cruzada. Por calcular a média aritmética simples dos F1-Scores de cada uma das seis classes sem ponderação por frequência, a métrica Macro trata todas as atividades com a mesma relevância, impedindo que classes mais fáceis de classificar mascarem eventuais falhas em classes com maior grau de sobreposição (como a diferenciação entre *Sentado* e *Em Pé*).

Uma vez identificada a combinação de hiperparâmetros com maior F1-Score Macro médio na validação cruzada, o modelo deve ser retreinado utilizando a totalidade do conjunto de treinamento e avaliado de forma definitiva no conjunto de teste independente.

O código com a lógica principal da regressão está no arquivo `logistic_regression.py` enquanto as pipelines de treino para cada caso encontram-se em diferentes notebooks (`.ipynb`) na pasta `notebooks`.

---

## Referências

1. TochaFh/IA048-ML-Atividades (repositório com as atividades na disciplina IA048 em 2s2026) https://github.com/TochaFh/IA048-ML-Atividades
2. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. https://scikit-learn.org/stable/
3. Plotly-express. https://plotly.com/python/plotly-express/
4. Cookiecutter Data Science. https://cookiecutter-data-science.drivendata.org/
5. Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). Human Activity Recognition Using Smartphones [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K.