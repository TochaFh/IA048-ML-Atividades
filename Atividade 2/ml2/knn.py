import pandas as pd
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from ml2.config import RANDOM_SEED

def train_knn_cv(
        X_train, y_train, 
        cv_splits, 
        n_neighbors_list, 
        weights_list, 
        p_list,
        algorithm='brute',
        random_state=RANDOM_SEED
        ):
    """
    Treina e otimiza um modelo k-NN usando busca em grade (GridSearchCV) e validação cruzada.
    
    Parâmetros:
    - X_train, y_train: Matrizes NumPy com os dados de treinamento e rótulos padronizados.
    - cv_splits (int): Número de folds para a validação cruzada estratificada (padrão 5).
    - n_neighbors_list (list): Grade de valores para 'k' (número de vizinhos).
    - weights_list (list): Estratégias de peso ('uniform' ou 'distance').
    - p_list (list): Expoentes de distância Minkowski (1 = Manhattan, 2 = Euclidiana).
    - algorithm (str): Algoritmo de busca. 'brute' costuma ser mais rápido para dados de alta 
                       dimensionalidade (como as 561 features do HAR).
    Retorna:
    - best_model: O modelo KNeighborsClassifier re-treinado com a melhor combinação.
    - best_params: Dicionário contendo os melhores hiperparâmetros encontrados.
    - cv_results: Dicionário com os resultados completos (útil para plotar o heatmap).
    """
    
    # 1. Instanciar o classificador base
    knn = KNeighborsClassifier(algorithm=algorithm)
    
    # 2. Definir a grade de hiperparâmetros
    param_grid = {
        'n_neighbors': n_neighbors_list,
        'weights': weights_list,
        'p': p_list
    }

    cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    # 3. Configurar a busca em grade com validação cruzada
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='f1_macro',   # Métrica oficial definida na metodologia
        n_jobs=-1,
        return_train_score=False, # Economiza memória e tempo (não precisamos do score de treino aqui)
        verbose=1 # Mostra um log de progresso no terminal
    )
    
    # 4. Executar o treinamento e validação
    grid_search.fit(X_train, y_train)
    
    # Retorna o melhor modelo, os parâmetros ótimos e todo o log da busca
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

def plot_knn_grid_heatmap(cv_results, min_v=0, max_v=1):
    """
    Gera um Heatmap Plotly comparando k x (weights + distância p).
    - Eixo X: k (n_neighbors)
    - Eixo Y: 'uniform' / 'distance' + 'Manhattan (L1)' / 'Euclidiana (L2)'
    """
    df = pd.DataFrame(cv_results)
    
    # Mapeamento para rótulos legíveis
    p_map = {1: 'Manhattan', 2: 'Euclidiana'}
    df['p_label'] = df['param_p'].map(p_map)
    df['config_label'] = df['param_weights'].astype(str) + " | " + df['p_label'].astype(str)
    
    # Tabela dinâmica para o Heatmap
    pivot_df = df.pivot(
        index='config_label', 
        columns='param_n_neighbors', 
        values='mean_test_score'
    )
    
    fig = px.imshow(
        pivot_df.values,
        x=[str(k) for k in pivot_df.columns],
        y=list(pivot_df.index),
        color_continuous_scale="Viridis",
        zmin=min_v,  # Limite inferior da cor em 0
        zmax=max_v,  # Limite superior da cor em 1
        text_auto=".4f",
        labels=dict(
            x="Número de Vizinhos (k)", 
            y="Configuração (Pesos | Distância)", 
            color="F1-Score Macro"
        ),
        aspect="auto"
    )
    
    fig.update_layout(
        title="Busca de Hiperparâmetros k-NN (F1-Macro)",
        xaxis_title="Número de Vizinhos (k)",
        yaxis_title="Configuração (Pesos | Distância)",
        font=dict(size=12)
    )
    
    return fig


def plot_knn_grid_lines(cv_results, range_y=[0.8, 1.02]):
    """
    Gera um gráfico de linhas Plotly mostrando o comportamento do F1-Score conforme k varia.
    Ideal para ver visualmente o ponto ideal de k antes da queda de desempenho.
    """
    df = pd.DataFrame(cv_results)
    
    p_map = {1: 'Manhattan', 2: 'Euclidiana'}
    df['p_label'] = df['param_p'].map(p_map)
    df['Estratégia'] = df['param_weights'].astype(str) + " | " + df['p_label'].astype(str)
    
    fig = px.line(
        df,
        x='param_n_neighbors',
        y='mean_test_score',
        color='Estratégia',
        markers=True,
        labels={
            'param_n_neighbors': 'Número de Vizinhos (k)',
            'mean_test_score': 'F1-Score Macro Médio',
            'Estratégia': 'Configuração'
        }
    )
    
    fig.update_layout(
        title="Evolução do F1-Score em Função de k",
        yaxis=dict(range=range_y),
        xaxis=dict(type='category'),
        hovermode="x unified"
    )
    
    return fig