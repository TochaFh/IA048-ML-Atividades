import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from ml2.config import RANDOM_SEED


def train_logistic_regression_cv(
        X_train, 
        y_train,
        cv_splits, 
        c_values, 
        l1_ratios, 
        max_iter=300,
        tol=1e-2,
        random_state=RANDOM_SEED
        ):
    """Treina uma regressão logística e seleciona hiperparâmetros por validação cruzada (KFold).

    Avalia as combinações de ``C`` e ``l1_ratio`` (elastic net) usando validação cruzada
    estratificada e a métrica F1 macro.

    Returns:
        tuple: Melhor estimador, melhores hiperparâmetros e resultados completos
        da validação cruzada.
    """

    # O uso de l1_ratio automaticamente ativa o ElasticNet/L1/L2 com o solver SAGA.
    base_model = LogisticRegression(
        solver='saga',
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )

    param_grid = {
        'C': c_values,
        'l1_ratio': l1_ratios
    }

    cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # O paralelismo dos processadores é gerenciado aqui pelo n_jobs=-1
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='f1_macro',
        n_jobs=-1,
        return_train_score=True,
        verbose=1 # Mostra um log de progresso no terminal
    )

    # treino
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


def plot_cv_hyperparameters_heatmap(cv_results, min_v=0, max_v=1):
    df_results = pd.DataFrame(cv_results)
    
    scores_matrix = df_results.pivot(
        index='param_l1_ratio', 
        columns='param_C', 
        values='mean_test_score'
    )

    fig = px.imshow(
        scores_matrix,
        labels=dict(x="Parâmetro C", y="l1_ratio", color="F1-Score Médio"),
        x=[str(c) for c in scores_matrix.columns],
        y=[str(l1) for l1 in scores_matrix.index],
        text_auto=".4f",
        color_continuous_scale="Viridis",
        zmin=min_v,  # Limite inferior da cor em 0
        zmax=max_v,  # Limite superior da cor em 1
        title="Busca de Hiperparâmetros via Validação Cruzada (F1-Macro)"
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_top_features_importance(model, feature_names=None, top_n=15, class_index=0):
    coefs = model.coef_[class_index]
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(coefs))]
        
    df_coefs = pd.DataFrame({
        'feature': feature_names,
        'coef': coefs,
        'abs_coef': np.abs(coefs)
    }).sort_values(by='abs_coef', ascending=False).head(top_n)

    fig = px.bar(
        df_coefs,
        x='coef',
        y='feature',
        orientation='h',
        color='coef',
        color_continuous_scale='RdBu_r',
        title=f"Top {top_n} Atributos mais Relevantes (Classe {class_index + 1})",
        labels={'coef': 'Magnitude do Coeficiente (Peso)', 'feature': 'Atributo'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, template="plotly_white")
    return fig