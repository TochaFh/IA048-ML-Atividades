import numpy as np
import plotly.express as px
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def evaluate_model(model, X_test, y_test, class_names=None, title="Matriz de Confusão"):
    """
    Avalia um modelo de classificação gerando métricas globais e uma Matriz de Confusão interativa.
    
    Parâmetros:
    - model: Modelo treinado do scikit-learn (ex: LogisticRegression ou KNeighborsClassifier).
    - X_test: Atributos do conjunto de teste.
    - y_test: Rótulos verdadeiros do conjunto de teste.
    - class_names: Lista com os nomes das classes para o plot. Se None, usará números.
    - title: Título do gráfico.
    
    Retorna:
    - Um dicionário com as métricas calculadas.
    """
    
    # 1. Realizar as predições
    y_pred = model.predict(X_test)
    
    # 2. Calcular as métricas
    # Acurácia: porcentagem de padrões classificados corretamente ((TP+TN)/N)
    acc = accuracy_score(y_test, y_pred)
    
    # Acurácia Balanceada: média do recall obtido para cada classe, lida bem com desbalanceamento
    ba = balanced_accuracy_score(y_test, y_pred)
    
    # Precisão: proporção de padrões classificados corretamente em relação a todos os atribuídos à classe
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Sensibilidade (Recall): proporção de amostras da classe corretamente classificadas
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    # F-medida (F1-score): métrica única que combina precisão e recall (média harmônica)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # 3. Printar os resultados
    print("-" * 30)
    print("MÉTRICAS DE AVALIAÇÃO")
    print("-" * 30)
    print(f"Acurácia Global        : {acc:.4f}")
    print(f"Acurácia Balanceada    : {ba:.4f}")
    print(f"Precisão (Macro)       : {prec:.4f}")
    print(f"Sensibilidade (Macro)  : {rec:.4f}")
    print(f"F1-Score (Macro)       : {f1:.4f}")
    print("-" * 30)
    
    # 4. Gerar a Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    
    # Se os nomes das classes não forem passados, tentamos inferir a partir do y_test
    if class_names is None:
        class_names = [f"Classe {c}" for c in np.unique(y_test)]
        
    # Plotly Express Heatmap para a Matriz de Confusão
    fig = px.imshow(
        cm,
        text_auto=True, # Adiciona os números dentro das células
        color_continuous_scale='Blues',
        labels=dict(x="Classe Estimada", y="Classe Verdadeira", color="Contagem"),
        x=class_names,
        y=class_names,
        title=title
    )
    
    # Ajustes estéticos no Plotly
    fig.update_layout(
        xaxis_title="Classe Estimada (Predita)",
        yaxis_title="Classe Verdadeira (Real)",
        title_x=0.5, # Centraliza o título
        width=700,
        height=600
    )
    
    # Exibir o gráfico interativo
    fig.show()
    
    # 5. Retornar os valores caso queira usá-los no notebook
    metrics = {
        'accuracy': acc,
        'balanced_accuracy': ba,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }
    
    return metrics