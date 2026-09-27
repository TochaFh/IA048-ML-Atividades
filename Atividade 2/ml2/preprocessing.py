import numpy as np
import pandas as pd

def load_csv_X(file_path):
    return pd.read_csv(file_path, sep=r'\s+', header=None).values

def load_csv_y(file_path):
    """
    Carrega o csv com os rótulos (aplica squeeze)
    """
    return pd.read_csv(file_path, sep=r'\s+', header=None).values.squeeze()

def list_inertial_files(group):
    return [
        f"total_acc_x_{group}.txt", f"total_acc_y_{group}.txt", f"total_acc_z_{group}.txt",
        f"body_gyro_x_{group}.txt", f"body_gyro_y_{group}.txt", f"body_gyro_z_{group}.txt"
    ]


def load_files_stacking(parent_dir, files_list):
    """
    Combina os sinais de diferentes arquivos concatenando-os horizontalmente.
    Como são 6 arquivos de formato (N, 128), o np.hstack retorna direto (N, 768).
    """
    dados_inerciais = []
    for file in files_list:
        dado = load_csv_X(parent_dir / file)
        dados_inerciais.append(dado)
        
    return np.hstack(dados_inerciais)

def apply_standardization(X, mu=None, sigma=None):
    """
    Aplica data standardization (média nula e desvio padrão unitário).
    Se mu e sigma não forem passados, calcula a partir do próprio X (usado no treino).
    """
    if mu is None or sigma is None:
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
    
    X_norm = (X - mu) / (sigma + 1e-8)
    return X_norm, mu, sigma