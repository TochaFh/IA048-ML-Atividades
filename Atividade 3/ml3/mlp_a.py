import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

class BloodMNISTMLP(nn.Module):
    """
    Perceptron Multicamadas (MLP) com exatamente uma camada oculta
    desenvolvida para a classificação do BloodMNIST (8 classes).
    """
    def __init__(self, input_dim=2352, hidden_dim=256, num_classes=8, dropout_prob=0.2):
        """
        Parâmetros:
        -----------
        input_dim : int
            Tamanho do vetor de entrada (28x28x3 = 2352).
        hidden_dim : int
            Número de neurônios na camada oculta (ex: 128, 256, 512).
        num_classes : int
            Número de classes de saída (8 classes de células sanguíneas).
        dropout_prob : float
            Probabilidade de desligamento dos neurônios na camada oculta (ex: 0.0, 0.2, 0.5).
        """
        super(BloodMNISTMLP, self).__init__()
        
        # 1. Camada de Achatamento (Transforms [B, 3, 28, 28] -> [B, 2352])
        self.flatten = nn.Flatten()
        
        # 2. Camada Oculta
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # 3. Camada de Saída (8 logits brutos)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Fluxo de propagação direta (forward pass)
        """
        # Achatamento da imagem de entrada
        x = self.flatten(x)
        
        # Passagem pela camada oculta
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Obtenção dos logits brutos na saída (sem Softmax aqui!)
        logits = self.fc2(x)
        
        return logits


def train_and_validate(model: BloodMNISTMLP, train_loader, val_loader, criterion, optimizer, epochs=20, device='cpu'):
    """
    Função de suporte para executar o loop de treinamento e validação no PyTorch.
    """
    model.to(device)
    best_val_acc = 0.0
    best_model_weights = None

    for epoch in range(epochs):
        # ==========================================
        # 1. FASE DE TREINAMENTO
        # ==========================================
        model.train()  # Ativa modo de treino (ativa Dropout)
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).squeeze().long()
            
            # Os 5 passos fundamentais do PyTorch:
            optimizer.zero_grad()               # 1. Zerar gradientes
            outputs = model(images)             # 2. Forward pass
            loss = criterion(outputs, labels)   # 3. Calcular a perda
            loss.backward()                     # 4. Backward pass (gradientes)
            optimizer.step()                    # 5. Atualizar pesos
            
            running_loss += loss.item() * images.size(0)
            
        train_loss = running_loss / len(train_loader.dataset)

        # ==========================================
        # 2. FASE DE VALIDAÇÃO
        # ==========================================
        model.eval()  # Ativa modo de avaliação (desativa Dropout)
        val_preds = []
        val_targets = []
        val_running_loss = 0.0
        
        with torch.no_grad():  # Desativa o cálculo de gradientes para economizar memória e acelerar
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).squeeze().long()
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                
                # Obter a classe predita (argmax sobre os logits)
                preds = torch.argmax(outputs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = balanced_accuracy_score(val_targets, val_preds)

        # Salvar o melhor modelo com base na Acurácia Balanceada de Validação
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy()

        #print(f"Época [{epoch+1}/{epochs}] | "
        #      f"Treino Loss: {train_loss:.4f} | "
        #      f"Val Loss: {val_loss:.4f} | "
        #      f"Val Bal Acc: {val_acc:.4f}")

    # Carrega os pesos do melhor modelo encontrado durante a validação
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    return model, best_val_acc

def wrap_train_and_validate(h_dim, dropout_p, lr, train_loader, val_loader, device):
    model = BloodMNISTMLP(hidden_dim=h_dim, dropout_prob=dropout_p)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Treinar e validar esta combinação
    trained_model, val_acc = train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=20,
        device=device
    )

    return trained_model, val_acc

def fit_mlp(train_loader, val_loader, device, hidden_dims = [128, 256, 512], lrs = [1e-3, 1e-4], dropouts = [0.0, 0.2, 0.5], save_tag = ""):

    results = [] # Para armazenar o desempenho de cada teste
    best_overall_acc = 0.0
    best_config = None

    # 2. Varredura (Grid Search)
    for h_dim in hidden_dims:
        for lr in lrs:
            for p in dropouts:
                print(f"\n--- Testando: hidden_dim={h_dim}, lr={lr}, dropout={p} ---")
                
                trained_model, val_acc = wrap_train_and_validate(h_dim, p, lr, train_loader, val_loader, device)
                
                # Guardar o resultado
                results.append({
                    'hidden_dim': h_dim,
                    'lr': lr,
                    'dropout': p,
                    'val_acc': val_acc
                })
                
                # Verificar se é a melhor combinação até agora
                if val_acc > best_overall_acc:
                    best_overall_acc = val_acc
                    best_config = {'hidden_dim': h_dim, 'lr': lr, 'dropout': p}
                    # Salvar os pesos do modelo campeão
                    torch.save(trained_model.state_dict(), f'best_model{save_tag}.pth')

    print("\n==========================================")
    print(f"Melhor Configuração Encontrada: {best_config}")
    print(f"Acurácia Balanceada na Validação: {best_overall_acc:.4f}")
    print("==========================================")

    return results
