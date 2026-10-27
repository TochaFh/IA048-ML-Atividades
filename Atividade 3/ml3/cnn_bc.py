import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

class BloodMNISTCNN(nn.Module):
    """
    Rede Neural Convolucional (CNN) Simples para classificação do BloodMNIST (8 classes).
    Estrutura: 
      1. Camada Convolucional 2D + ReLU (com Same Padding P = (k-1)//2)
      2. Camada de Max-Pooling 2x2 (Stride = 2)
      3. Achatamento (Flatten)
      4. Camada Totalmente Conectada (Linear) de Saída
    """
    def __init__(self, in_channels=3, num_kernels=32, kernel_size=3, num_classes=8):
        """
        Parâmetros:
        -----------
        in_channels : int
            Número de canais da imagem de entrada (3 para RGB).
        num_kernels : int (K)
            Quantidade de filtros/kernels na camada convolucional (ex: 8, 16, 32, 64).
        kernel_size : int (k)
            Tamanho do kernel de convolução (ex: 3, 5, 7).
        num_classes : int
            Número de classes de saída (8 no BloodMNIST).
        """
        super(BloodMNISTCNN, self).__init__()
        
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        
        # Calculo do Same Padding para kernels ímpares (k=3 -> P=1; k=5 -> P=2; k=7 -> P=3)
        # Mantém a resolução espacial em 28x28 após a convolução.
        padding = (kernel_size - 1) // 2
        
        # 1. Camada Convolucional + Ativação ReLU
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_kernels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        self.relu = nn.ReLU()
        
        # 2. Camada de Pooling (Max-Pooling 2x2 com Stride 2)
        # Reduz a dimensão de 28x28 para 14x14
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3. Achatamento (Flatten)
        self.flatten = nn.Flatten()
        
        # Dimensão fixa resultante do pooling: 14 x 14
        spatial_dim_after_pool = 14
        flatten_dim = num_kernels * spatial_dim_after_pool * spatial_dim_after_pool
        
        # 4. Camada Densa de Saída (Logits Brutos)
        self.fc = nn.Linear(flatten_dim, num_classes)

    def forward(self, x):
        """
        Fluxo de propagação direta (forward pass)
        """
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits


def train_and_validate(model: nn.Module, train_loader, val_loader, criterion, optimizer, epochs=20, device='cpu'):
    """
    Executa o loop de treinamento e validação para uma CNN no PyTorch.
    """
    model.to(device)
    best_val_acc = 0.0
    best_model_weights = None

    for epoch in range(epochs):
        # ==========================================
        # 1. FASE DE TREINAMENTO
        # ==========================================
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).squeeze().long()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        train_loss = running_loss / len(train_loader.dataset)

        # ==========================================
        # 2. FASE DE VALIDAÇÃO
        # ==========================================
        model.eval()
        val_preds = []
        val_targets = []
        val_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).squeeze().long()
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = balanced_accuracy_score(val_targets, val_preds)

        # Salva os melhores pesos obtidos na validação
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = {k: v.clone() for k, v in model.state_dict().items()}

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    return model, best_val_acc


def wrap_train_and_validate(num_kernels, kernel_size, lr, train_loader, val_loader, device, epochs=20):
    """
    Instancia a CNN com a configuração especificada e executa o treino/validação.
    """
    model = BloodMNISTCNN(num_kernels=num_kernels, kernel_size=kernel_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    trained_model, val_acc = train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device
    )

    return trained_model, val_acc


def fit_cnn(train_loader, val_loader, device, 
            num_kernels_list=[8, 16, 32, 64], 
            kernel_sizes=[3, 5, 7], 
            lr=1e-3, 
            epochs=20, 
            save_tag=""):
    """
    Busca em Grade (Grid Search) para explorar:
      - Quantidade de Kernels (num_kernels)
      - Tamanhos do Kernel (kernel_size)
    """
    results = []
    best_overall_acc = 0.0
    best_config = None

    for K in num_kernels_list:
        for k in kernel_sizes:
            print(f"\n--- Testando CNN: num_kernels={K}, kernel_size={k} ---")
            
            trained_model, val_acc = wrap_train_and_validate(
                num_kernels=K, 
                kernel_size=k, 
                lr=lr, 
                train_loader=train_loader, 
                val_loader=val_loader, 
                device=device,
                epochs=epochs
            )
            
            results.append({
                'num_kernels': K,
                'kernel_size': k,
                'val_acc': val_acc
            })
            
            if val_acc > best_overall_acc:
                best_overall_acc = val_acc
                best_config = {'num_kernels': K, 'kernel_size': k}
                torch.save(trained_model.state_dict(), f'best_model{save_tag}.pth')

    print("\n==========================================")
    print(f"Melhor Configuração de CNN Encontrada: {best_config}")
    print(f"Acurácia Balanceada na Validação: {best_overall_acc:.4f}")
    print("==========================================")

    return results


def evaluate_model(model: nn.Module, data_loader, device='cpu') -> dict:
    """
    Calcula as predições e métricas finais (Acurácia Balanceada, F1-macro e Matriz de Confusão).
    """
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)
            
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    cm = confusion_matrix(all_targets, all_preds)
    
    return {
        'targets': all_targets,
        'preds': all_preds,
        'probs': all_probs,
        'bal_acc': bal_acc,
        'f1_macro': f1_macro,
        'cm': cm
    }


def get_misclassified_samples(model: nn.Module, data_loader, device='cpu', max_samples=5):
    """
    Helper para o Item (c): Retorna N amostras classificadas incorretamente no conjunto de teste,
    junto com a imagem, classe real, classe predita e o vetor de probabilidades do Softmax.
    """
    model.to(device)
    model.eval()
    
    misclassified = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images_dev = images.to(device)
            labels_dev = labels.view(-1).long().to(device)
            
            logits = model(images_dev)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            for i in range(len(labels_dev)):
                target = labels_dev[i].item()
                pred = preds[i].item()
                
                if pred != target:
                    misclassified.append({
                        'image': images[i].cpu(),          # Tensor (3, 28, 28)
                        'true_label': target,
                        'pred_label': pred,
                        'probs': probs[i].cpu().numpy()     # Vetor de 8 probabilidades
                    })
                    
                    if len(misclassified) >= max_samples:
                        return misclassified
                        
    return misclassified