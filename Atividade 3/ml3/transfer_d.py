import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
import numpy as np

def get_pretrained_resnet18(num_classes=8, freeze_backbone=False):
    """
    Carrega a ResNet-18 pré-treinada na ImageNet e adapta a camada final.
    """
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    # Substituir a camada linear final (fc) de 1000 classes para 8 classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


def count_parameters(model):
    """
    Retorna o número total e o número de parâmetros treináveis do modelo.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def train_and_validate_transfer(model, train_loader, val_loader, criterion, optimizer, epochs=15, device='cpu'):
    """
    Treina a ResNet-18 e monitora o tempo de execução, a melhor acurácia na validação
    e o histórico por época (loss de treino, loss de validação e acurácia balanceada).
    """
    model.to(device)
    best_val_acc = 0.0
    best_model_weights = None
    start_time = time.time()

    history = {'train_loss': [], 'val_loss': [], 'val_bal_acc': []}

    for epoch in range(epochs):
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

        # Validação
        model.eval()
        val_preds, val_targets = [], []
        val_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).squeeze().long()
                outputs = model(images)
                val_running_loss += criterion(outputs, labels).item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = balanced_accuracy_score(val_targets, val_preds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_bal_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = {k: v.clone() for k, v in model.state_dict().items()}

    total_time = time.time() - start_time

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    return model, best_val_acc, total_time, history



def evaluate_model(model, data_loader, device='cpu'):
    """
    Avaliação final no conjunto de teste.
    """
    model.to(device)
    model.eval()
    
    all_preds, all_targets, all_probs = [], [], []
    
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