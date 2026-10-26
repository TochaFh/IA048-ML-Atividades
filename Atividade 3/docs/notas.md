```mermaid
flowchart LR
    A["Entrada:\n> imagem (28 x 28 x 3)"] --> B["Achatamento (flatten)\n> [2352]"]
    B --> C["Camada Oculta: Linear + ReLU + Dropout(p)\n> [N_oculta]"]
    C --> D["Camada de Saída: Linear\n> 8 Logits"]
    
    D -- "Treino" --> E["Entropia Cruzada (Softmax Interno + Loss)"]
    D -- "Teste / Inferência" --> F["Softmax Explícito\n> Argmax"]
    
    F --> G["> Classe Predita (0 a 7)\n> Métricas (Acurácia, F1, Matriz)"]
```

### Decisões Fixadas de Projeto (MLP)

| Configuração | Escolha / Valor | Justificativa |
| :--- | :--- | :--- |
| **Dimensão de Entrada** | $2352$ neurônios | Achatamento (*Flatten*) da imagem colorida de $28 \times 28 \times 3$ pixels. |
| **Pré-processamento** | Escalonamento em $[0.0, 1.0]$ | Divisão dos pixels por $255$ para manter os valores em escala adequada. |
| **Nº de Camadas Ocultas** | Exatamente $1$ camada | Exigência da atividade. |
| **Ativação Intermediária** | ReLU ($\max(0, x)$) | Evita o desvanecimento de gradientes e possui baixo custo computacional. |
| **Camada de Saída** | $8$ neurônios (*Logits* brutos) | Corresponde às 8 classes de células sanguíneas, sem ativação no *forward*. |
| **Função de Custo** | Entropia Cruzada Multiclasse | Aplica o Softmax internamente com estabilidade numérica. |
| **Otimizador** | Adam | Algoritmo com taxas de aprendizado adaptativas e rápida convergência. |
| **Regularização $L_2$** | weight_decay = $10^{-4}$ | Valor fixado para restringir a magnitude dos pesos e evitar *overfitting*. |


### Hiperparâmetros Explorados na Validação (MLP)

| Hiperparâmetro | Valores Candidatos | Propósito da Exploração |
| :--- | :--- | :--- |
| **Neurônios Ocultos ($N_{\text{oculta}}$)** | $\{128, 256, 512\}$ | Avaliar o impacto da capacidade de representação e complexidade da rede. |
| **Taxa de Aprendizado ($\eta$)** | $\{10^{-3}, 10^{-4}\}$ | Controlar o tamanho do passo de atualização dos pesos pelo otimizador. |
| **Probabilidade de Dropout ($p$)** | $\{0.0, 0.2, 0.3, 0.5\}$ | Investigar o efeito da inativação aleatória de neurônios na generalização. |