"""
model.py
Arquitectura del modelo: un Perceptrón Multicapa (MLP) simple para clasificación.
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    MLP simple para clasificación multiclase.

    Arquitectura:
        Input -> Linear -> ReLU -> Linear -> ReLU -> Linear (logits) -> num_classes

    Args:
        input_dim (int): cantidad de features de entrada.
        hidden_dim (int): tamaño de las capas ocultas.
        num_classes (int): cantidad de clases a predecir.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve los logits (sin softmax; se usa CrossEntropyLoss)."""
        return self.net(x)


if __name__ == "__main__":
    model = MLPClassifier(input_dim=4, hidden_dim=16, num_classes=3)
    print(model)
    dummy = torch.randn(5, 4)
    out = model(dummy)
    print("Output shape:", out.shape)
