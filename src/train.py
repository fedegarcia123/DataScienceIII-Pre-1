"""
train.py
Pipeline de entrenamiento y validación del MLPClassifier sobre el dataset Iris.

Ejecutar con:
    python src/train.py
"""

import json
import os

import torch
import torch.nn as nn

from data import get_dataloaders
from device import get_device
from model import MLPClassifier

# ---------------------------
# Hiperparámetros
# ---------------------------
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
BATCH_SIZE = 16
HIDDEN_DIM = 16
SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def evaluate(model: nn.Module, loader, criterion, device) -> tuple[float, float]:
    """
    Evalúa el modelo sobre un DataLoader (validación).

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device) -> float:
    """Entrena el modelo durante una época y devuelve la pérdida promedio."""
    model.train()
    total_loss = 0.0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()  # backpropagation via torch.autograd
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        total += X_batch.size(0)

    return total_loss / total


def main():
    torch.manual_seed(SEED)

    device = get_device()
    train_loader, val_loader, input_dim, num_classes = get_dataloaders(
        batch_size=BATCH_SIZE, seed=SEED
    )

    model = MLPClassifier(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    print(f"\nIniciando entrenamiento por {NUM_EPOCHS} épocas | lr={LEARNING_RATE} | device={device}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if epoch == 1 or epoch % 5 == 0 or epoch == NUM_EPOCHS:
            print(
                f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "model_checkpoint.pt"))

    print(f"\nEntrenamiento finalizado. Accuracy final de validación: {history['val_accuracy'][-1]:.4f}")
    print(f"Resultados guardados en: {RESULTS_DIR}/")

    return history


if __name__ == "__main__":
    main()
