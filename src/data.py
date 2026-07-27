"""
data.py
Carga y preparación del dataset Iris (clásico, multiclase) como tensores de PyTorch,
con split train/validation y estandarización de features.

El dataset original (en formato CSV, generado por sklearn) se guarda en data/iris.csv
para dejar registro del insumo crudo del pipeline.
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _ensure_raw_csv() -> str:
    """Genera data/iris.csv a partir de sklearn si todavía no existe."""
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "iris.csv")
    if not os.path.exists(csv_path):
        iris = load_iris(as_frame=True)
        df = iris.frame.rename(columns={"target": "target"})
        df.to_csv(csv_path, index=False)
    return csv_path


def get_dataloaders(batch_size: int = 16, val_size: float = 0.2, seed: int = 42):
    """
    Carga el dataset Iris, lo divide en train/validation y arma los DataLoaders.

    Args:
        batch_size (int): tamaño de batch para entrenamiento.
        val_size (float): proporción del set de validación.
        seed (int): semilla para reproducibilidad del split.

    Returns:
        train_loader, val_loader, input_dim, num_classes
    """
    csv_path = _ensure_raw_csv()
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["target"]).values.astype(np.float32)
    y = df["target"].values.astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )

    # Estandarización (fit solo en train, para evitar data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    return train_loader, val_loader, input_dim, num_classes


if __name__ == "__main__":
    train_loader, val_loader, input_dim, num_classes = get_dataloaders()
    xb, yb = next(iter(train_loader))
    print(f"input_dim={input_dim}, num_classes={num_classes}")
    print("Batch X shape:", xb.shape, "Batch y shape:", yb.shape)
