"""
device.py
Detección automática del dispositivo de cómputo disponible: CUDA (GPU NVIDIA),
MPS (GPU Apple Silicon) o CPU como fallback.
"""

import torch


def get_device() -> torch.device:
    """
    Detecta y devuelve el mejor dispositivo disponible para entrenar el modelo.

    Orden de prioridad:
        1. CUDA (GPU NVIDIA)
        2. MPS (GPU Apple Silicon, Mac M1/M2/M3)
        3. CPU (fallback universal)

    Returns:
        torch.device: dispositivo seleccionado.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Dispositivo detectado: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Dispositivo detectado: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Dispositivo detectado: CPU")
    return device


if __name__ == "__main__":
    get_device()
