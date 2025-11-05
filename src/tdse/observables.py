from __future__ import annotations
import numpy as np

def norm(psi: np.ndarray) -> float:
    return float(np.vdot(psi, psi).real)

def normalize(psi: np.ndarray) -> np.ndarray:
    n = np.sqrt(norm(psi))
    return psi / (n if n != 0 else 1.0)

def probabilities(psi: np.ndarray) -> np.ndarray:
    return np.abs(psi)**2
