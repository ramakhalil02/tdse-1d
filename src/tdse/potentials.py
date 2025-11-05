from __future__ import annotations
import numpy as np

def harmonic(x: np.ndarray, omega: float = 1.0) -> np.ndarray:
    """V(x) = 1/2 * omega^2 * x^2"""
    return 0.5 * (omega**2) * x**2

def gaussian_bump(x: np.ndarray, C1: float = 10.0, C2: float = 5.0) -> np.ndarray:
    """Symmetric Gaussian: C1 * exp(-C2 * x^2)."""
    return C1 * np.exp(-C2 * x**2)

def bimodal_antisymmetric(x: np.ndarray, A: float = 5.0, B: float = 5.0, mu: float = 0.5) -> np.ndarray:
    """
    Anti-symmetric bimodal Gaussian: A*exp(-B*(x-mu)^2) - A*exp(-B*(x+mu)^2).
    """
    return A * np.exp(-B * (x - mu)**2) - A * np.exp(-B * (x + mu)**2)

def total_potential(x: np.ndarray, base: str = "harmonic", **kwargs) -> np.ndarray:
    """
    Compose total potential: V_total = V_harmonic + optional extras.
    kwargs may include keys accepted by gaussian_bump or bimodal_antisymmetric.
    """
    V = harmonic(x, omega=kwargs.pop("omega", 1.0))
    extra = kwargs.pop("extra", None)
    if extra == "gaussian":
        V = V + gaussian_bump(x, **kwargs)
    elif extra == "antisymmetric":
        V = V + bimodal_antisymmetric(x, **kwargs)
    return V
