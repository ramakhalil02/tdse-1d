from __future__ import annotations
import numpy as np

def make_grid(xmin: float, xmax: float, num: int) -> tuple[np.ndarray, float]:
    """
    Create a uniform spatial grid.
    Returns (x, dx).
    """
    x = np.linspace(xmin, xmax, num, dtype=float)
    dx = x[1] - x[0]
    return x, dx

def kinetic_matrix(n: int, dx: float) -> np.ndarray:
    """
    Construct the finite-difference kinetic energy matrix for 1D with ħ=1, m=1.
    H_kin = -1/2 * d^2/dx^2 using 3-point stencil.
    """
    main = np.full(n, -2.0)
    off = np.ones(n-1)
    lap = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / (dx**2)
    return -0.5 * lap
