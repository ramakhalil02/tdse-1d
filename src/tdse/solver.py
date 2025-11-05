from __future__ import annotations
import numpy as np
from numpy.linalg import eigh
from typing import Callable, Tuple

from .grid import kinetic_matrix
from .observables import normalize

Array = np.ndarray

def hamiltonian_from_potential(V: Array, dx: float) -> Array:
    """Build H = T + V (diagonal) given V(x)."""
    n = V.size
    T = kinetic_matrix(n, dx)
    H = T + np.diag(V)
    return H

def stationary_states(H: Array, k: int | None = None) -> Tuple[Array, Array]:
    """
    Diagonalize H: returns (E, Phi) with Phi[:,i] eigenvector of E[i].
    If k is provided and k < N, return the lowest k eigenvalues/vectors.
    """
    E, Phi = eigh(H)
    if k is not None and k < E.size:
        E = E[:k]
        Phi = Phi[:, :k]
    # Normalize eigenvectors (L2)
    for i in range(Phi.shape[1]):
        Phi[:, i] = normalize(Phi[:, i])
    return E, Phi

def propagate(
    psi0: Array,
    H_of_t: Callable[[float], Array],
    t_grid: Array,
) -> Tuple[Array, Array]:
    """
    Propagate psi using instantaneous eigen-decomposition at each step:
      psi_{n+1} = sum_i exp(-i*E_i*dt) |phi_i><phi_i| psi_n
    Returns (psis, energies_expectation) where
        psis shape = (T, N), energy exp value uses H at each step.
    """
    N = psi0.size
    T = t_grid.size
    psis = np.zeros((T, N), dtype=complex)
    Es = np.zeros(T, dtype=float)

    psis[0] = normalize(psi0)
    H_prev = H_of_t(t_grid[0])

    # expectation value at t0
    Es[0] = float(np.vdot(psis[0], H_prev @ psis[0]).real)

    for ti in range(1, T):
        t_prev, t_next = t_grid[ti-1], t_grid[ti]
        dt = t_next - t_prev
        H_next = H_of_t(t_next)

        # Use instantaneous eigenbasis at t_next
        Evals, Evecs = eigh(H_next)
        psi_prev = psis[ti-1]
        coeffs = Evecs.conj().T @ psi_prev
        phase = np.exp(-1j * Evals * dt)
        psi_next = Evecs @ (phase * coeffs)

        psis[ti] = normalize(psi_next)
        Es[ti] = float(np.vdot(psis[ti], H_next @ psis[ti]).real)

        H_prev = H_next

    return psis, Es
