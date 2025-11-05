"""
tdse: Time-Dependent Schrödinger Equation utilities for 1D harmonic oscillator models.

This package provides:
- Finite-difference spatial discretization
- Standard and custom perturbation potentials (symmetric Gaussian, antisymmetric bimodal Gaussian)
- Eigen-solver for time-independent Hamiltonians
- Simple time propagation using instantaneous eigen-decomposition
- Plotting utilities for eigenstates and time evolution (animations saved automatically)

Default units: ħ = 1, m = 1, ω = 1.
Then H_ho = -1/2 d^2/dx^2 + 1/2 x^2 has eigenvalues E_n = n + 1/2.
"""
__all__ = [
    "grid",
    "potentials",
    "solver",
    "observables",
    "plot",
]
