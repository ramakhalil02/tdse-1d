
import numpy as np
from tdse.grid import make_grid
from tdse.potentials import harmonic
from tdse.solver import hamiltonian_from_potential, stationary_states

def test_ho_energies_spacing():
    x, dx = make_grid(-6, 6, 801)
    V = harmonic(x, omega=1.0)
    H = hamiltonian_from_potential(V, dx)
    E, _ = stationary_states(H, k=5)
    diffs = np.diff(E)
    assert np.allclose(diffs, np.ones_like(diffs), atol=5e-3)
