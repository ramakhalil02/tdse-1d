
import numpy as np
from tdse.grid import make_grid
from tdse.potentials import harmonic
from tdse.solver import hamiltonian_from_potential, stationary_states, propagate

def test_norm_preserved():
    x, dx = make_grid(-4, 4, 401)
    V = harmonic(x)
    H = hamiltonian_from_potential(V, dx)
    E, Phi = stationary_states(H, k=1)
    psi0 = Phi[:,0]
    t = np.linspace(0, 0.1, 5)
    psis, _ = propagate(psi0, lambda _t: H, t)  # time-independent
    norms = [float((p.conj()@p).real) for p in psis]
    assert np.allclose(norms, np.ones_like(norms), atol=1e-6)
