
from pathlib import Path
import numpy as np
from tdse.grid import make_grid
from tdse.potentials import harmonic, gaussian_bump
from tdse.solver import hamiltonian_from_potential, stationary_states, propagate
from tdse.plot import save_eigen_plot, save_probability_animation

x, dx = make_grid(-6, 6, 801)
V = harmonic(x) + gaussian_bump(x, C1=10.0, C2=5.0)
H = hamiltonian_from_potential(V, dx)
E, Phi = stationary_states(H, k=6)
save_eigen_plot(x, V, E, Phi, Path("outputs/examples_symmetric_eigs.png"))

# time propagation with Ω = 1
def H_of_t(t):
    return hamiltonian_from_potential(harmonic(x) + np.sin(1.0*t)*gaussian_bump(x, C1=10.0, C2=5.0), dx)

psi0 = Phi[:, 0]
t = np.arange(0, np.pi+1e-12, 0.01)
psis, Es = propagate(psi0, H_of_t, t)
save_probability_animation(x, psis, V, t, "outputs/examples_symmetric_prop")
print("Saved outputs/examples_symmetric_eigs.png and animation in outputs/.")
