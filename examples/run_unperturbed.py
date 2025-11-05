
from tdse.grid import make_grid
from tdse.potentials import harmonic
from tdse.solver import hamiltonian_from_potential, stationary_states
from tdse.plot import save_eigen_plot
from pathlib import Path

x, dx = make_grid(-6, 6, 801)
V = harmonic(x, omega=1.0)
H = hamiltonian_from_potential(V, dx)
E, Phi = stationary_states(H, k=5)
save_eigen_plot(x, V, E, Phi, Path("outputs/examples_unperturbed.png"))
print("Saved outputs/examples_unperturbed.png")
