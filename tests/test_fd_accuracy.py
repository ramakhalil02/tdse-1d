
import numpy as np
from tdse.grid import make_grid, kinetic_matrix

def test_kinetic_symmetry():
    x, dx = make_grid(-1, 1, 101)
    T = kinetic_matrix(len(x), dx)
    assert np.allclose(T, T.T)
