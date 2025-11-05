# TDSE – 1D Harmonic Oscillator with Perturbations

Refactored from the BSc thesis *"Numerical solutions to the time-dependent Schrödinger equation for a model problem"*.

## Features
- Finite-difference discretization of 1D Hamiltonian
- Harmonic oscillator base potential (ω=1 by default)
- Symmetric Gaussian bump and antisymmetric bimodal Gaussian perturbations
- Eigen-solve for stationary states
- Time propagation using instantaneous eigen-decomposition (per thesis Eq. (42))
- Matplotlib plots and **automatic animation saving** (MP4 if ffmpeg is available, else GIF)

## Install (editable)
```bash
pip install -e .
```

## CLI Examples
Compute eigenstates (symmetric bump):
```bash
tdse --potential symmetric --C1 10 --C2 5 --task eigs --outdir outputs/symmetric
```

Propagate ground state with sinusoidal modulation Ω = ω = 1:
```bash
tdse --potential symmetric --dt 0.01 --tmax 3.14159 --Omega 1.0 --task propagate --outdir outputs/symmetric_time
```

Propagate with Ω = 2ω (adds net energy):
```bash
tdse --potential symmetric --Omega 2.0 --task propagate --outdir outputs/symmetric_2omega
```

Antisymmetric perturbation:
```bash
tdse --potential antisymmetric --A 5 --B 5 --mu 0.5 --task propagate --outdir outputs/antisymm
```

## Notes on Units
We use ħ=1, m=1, ω=1. The unperturbed harmonic oscillator has:
- Hamiltonian:  H = -1/2 d²/dx² + 1/2 x²
- Energies:     Eₙ = n + 1/2

This differs from an alternative scaling in the thesis where energies appear as 2n+1.
All results are consistent up to a fixed factor.

## Outputs
- `*_eigs.png` plots the potential and first eigenfunctions (shifted by energies).
- `*_propagation.mp4` or `.gif` shows |ψ(x,t)|² vs time (saved automatically).
- JSON files include eigenvalues and ⟨H⟩(t).

## Python
Tested with Python 3.11.
