from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from .grid import make_grid
from .potentials import harmonic, gaussian_bump, bimodal_antisymmetric
from .solver import hamiltonian_from_potential, stationary_states, propagate
from .plot import save_eigen_plot, save_probability_animation

def cli():
    p = argparse.ArgumentParser(description="TDSE 1D harmonic oscillator with perturbations")
    p.add_argument("--xmin", type=float, default=-6.0)
    p.add_argument("--xmax", type=float, default=6.0)
    p.add_argument("--nx", type=int, default=801, help="number of grid points")
    p.add_argument("--omega", type=float, default=1.0)

    p.add_argument("--potential", choices=["unperturbed", "symmetric", "antisymmetric"], default="unperturbed")
    p.add_argument("--C1", type=float, default=10.0)  # symmetric
    p.add_argument("--C2", type=float, default=5.0)
    p.add_argument("--A", type=float, default=5.0)    # antisymmetric
    p.add_argument("--B", type=float, default=5.0)
    p.add_argument("--mu", type=float, default=0.5)

    p.add_argument("--task", choices=["eigs", "propagate"], default="eigs")

    p.add_argument("--tmax", type=float, default=np.pi)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--Omega", type=float, default=1.0, help="temporal modulation frequency")

    p.add_argument("--outdir", type=str, default="outputs/run")
    p.add_argument("--save_prefix", type=str, default="tdse")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Grid & base potential
    x, dx = make_grid(args.xmin, args.xmax, args.nx)

    # Static potential V(x) per mode
    if args.potential == "unperturbed":
        Vx = harmonic(x, omega=args.omega)
        label = "unperturbed"
    elif args.potential == "symmetric":
        Vx = harmonic(x, omega=args.omega) + gaussian_bump(x, C1=args.C1, C2=args.C2)
        label = "symmetric"
    else:
        Vx = harmonic(x, omega=args.omega) + bimodal_antisymmetric(x, A=args.A, B=args.B, mu=args.mu)
        label = "antisymmetric"

    H0 = hamiltonian_from_potential(Vx, dx)

    if args.task == "eigs":
        E, Phi = stationary_states(H0, k=8)
        save_eigen_plot(x, Vx, E, Phi, outdir / f"{args.save_prefix}_{label}_eigs.png")
        energies_path = outdir / f"{args.save_prefix}_{label}_energies.json"
        energies_path.write_text(json.dumps({"energies": E.tolist()}, indent=2))
        print(f"Saved eigen plot and energies to: {outdir}")
    else:
        # time-dependent modulation: V(x,t) = V_ho(x) + sin(Omega t) * V_pert(x)
        def H_of_t(t: float):
            Vt = harmonic(x, omega=args.omega)
            if args.potential == "unperturbed":
                Vt = Vt
            elif args.potential == "symmetric":
                Vt = Vt + np.sin(args.Omega * t) * gaussian_bump(x, C1=args.C1, C2=args.C2)
            else:
                Vt = Vt + np.sin(args.Omega * t) * bimodal_antisymmetric(x, A=args.A, B=args.B, mu=args.mu)
            return hamiltonian_from_potential(Vt, dx)

        # initial state = ground state of H0
        E0, Phi0 = stationary_states(H0, k=1)
        psi0 = Phi0[:, 0]

        t_grid = np.arange(0.0, args.tmax + 1e-12, args.dt)
        psis, Es = propagate(psi0, H_of_t, t_grid)
        # Save animation
        anim_path = outdir / f"{args.save_prefix}_{label}_propagation"
        saved_path = save_probability_animation(x, psis, Vx, t_grid, str(anim_path))
        # Save energies
        (outdir / f"{args.save_prefix}_{label}_energies_t.json").write_text(
            json.dumps({"t": t_grid.tolist(), "E_expect": Es.tolist()}, indent=2)
        )
        print(f"Saved animation to: {saved_path}")
        print(f"Saved energy vs time to: {outdir}")

if __name__ == "__main__":
    cli()
