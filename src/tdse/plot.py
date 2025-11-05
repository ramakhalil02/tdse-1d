from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from pathlib import Path

from .observables import probabilities

def save_eigen_plot(x: np.ndarray, V: np.ndarray, E: np.ndarray, Phi: np.ndarray, outpath: str) -> None:
    """
    Plot potential and first few eigenfunctions (shifted by energy) and save as PNG.
    """
    fig, ax = plt.subplots(figsize=(6,4), dpi=160)
    ax.plot(x, V, label="V(x)")
    for i in range(min(5, Phi.shape[1])):
        ax.plot(x, Phi[:, i] + E[i], label=f"$\\phi_{i}(x)$ + E{i}")
    ax.set_xlabel("x")
    ax.set_ylabel("Energy / Amplitude (shifted)")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Potential and eigenfunctions")
    fig.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)

def save_probability_animation(
    x: np.ndarray,
    psis: np.ndarray,
    V: np.ndarray | None,
    t_grid: np.ndarray,
    outpath: str,
    fps: int = 30,
) -> str:
    """
    Save an animation of |psi(x,t)|^2 over time. Tries MP4 via FFMpeg first, falls back to GIF.
    """
    P = probabilities(psis)
    Pmax = float(np.max(P)) * 1.05 if np.max(P) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(6,4), dpi=160)
    line, = ax.plot([], [])
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, Pmax)
    ax.set_xlabel("x")
    ax.set_ylabel("|psi(x,t)|^2")
    if V is not None:
        V_scaled = (V - V.min()) / (V.ptp() + 1e-12) * Pmax
        ax.plot(x, V_scaled, alpha=0.4, linestyle="--")

    title = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        line.set_data([], [])
        title.set_text("")
        return line, title

    def update(i):
        line.set_data(x, P[i])
        title.set_text(f"t = {t_grid[i]:.3f}")
        return line, title

    anim = FuncAnimation(fig, update, frames=len(t_grid), init_func=init, blit=True, interval=1000/fps)
    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(str(out.with_suffix(".mp4")), writer=writer)
        saved = out.with_suffix(".mp4")
    except Exception:
        writer = PillowWriter(fps=fps)
        anim.save(str(out.with_suffix(".gif")), writer=writer)
        saved = out.with_suffix(".gif")

    plt.close(fig)
    return str(saved)
