"""Supersonic flow over a cylinder: Mach sweep, grid study and figures for the 2D blunt-body extension.

    python scripts/blunt_body.py            # full run: grid study at M = 3 and Mach sweep (~25 min)
    python scripts/blunt_body.py --quick    # coarse grid only (~3 min)

Writes results/blunt_body.md and figures/blunt_*.png.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shocktube import blunt2d as b

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / "figures", ROOT / "results"


def run(M, Ni, Nj, flux="hll"):
    grid = b.make_grid(Ni, Nj, b.outer_radius(M), 1.03 ** (64 / Ni))
    t0 = time.perf_counter()
    r = b.solve(M, grid=grid, flux=flux, max_iters=9000, first_order_iters=1500, tol=1e-4)
    return r, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    lines = ["# Supersonic flow over a circular cylinder", ""]

    # ---- grid study at M = 3
    grids = [(32, 48), (64, 96)] if args.quick else [(32, 48), (64, 96), (128, 192)]
    lines += ["## Grid study, M = 3, HLL + van Leer", "",
              "| Grid (radial x angular) | Standoff Δ/R | Billig | Difference | p₀/p∞ | Pitot (exact) | Difference |",
              "|---|---|---|---|---|---|---|"]
    for Ni, Nj in grids:
        r, _ = run(3.0, Ni, Nj)
        d, p0 = b.shock_standoff(r), b.stagnation_pressure(r)
        dB, pP = b.billig_standoff(3.0), b.pitot_pressure_ratio(3.0)
        lines.append(f"| {Ni} x {Nj} | {d:.3f} | {dB:.3f} | {100 * (d / dB - 1):+.1f}% | {p0:.3f} | {pP:.3f} | {100 * (p0 / pP - 1):+.2f}% |")
        print(lines[-1], flush=True)

    # ---- Mach sweep
    Ni, Nj = (64, 96) if args.quick else (96, 144)
    lines += ["", f"## Mach sweep, {Ni} x {Nj} grid, HLL + van Leer", "",
              "| M∞ | Standoff Δ/R | Billig | Difference | p₀/p∞ | Pitot (exact) | Difference | Iterations | Time (s) |",
              "|---|---|---|---|---|---|---|---|---|"]
    sweep, keep = [], {}
    for M in [2.0, 3.0, 4.0, 6.0, 8.0]:
        r, sec = run(M, Ni, Nj)
        d, p0 = b.shock_standoff(r), b.stagnation_pressure(r)
        sweep.append((M, d))
        keep[M] = r
        lines.append(f"| {M:g} | {d:.3f} | {b.billig_standoff(M):.3f} | {100 * (d / b.billig_standoff(M) - 1):+.1f}% | "
                     f"{p0:.2f} | {b.pitot_pressure_ratio(M):.2f} | {100 * (p0 / b.pitot_pressure_ratio(M) - 1):+.2f}% | "
                     f"{r.iterations} | {sec:.0f} |")
        print(lines[-1], flush=True)
    (RES / "blunt_body.md").write_text("\n".join(lines) + "\n")

    # ---- figures
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    r = keep[4.0]
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    cs = ax.contourf(r.grid.xc, r.grid.yc, b.mach_field(r), levels=np.linspace(0, 4.04, 34), cmap="viridis")
    ax.contour(r.grid.xc, r.grid.yc, b.mach_field(r), levels=[1.0], colors="white", linewidths=1.2)
    yy = np.linspace(-3, 3, 200)
    ax.plot(b.billig_shock_shape(4.0, yy), yy, "r--", lw=1.2, label="Billig shock shape")
    th = np.linspace(-np.pi / 2, np.pi / 2, 200)
    ax.fill(-np.cos(th), np.sin(th), color="0.85", ec="k", lw=1)
    ax.set_aspect("equal")
    ax.set_xlim(-3.2, 0.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("x / R")
    ax.set_ylabel("y / R")
    ax.set_title("Mach number, M∞ = 4 (white line: sonic line)")
    ax.legend(loc="lower left", frameon=False)
    fig.colorbar(cs, ax=ax, shrink=0.8, label="Mach number")
    fig.tight_layout()
    fig.savefig(FIG / "blunt_mach_M4.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    Ms = np.linspace(1.8, 9, 200)
    ax.plot(Ms, b.billig_standoff(Ms), "k-", label="Billig (1967) correlation")
    ax.plot(*zip(*sweep), "o", color="C0", label=f"this solver, {Ni} x {Nj}")
    ax.set_xlabel("Free-stream Mach number")
    ax.set_ylabel("Shock standoff Δ / R")
    ax.legend(frameon=False)
    ax.set_title("Bow-shock standoff on a circular cylinder")
    fig.tight_layout()
    fig.savefig(FIG / "blunt_standoff.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for M, rr in keep.items():
        ax.semilogy(rr.residuals, lw=1, label=f"M = {M:g}")
    ax.axvline(1500, color="0.6", lw=0.8, ls=":")
    ax.text(1550, 1, "second order on", fontsize=8, color="0.4", va="top")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Density residual (relative)")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "blunt_residuals.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
