"""Grid convergence study (spec §6.2).

Left panel: L1(rho) vs. N on Sod for N = 50 ... 1600.
Right panel: the same schemes on a smooth periodic density wave, where the formal
order is attainable, for contrast.

Writes figures/convergence.png and results/convergence_sod.csv.
Usage: python scripts/convergence.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shocktube.analysis import l1_error, observed_order
from shocktube.cases import CASES, RiemannCase
from shocktube.solver import solve

ROOT = Path(__file__).resolve().parent.parent
N_LIST = np.array([50, 100, 200, 400, 800, 1600])
FLUXES = ("rusanov", "roe", "hllc")
LIMITERS = ("none", "minmod", "vanleer", "superbee")

# (flux, limiter, label, color, marker) for the plotted subset
PLOTTED = (
    ("rusanov", "none", "Rusanov, 1st order", "tab:green", "o"),
    ("roe", "none", "Roe, 1st order", "tab:orange", "D"),
    ("hllc", "none", "HLLC, 1st order", "tab:gray", "s"),
    ("hllc", "minmod", "HLLC + minmod", "tab:purple", "v"),
    ("hllc", "vanleer", "HLLC + van Leer", "tab:blue", "^"),
    ("hllc", "superbee", "HLLC + superbee", "tab:red", "P"),
)

WAVE = RiemannCase("wave", "smooth density wave", (1.0, 1.0, 1.0), (1.0, 1.0, 1.0),
                   x0=0.5, t_end=1.0)


def wave_init(x):
    """rho = 1 + 0.2 sin(2 pi x), u = 1, p = 1: exact solution is rho(x - t) (pure advection)."""
    return np.array([1.0 + 0.2 * np.sin(2 * np.pi * x), np.ones_like(x), np.ones_like(x)])


def sod_errors(flux, limiter):
    case = CASES["sod"]
    return np.array([l1_error(solve(case, N=N, flux=flux, limiter=limiter), case) for N in N_LIST])


def wave_errors(flux, limiter):
    errs = []
    for N in N_LIST:
        res = solve(WAVE, N=N, flux=flux, limiter=limiter, bc="periodic", W_init=wave_init)
        errs.append(np.sum(np.abs(res.W[0] - wave_init(res.x - WAVE.t_end)[0])) * res.dx)
    return np.array(errs)


def add_reference_slopes(ax, orders, x_anchor, y_anchor):
    for p in orders:
        N = np.array([x_anchor, x_anchor * 4])
        ax.plot(N, y_anchor * (N / x_anchor) ** -p, color="black", ls="--", lw=0.8)
        ax.text(N[1] * 1.05, y_anchor * 4.0**-p, f"slope {p:g}", fontsize=8, va="center")


def main():
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    sod = {(f, l): sod_errors(f, l) for f in FLUXES for l in LIMITERS}
    with open(ROOT / "results" / "convergence_sod.csv", "w") as fh:
        fh.write("flux,limiter," + ",".join(f"N={N}" for N in N_LIST) + ",observed_order\n")
        for (f, l), e in sod.items():
            fh.write(f"{f},{l}," + ",".join(f"{v:.6e}" for v in e) +
                     f",{observed_order(N_LIST, e):.3f}\n")
            print(f"Sod  {f:8s} {l:9s} order = {observed_order(N_LIST, e):.2f}")

    fig, (ax_sod, ax_wave) = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained")
    for flux, limiter, label, color, marker in PLOTTED:
        e = sod[(flux, limiter)]
        ax_sod.loglog(N_LIST, e, marker=marker, color=color, mfc="none",
                      label=f"{label} (order {observed_order(N_LIST, e):.2f})")
        ew = wave_errors(flux, limiter)
        print(f"wave {flux:8s} {limiter:9s} order = {observed_order(N_LIST, ew):.2f}")
        ax_wave.loglog(N_LIST, ew, marker=marker, color=color, mfc="none",
                       label=f"{label} (order {observed_order(N_LIST, ew):.2f})")

    add_reference_slopes(ax_sod, (0.5, 1.0), 60, 8e-4)
    add_reference_slopes(ax_wave, (1.0, 2.0), 60, 3e-5)
    for ax, title in ((ax_sod, "Sod shock tube (discontinuous), $t = 0.25$"),
                      (ax_wave, r"Smooth density wave, periodic, $t = 1$")):
        ax.set_xlabel("Number of cells $N$")
        ax.set_ylabel(r"$L_1$ error in density $\rho$")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.suptitle("Grid convergence (orders: least-squares fit over N = 50-1600)")
    out = ROOT / "figures" / "convergence.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
