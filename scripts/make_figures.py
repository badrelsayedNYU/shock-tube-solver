"""Regenerate every README figure into figures/.

Usage: python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shocktube.cases import CASES
from shocktube.exact import exact_solution
from shocktube.gas import internal_energy
from shocktube.solver import PositivityError, solve

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def plot_profiles(axes, x, W, gamma, **style):
    """Plot rho, u, p and internal energy e on a 2x2 grid of axes."""
    rho, u, p = W
    for ax, y in zip(axes.flat, (rho, u, p, internal_energy(rho, p, gamma))):
        ax.plot(x, y, **style)


def label_axes(axes):
    labels = (r"Density $\rho$", r"Velocity $u$", r"Pressure $p$", r"Internal energy $e$")
    for ax, lab in zip(axes.flat, labels):
        ax.set_ylabel(lab)
    for ax in axes[-1]:
        ax.set_xlabel(r"Position $x$")


def fig_exact(case_name):
    """Exact solution of one test case at t_end (Toro ch. 4)."""
    case = CASES[case_name]
    x = np.linspace(0.0, 1.0, 4001)
    W = exact_solution(x, case.t_end, case.WL, case.WR, case.x0, case.gamma)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, layout="constrained")
    plot_profiles(axes, x, W, case.gamma, color="black", lw=1.5, label="Exact")
    label_axes(axes)
    axes[0, 0].legend(loc="best")
    fig.suptitle(f"{case.description}: exact solution at $t = {case.t_end}$\n"
                 r"(nondimensional units, $\gamma = %.1f$)" % case.gamma)
    out = FIG_DIR / f"exact_{case_name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_first_order_rusanov_sod():
    """First-order Rusanov on Sod at two resolutions vs. exact (milestone 2)."""
    case = CASES["sod"]
    x = np.linspace(0.0, 1.0, 4001)
    W = exact_solution(x, case.t_end, case.WL, case.WR, case.x0, case.gamma)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, layout="constrained")
    plot_profiles(axes, x, W, case.gamma, color="black", lw=1.5, label="Exact")
    for N, marker, color in ((100, "o", "tab:blue"), (400, ".", "tab:orange")):
        res = solve(case, N=N, flux="rusanov")
        plot_profiles(axes, res.x, res.W, case.gamma, ls="none", marker=marker, ms=3,
                      mfc="none" if marker == "o" else color, color=color,
                      label=f"Rusanov, 1st order, $N = {N}$")
    label_axes(axes)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Sod, $t = {case.t_end}$: first-order Rusanov vs. exact\n"
                 r"(nondimensional units, $\gamma = %.1f$, CFL = 0.8)" % case.gamma)
    out = FIG_DIR / "sod_rusanov_first_order.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_expansion_shock(N=100):
    """§6.5a: first-order Roe with and without the Harten fix on Case 4 (transonic fan)."""
    case = CASES["modsod"]
    x = np.linspace(0.0, 1.0, 4001)
    W = exact_solution(x, case.t_end, case.WL, case.WR, case.x0, case.gamma)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, layout="constrained")
    plot_profiles(axes, x, W, case.gamma, color="black", lw=1.5, label="Exact")
    for flux, marker, color, label in (("roe-nofix", "o", "tab:red", "Roe, no entropy fix"),
                                       ("roe", "s", "tab:blue", "Roe + Harten fix")):
        res = solve(case, N=N, flux=flux)
        plot_profiles(axes, res.x, res.W, case.gamma, ls="none", marker=marker, ms=3.5,
                      mfc="none", color=color, label=f"{label}, $N = {N}$")
    for ax in axes.flat:
        ax.axvline(case.x0, color="gray", ls=":", lw=1)
    label_axes(axes)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Modified Sod (transonic fan), $t = {case.t_end}$: first-order Roe\n"
                 r"dotted line: $x_0$, where the sonic point $x/t = 0$ sits"
                 r" (nondimensional, $\gamma = %.1f$, CFL = 0.8)" % case.gamma)
    out = FIG_DIR / "roe_expansion_shock_modsod.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_positivity_123(N=100):
    """§6.5b: 123 problem, Roe with/without fix vs. Rusanov and HLLC."""
    case = CASES["123"]
    x = np.linspace(0.0, 1.0, 4001)
    W = exact_solution(x, case.t_end, case.WL, case.WR, case.x0, case.gamma)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, layout="constrained")
    plot_profiles(axes, x, W, case.gamma, color="black", lw=1.5, label="Exact")
    failures = []
    for flux, marker, color, label in (("rusanov", "o", "tab:green", "Rusanov"),
                                       ("hllc", "s", "tab:purple", "HLLC"),
                                       ("roe", None, None, "Roe + Harten fix"),
                                       ("roe-nofix", None, None, "Roe, no fix")):
        try:
            res = solve(case, N=N, flux=flux)
        except PositivityError as err:
            failures.append(f"{label}: FAILED at t = {err.t:.3g} (step 1 = first dt),\n"
                            f"  cell {err.cell}, x = {err.x:.3f}: "
                            f"$\\rho$ = {err.rho:.3g}, $p$ = {err.p:.3g}")
            print(f"  {label}: {err}")
            continue
        plot_profiles(axes, res.x, res.W, case.gamma, ls="none", marker=marker, ms=3.5,
                      mfc="none", color=color, label=f"{label}, $N = {N}$")
    label_axes(axes)
    axes[0, 0].legend(loc="upper center", fontsize=8)
    if failures:
        axes[0, 1].text(0.03, 0.97, "\n".join(failures), transform=axes[0, 1].transAxes,
                        va="top", fontsize=7, color="tab:red",
                        bbox=dict(facecolor="white", edgecolor="tab:red", alpha=0.9))
    fig.suptitle(f'"123" double rarefaction, $t = {case.t_end}$: first-order schemes\n'
                 r"(nondimensional units, $\gamma = %.1f$, CFL = 0.8)" % case.gamma)
    out = FIG_DIR / "positivity_123.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    for name in CASES:
        fig_exact(name)
    fig_first_order_rusanov_sod()
    fig_expansion_shock()
    fig_positivity_123()


if __name__ == "__main__":
    main()
