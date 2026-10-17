"""Regenerate every README figure (figures/) and table (results/).

Usage: python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import comparison
import convergence
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


def fig_first_vs_second_order_sod(N=100):
    """HLLC on Sod: first order vs. MUSCL + SSP-RK2 with van Leer and superbee (milestone 4)."""
    case = CASES["sod"]
    x = np.linspace(0.0, 1.0, 4001)
    W = exact_solution(x, case.t_end, case.WL, case.WR, case.x0, case.gamma)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, layout="constrained")
    plot_profiles(axes, x, W, case.gamma, color="black", lw=1.5, label="Exact")
    for limiter, marker, color, label in (
            ("none", "o", "tab:gray", "1st order (CFL 0.8)"),
            ("vanleer", "s", "tab:blue", "MUSCL van Leer + SSP-RK2 (CFL 0.5)"),
            ("superbee", "^", "tab:red", "MUSCL superbee + SSP-RK2 (CFL 0.5)")):
        res = solve(case, N=N, flux="hllc", limiter=limiter)
        plot_profiles(axes, res.x, res.W, case.gamma, ls="none", marker=marker, ms=3.5,
                      mfc="none", color=color, label=f"HLLC, {label}")
    label_axes(axes)
    axes[0, 0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle(f"Sod, $t = {case.t_end}$, $N = {N}$: first vs. second order (HLLC flux)\n"
                 r"(nondimensional units, $\gamma = %.1f$)" % case.gamma)
    out = FIG_DIR / "sod_first_vs_second_order.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_overlay(case_name, N=100, limiter="vanleer"):
    """§6.1: rho, u, p, e (rows) for each flux (columns): exact, first order, second order."""
    case = CASES[case_name]
    g = case.gamma
    x = np.linspace(0.0, 1.0, 4001)
    We = exact_solution(x, case.t_end, case.WL, case.WR, case.x0, g)
    fluxes = (("rusanov", "Rusanov"), ("roe", "Roe (+ Harten fix)"), ("hllc", "HLLC"))
    ylabels = (r"$\rho$", r"$u$", r"$p$", r"$e$")

    def rows(W):
        return W[0], W[1], W[2], internal_energy(W[0], W[2], g)

    fig, axes = plt.subplots(4, 3, figsize=(11, 10), sharex=True, sharey="row",
                             layout="constrained")
    for col, (flux, title) in enumerate(fluxes):
        axes[0, col].set_title(title)
        for ax, y in zip(axes[:, col], rows(We)):
            ax.plot(x, y, color="black", lw=1.3, label="Exact")
        for lim, marker, color, label in (("none", "o", "tab:gray", "1st order"),
                                          (limiter, "s", "tab:blue", f"MUSCL {limiter} + SSP-RK2")):
            try:
                res = solve(case, N=N, flux=flux, limiter=lim)
            except PositivityError as err:
                axes[3 if lim == "none" else 2, col].text(
                    0.5, 0.5, f"{label}: FAILED\n$p < 0$ at $t$ = {err.t:.2g}, x = {err.x:.3f}",
                    transform=axes[3 if lim == "none" else 2, col].transAxes, ha="center",
                    va="center", fontsize=8, color="tab:red",
                    bbox=dict(facecolor="white", edgecolor="tab:red"))
                continue
            for ax, y in zip(axes[:, col], rows(res.W)):
                ax.plot(res.x, y, ls="none", marker=marker, ms=2.8, mfc="none", color=color,
                        label=label)
    for ax, lab in zip(axes[:, 0], ylabels):
        ax.set_ylabel(lab)
    for ax in axes[-1]:
        ax.set_xlabel(r"Position $x$")
    axes[0, 0].legend(fontsize=7.5, loc="best")
    fig.suptitle(f"{case.description}, $t = {case.t_end}$, $N = {N}$\n"
                 r"(nondimensional units, $\gamma = %.1f$; $e = p/((\gamma-1)\rho)$)" % g)
    out = FIG_DIR / f"overlay_{case_name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    for name in CASES:
        fig_exact(name)
        fig_overlay(name)
    fig_first_order_rusanov_sod()
    fig_expansion_shock()
    fig_positivity_123()
    fig_first_vs_second_order_sod()
    convergence.main()
    comparison.main()


if __name__ == "__main__":
    main()
