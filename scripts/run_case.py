"""Run one test case and compare it against the exact solution.

Usage: python scripts/run_case.py --case sod --flux rusanov --N 400 [--plot out.png]
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from shocktube.analysis import l1_error
from shocktube.cases import CASES
from shocktube.exact import exact_solution
from shocktube.fluxes import FLUXES
from shocktube.reconstruction import LIMITERS
from shocktube.solver import solve
from make_figures import label_axes, plot_profiles


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default="sod", choices=sorted(CASES))
    ap.add_argument("--flux", default="rusanov", choices=sorted(FLUXES))
    ap.add_argument("--limiter", default="none", choices=sorted(LIMITERS))
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--cfl", type=float, default=None,
                    help="default 0.8 for first order, 0.5 for second order")
    ap.add_argument("--bc", default="transmissive", choices=["transmissive", "periodic"])
    ap.add_argument("--plot", metavar="PATH", help="save an overlay plot vs. the exact solution")
    args = ap.parse_args()

    case = CASES[args.case]
    t0 = time.perf_counter()
    res = solve(case, N=args.N, flux=args.flux, limiter=args.limiter, cfl=args.cfl, bc=args.bc)
    runtime = time.perf_counter() - t0
    print(f"{case.description} | flux={args.flux} limiter={args.limiter} N={args.N}")
    print(f"  t = {res.t}, steps = {res.n_steps}, runtime = {runtime:.3f} s, "
          f"positivity fallbacks = {res.n_fallback}")
    print(f"  L1 error (rho) = {l1_error(res, case):.4e}")

    if args.plot:
        x = np.linspace(0, 1, 4001)
        W = exact_solution(x, case.t_end, case.WL, case.WR, case.x0, case.gamma)
        fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, layout="constrained")
        plot_profiles(axes, x, W, case.gamma, color="black", lw=1.5, label="Exact")
        plot_profiles(axes, res.x, res.W, case.gamma, ls="none", marker="o", ms=2.5,
                      label=f"{args.flux} + {args.limiter}, N = {args.N}")
        label_axes(axes)
        axes[0, 0].legend()
        fig.suptitle(f"{case.description}, $t = {case.t_end}$")
        fig.savefig(args.plot, dpi=200)
        print(f"  wrote {args.plot}")


if __name__ == "__main__":
    main()
