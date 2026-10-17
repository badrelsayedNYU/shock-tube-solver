"""Scheme comparison on Sod at N = 400 (spec §6.3, §6.4).

For every flux x limiter: L1(rho) error, runtime (median of repeated runs), and the
number of cells across the shock and the contact (see analysis.sod_resolution).
Writes results/comparison_sod_N400.{md,csv}.
Usage: python scripts/comparison.py
"""

import time
from pathlib import Path

import numpy as np

from shocktube.analysis import l1_error, sod_resolution
from shocktube.cases import CASES
from shocktube.solver import solve

ROOT = Path(__file__).resolve().parent.parent
N = 400
REPEATS = 7
FLUXES = ("rusanov", "roe", "hllc")
LIMITERS = ("none", "minmod", "vanleer", "superbee")


def timed_run(case, flux, limiter):
    """Median wall-clock time over REPEATS runs (after one warm-up run)."""
    res = solve(case, N=N, flux=flux, limiter=limiter)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        solve(case, N=N, flux=flux, limiter=limiter)
        times.append(time.perf_counter() - t0)
    return res, float(np.median(times))


def main():
    case = CASES["sod"]
    rows = []
    for flux in FLUXES:
        for limiter in LIMITERS:
            res, runtime = timed_run(case, flux, limiter)
            width = sod_resolution(res, case)
            rows.append(dict(flux=flux, limiter=limiter, l1=l1_error(res, case),
                             runtime_ms=1e3 * runtime, steps=res.n_steps,
                             shock_cells=width["shock"], contact_cells=width["contact"]))
            print(rows[-1])

    (ROOT / "results").mkdir(exist_ok=True)
    keys = list(rows[0])
    with open(ROOT / "results" / "comparison_sod_N400.csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")

    best = min(r["l1"] for r in rows)
    lines = [
        f"Sod, N = {N}, t = {case.t_end}. Runtime: median of {REPEATS} runs on one CPU core "
        "(first order: forward Euler, CFL 0.8; second order: SSP-RK2, CFL 0.5). "
        "Cells across a discontinuity: cells inside the 5-95% band of the jump "
        "(shock measured in p, contact in rho).",
        "",
        "| Flux | Limiter | L1 error (rho) | Runtime (ms) | Steps | Cells across shock | Cells across contact |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        l1 = f"**{r['l1']:.3e}**" if r["l1"] == best else f"{r['l1']:.3e}"
        lines.append(f"| {r['flux']} | {r['limiter']} | {l1} | {r['runtime_ms']:.1f} | {r['steps']} "
                     f"| {r['shock_cells']} | {r['contact_cells']} |")
    out = ROOT / "results" / "comparison_sod_N400.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
