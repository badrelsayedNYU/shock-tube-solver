"""Error measures against the exact Riemann solution."""

import numpy as np

from shocktube.exact import exact_solution


def l1_error(result, case, var=0):
    """Discrete L1 error  sum_i |q_i - q_exact(x_i, t)| dx  (default q = rho).

    The exact solution is sampled at cell centers.
    """
    W_exact = exact_solution(result.x, result.t, case.WL, case.WR, case.x0, case.gamma)
    return np.sum(np.abs(result.W[var] - W_exact[var])) * result.dx
