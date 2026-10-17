"""Sanity checks for the analysis metrics."""

import numpy as np

from shocktube.analysis import observed_order, sod_resolution
from shocktube.cases import CASES
from shocktube.exact import exact_solution
from shocktube.solver import Result, make_grid


def test_observed_order_recovers_power_law():
    N = np.array([50, 100, 200, 400])
    np.testing.assert_allclose(observed_order(N, 3.0 * N**-2.0), 2.0)
    np.testing.assert_allclose(observed_order(N, 0.7 * N**-0.5), 0.5)


def test_exact_data_has_no_transition_cells():
    """The exact solution sampled at cell centers jumps within one cell: zero cells
    lie strictly inside the 5-95% band of either discontinuity."""
    case = CASES["sod"]
    x, dx = make_grid(400)
    W = exact_solution(x, case.t_end, case.WL, case.WR, case.x0)
    res = Result(x=x, W=W, U0=None, U=None, t=case.t_end, dx=dx, n_steps=0,
                 boundary_flux=np.zeros(3))
    assert sod_resolution(res, case) == {"shock": 0, "contact": 0}
