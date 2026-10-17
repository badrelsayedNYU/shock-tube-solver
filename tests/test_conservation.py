"""Discrete conservation of mass, momentum and energy (spec §7).

(1) Periodic boundaries: the system is closed, so the totals sum_i U_i dx must stay
    constant to round-off.
(2) Transmissive boundaries (open system): the change in the totals must equal the
    time-integrated boundary flux, accumulated with the same weights as the update.
"""

import numpy as np
import pytest

from shocktube.cases import CASES
from shocktube.solver import PositivityError, check_positivity, make_grid, solve

EPS = np.finfo(float).eps


def totals(U, dx):
    return U.sum(axis=1) * dx


def roundoff_tol(res):
    """Round-off scale: eps * (cells summed) * (steps) * magnitude of the totals."""
    scale = np.abs(res.U).sum(axis=1) * res.dx
    return 10 * EPS * scale * np.sqrt(res.x.size * res.n_steps)


@pytest.mark.parametrize("name", ["sod", "123", "blast", "modsod"])
def test_periodic_totals_constant(name):
    res = solve(CASES[name], N=200, flux="rusanov", bc="periodic")
    drift = totals(res.U, res.dx) - totals(res.U0, res.dx)
    assert np.all(np.abs(drift) <= roundoff_tol(res)), f"drift = {drift}"


def test_transmissive_sod_budget_closes():
    res = solve(CASES["sod"], N=200, flux="rusanov", bc="transmissive")
    change = totals(res.U, res.dx) - totals(res.U0, res.dx)
    # Momentum genuinely changes by about (p_L - p_R) t = 0.225, so this is a real check.
    assert abs(change[1]) > 0.2
    assert np.all(np.abs(change - res.boundary_flux) <= roundoff_tol(res)), \
        f"change = {change}, boundary flux = {res.boundary_flux}"


def test_positivity_check_names_bad_cell():
    x, _ = make_grid(4)
    U = np.array([[1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [2.5, 2.5, -0.1, 2.5]])  # negative energy -> negative pressure in cell 2
    with pytest.raises(PositivityError, match=r"cell 2 \(x = 0\.6250\)"):
        check_positivity(U, x, t=0.1)
