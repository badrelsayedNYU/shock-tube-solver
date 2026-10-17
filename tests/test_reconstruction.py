"""MUSCL reconstruction, limiters and positivity fallback (spec §3.3)."""

import numpy as np
import pytest

from shocktube.cases import CASES, RiemannCase
from shocktube.analysis import l1_error
from shocktube.reconstruction import (LIMITERS, limited_slopes, minmod, muscl_faces,
                                      positivity_fallback, superbee, van_leer)
from shocktube.solver import solve

R = np.array([-2.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 10.0])


def test_limiter_values():
    np.testing.assert_allclose(minmod(R), [0, 0, 0, 0.25, 0.5, 1, 1, 1, 1])
    np.testing.assert_allclose(van_leer(R), [0, 0, 0, 0.4, 2 / 3, 1, 4 / 3, 1.5, 20 / 11])
    np.testing.assert_allclose(superbee(R), [0, 0, 0, 0.5, 1, 1, 2, 2, 2])


@pytest.mark.parametrize("name", ["minmod", "vanleer", "superbee"])
def test_limiters_are_tvd_and_symmetric(name):
    """Sweby TVD region 0 <= phi <= min(2r, 2) for r > 0, phi = 0 for r <= 0, phi(1) = 1,
    and symmetry phi(r)/r = phi(1/r), which makes phi(r)*Delta_- independent of direction."""
    phi = LIMITERS[name]
    r = np.linspace(1e-3, 20, 2000)
    assert np.all(phi(r) <= np.minimum(2 * r, 2) + 1e-14) and np.all(phi(r) >= 0)
    assert np.all(phi(-r) == 0)
    assert phi(np.array([1.0]))[0] == pytest.approx(1.0)
    np.testing.assert_allclose(phi(r) / r, phi(1 / r), rtol=1e-12)


@pytest.mark.parametrize("name", ["minmod", "vanleer", "superbee"])
def test_linear_data_reconstructed_exactly(name):
    x = np.linspace(0, 1, 12)
    W = np.array([1 + 2 * x, 0.5 - x, 3 + x])
    WL, WR, _ = muscl_faces(W, LIMITERS[name])
    faces = 0.5 * (x[1:-2] + x[2:-1])  # faces between interior cells
    exact = np.array([1 + 2 * faces, 0.5 - faces, 3 + faces])
    # atol: some exact face values are 0, where only an absolute round-off tolerance makes sense
    np.testing.assert_allclose(WL, exact, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(WR, exact, rtol=1e-13, atol=1e-14)


def test_slope_zero_at_extremum_and_flat_data():
    W = np.array([[1.0, 2.0, 1.0, 1.0, 1.0]] * 3)
    slopes = limited_slopes(W, van_leer)  # cells 1..3: extremum, then flat (Delta_- = 0 guard)
    np.testing.assert_array_equal(slopes, 0.0)


def test_positivity_fallback_zeroes_only_bad_cells():
    """Forced input: a slope that would put the face pressure below zero in cell 1 only."""
    W = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.1, 1.0]])
    slopes = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.1, 0.5, 0.1]])
    new, n = positivity_fallback(W, slopes)
    assert n == 1
    np.testing.assert_array_equal(new[:, 1], 0.0)
    np.testing.assert_array_equal(new[:, [0, 2]], slopes[:, [0, 2]])


@pytest.mark.parametrize("flux", ["rusanov", "roe", "hllc"])
@pytest.mark.parametrize("limiter", ["minmod", "vanleer", "superbee"])
def test_blast_runs_second_order(flux, limiter):
    """Test 3 (strong blast) must complete at second order without a positivity failure."""
    res = solve(CASES["blast"], N=400, flux=flux, limiter=limiter)
    assert res.t == CASES["blast"].t_end
    assert np.all(res.W[0] > 0) and np.all(res.W[2] > 0)


@pytest.mark.parametrize("flux", ["rusanov", "hllc"])
def test_second_order_sharper_than_first_on_sod(flux):
    case = CASES["sod"]
    e1 = l1_error(solve(case, N=200, flux=flux), case)
    e2 = l1_error(solve(case, N=200, flux=flux, limiter="vanleer"), case)
    assert e2 < 0.6 * e1


@pytest.mark.parametrize("limiter", ["vanleer", "superbee"])
def test_smooth_wave_converges_at_second_order(limiter):
    """Density wave rho = 1 + 0.2 sin(2 pi x), u = 1, p = 1 on a periodic domain: the exact
    solution is pure advection. Minmod is excluded: it clips smooth extrema to first order."""
    case = RiemannCase("wave", "advected density wave", (1, 1, 1), (1, 1, 1), x0=0.5, t_end=1.0)

    def W_init(x):
        return np.array([1 + 0.2 * np.sin(2 * np.pi * x), np.ones_like(x), np.ones_like(x)])

    errors = []
    for N in (50, 100, 200, 400):
        res = solve(case, N=N, flux="hllc", limiter=limiter, bc="periodic", W_init=W_init)
        errors.append(np.sum(np.abs(res.W[0] - W_init(res.x - case.t_end)[0])) * res.dx)
    order = np.log2(errors[-2] / errors[-1])
    assert order > 1.8, f"errors = {errors}, observed order = {order:.2f}"
