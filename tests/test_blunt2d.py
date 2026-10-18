import numpy as np
import pytest

from shocktube import blunt2d as b
from shocktube.reconstruction import get_limiter


def test_cells_are_closed():
    """Face vectors of every cell sum to zero (geometric conservation law)."""
    g = b.make_grid(20, 30, 3.0)
    for k in range(2):
        net = g.Si[k, 1:] - g.Si[k, :-1] + g.Sj[k, :, 1:] - g.Sj[k, :, :-1]
        assert np.max(np.abs(net)) < 1e-14


def test_grid_orientation_and_area():
    g = b.make_grid(20, 30, 3.0)
    # wall faces point outward (radially), angular faces toward increasing theta
    assert np.all(g.Si[0, 0] * g.xn[0, :-1] + g.Si[1, 0] * g.yn[0, :-1] > 0)
    # total area of the half annulus
    assert np.isclose(g.area.sum(), 0.5 * np.pi * (3.0**2 - 1.0), rtol=2e-3)


@pytest.mark.parametrize("flux", ["hll", "hllc"])
def test_free_stream_preserved_away_from_the_wall(flux):
    g = b.make_grid(24, 36, 3.0)
    W = b.freestream(3.0)
    U = np.broadcast_to(b.prim_to_cons(W)[:, None, None], (4,) + g.shape).copy()
    R = b.residual(U, g, W, b.FLUXES[flux], get_limiter("vanleer"))
    assert np.max(np.abs(R[:, 2:, 2:-2])) < 1e-11


@pytest.mark.parametrize("flux", ["hll", "hllc"])
def test_consistency_with_physical_flux(flux):
    rng = np.random.default_rng(1)
    W = np.array([rng.uniform(0.5, 2, 5), rng.uniform(-2, 2, 5), rng.uniform(-2, 2, 5), rng.uniform(0.5, 2, 5)])
    ang = rng.uniform(0, 2 * np.pi, 5)
    S = np.array([np.cos(ang), np.sin(ang)]) * 0.3
    F = b.face_flux(b.FLUXES[flux], W, W, S)
    rho, u, v, p = W
    E = p / 0.4 + 0.5 * rho * (u**2 + v**2)
    un = u * S[0] + v * S[1]
    exact = np.array([rho * un, rho * u * un + p * S[0], rho * v * un + p * S[1], (E + p) * un])
    assert np.allclose(F, exact, rtol=1e-12, atol=1e-12)


def test_hllc_preserves_a_stationary_contact_with_shear():
    """HLLC keeps a contact (density and tangential-velocity jump, same p, zero normal velocity)
    exactly; HLL smears it."""
    WL = np.array([[1.0], [0.0], [1.0], [1.0]])
    WR = np.array([[0.2], [0.0], [-1.0], [1.0]])
    S = np.array([[1.0], [0.0]])
    F = b.face_flux(b.hllc, WL, WR, S)
    assert np.allclose(F[:, 0], [0.0, 1.0, 0.0, 0.0])
    assert abs(b.face_flux(b.hll, WL, WR, S)[0, 0]) > 1e-3


def test_reference_formulas():
    assert np.isclose(b.pitot_pressure_ratio(2.0), 5.640, atol=1e-3)     # NACA 1135 table
    assert np.isclose(b.pitot_pressure_ratio(3.0), 12.06, atol=1e-2)
    assert np.isclose(b.normal_shock_pressure_ratio(2.0), 4.5)
    assert np.isclose(b.billig_standoff(1e6), 0.386)
    assert np.isclose(b.billig_shock_shape(4.0, 0.0), -(1.0 + b.billig_standoff(4.0)))
