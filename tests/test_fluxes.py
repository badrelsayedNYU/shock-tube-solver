"""Flux consistency: flux(W, W) == F(W) for every numerical flux (spec §7)."""

import numpy as np
import pytest

from shocktube.fluxes import FLUXES, hllc, roe
from shocktube.gas import physical_flux

STATES = np.array([
    [1.0, 0.125, 1.0, 5.99924, 1e-3],     # rho
    [0.0, 0.0, -2.0, 19.5975, 0.75],      # u
    [1.0, 0.1, 0.4, 460.894, 1e-4],       # p
])


@pytest.mark.parametrize("name", sorted(FLUXES))
def test_flux_consistency(name):
    np.testing.assert_allclose(FLUXES[name](STATES, STATES), physical_flux(STATES),
                               rtol=1e-13, atol=1e-13)


def test_roe_supersonic_upwinding():
    """When all lambda_i > 0, F = 1/2(F_L + F_R) - 1/2 sum lambda_i alpha_i r_i equals F_L only if
    sum lambda_i alpha_i r_i = F_R - F_L (Roe's property U). This checks the wave strengths
    alpha_i and eigenvectors r_i together; likewise all lambda_i < 0 must give F_R."""
    WL, WR = STATES[:, :-1], STATES[:, 1:]
    WL_sup = WL.copy(); WR_sup = WR.copy()
    WL_sup[1] += 100.0; WR_sup[1] += 100.0
    np.testing.assert_allclose(roe(WL_sup, WR_sup, entropy_fix=False), physical_flux(WL_sup),
                               rtol=1e-11)
    WL_sup[1] -= 200.0; WR_sup[1] -= 200.0
    np.testing.assert_allclose(roe(WL_sup, WR_sup, entropy_fix=False), physical_flux(WR_sup),
                               rtol=1e-11)


@pytest.mark.parametrize("flux", [roe, hllc])
def test_stationary_contact_is_exact(flux):
    """A stationary contact (u = 0, equal p, density jump) has zero numerical dissipation."""
    WL = np.array([[1.0], [0.0], [0.5]])
    WR = np.array([[0.1], [0.0], [0.5]])
    np.testing.assert_allclose(flux(WL, WR), physical_flux(WL), atol=1e-14)


def test_rusanov_smears_stationary_contact():
    WL = np.array([[1.0], [0.0], [0.5]])
    WR = np.array([[0.1], [0.0], [0.5]])
    assert abs(FLUXES["rusanov"](WL, WR)[0, 0]) > 0.1


def test_harten_fix_only_touches_acoustic_waves():
    """Contact with tiny u~: the fix must leave lambda_2 alone, so the flux still has zero mass dissipation."""
    WL = np.array([[1.0], [0.0], [0.5]])
    WR = np.array([[0.1], [0.0], [0.5]])
    np.testing.assert_allclose(roe(WL, WR, entropy_fix=True), roe(WL, WR, entropy_fix=False),
                               atol=1e-14)
