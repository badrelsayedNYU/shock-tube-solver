"""Flux consistency: flux(W, W) == F(W) for every numerical flux (spec §7)."""

import numpy as np
import pytest

from shocktube.fluxes import FLUXES
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
