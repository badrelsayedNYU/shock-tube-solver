"""Numerical flux functions F_{i+1/2} = flux(WL, WR) for the 1D Euler equations.

All fluxes take primitive face states WL, WR of shape (3, n_faces) and return the
flux, shape (3, n_faces). Extra keyword arguments (e.g. gamma) are passed through.
"""

import numpy as np

from shocktube.gas import GAMMA, physical_flux, prim_to_cons, sound_speed


def rusanov(WL, WR, gamma=GAMMA):
    """Rusanov / local Lax-Friedrichs flux  (spec §3.2a; Toro §10.5.1).

    F = 1/2 (F_L + F_R) - 1/2 S_max (U_R - U_L),
    S_max = max(|u_L| + a_L, |u_R| + a_R).
    """
    UL, UR = prim_to_cons(WL, gamma), prim_to_cons(WR, gamma)
    FL, FR = physical_flux(WL, gamma), physical_flux(WR, gamma)
    aL = sound_speed(WL[0], WL[2], gamma)
    aR = sound_speed(WR[0], WR[2], gamma)
    S_max = np.maximum(np.abs(WL[1]) + aL, np.abs(WR[1]) + aR)
    return 0.5 * (FL + FR) - 0.5 * S_max * (UR - UL)


FLUXES = {
    "rusanov": rusanov,
}


def get_flux(name):
    """Look up a numerical flux function by name."""
    try:
        return FLUXES[name]
    except KeyError:
        raise ValueError(f"unknown flux {name!r}; choose from {sorted(FLUXES)}") from None
