"""Calorically perfect gas utilities for the 1D Euler equations.

Array convention: state arrays have shape (3, ...), with the variable index
first. Primitive W = [rho, u, p]; conserved U = [rho, rho*u, E].
"""

import numpy as np

GAMMA = 1.4


def prim_to_cons(W, gamma=GAMMA):
    """Primitive -> conserved variables.

    U = [rho, rho*u, E], with E = p/(gamma - 1) + 0.5*rho*u^2  (spec §2; Toro ch. 1).
    """
    rho, u, p = W
    return np.array([rho, rho * u, p / (gamma - 1.0) + 0.5 * rho * u**2])


def cons_to_prim(U, gamma=GAMMA):
    """Conserved -> primitive variables.

    u = (rho*u)/rho,  p = (gamma - 1)*(E - 0.5*rho*u^2)  (spec §2; Toro ch. 1).
    """
    rho, mom, E = U
    u = mom / rho
    return np.array([rho, u, (gamma - 1.0) * (E - 0.5 * rho * u**2)])


def sound_speed(rho, p, gamma=GAMMA):
    """Sound speed a = sqrt(gamma*p/rho)  (spec §2; ideal gas, Toro ch. 1)."""
    return np.sqrt(gamma * p / rho)


def total_enthalpy(W, gamma=GAMMA):
    """Specific total enthalpy H = (E + p)/rho  (spec §2)."""
    rho, _, p = W
    E = prim_to_cons(W, gamma)[2]
    return (E + p) / rho


def internal_energy(rho, p, gamma=GAMMA):
    """Specific internal energy e = p/((gamma - 1)*rho)  (ideal gas; Toro ch. 1)."""
    return p / ((gamma - 1.0) * rho)


def physical_flux(W, gamma=GAMMA):
    """Physical Euler flux F(U) = [rho*u, rho*u^2 + p, u*(E + p)]  (spec §2; Toro ch. 3)."""
    rho, u, p = W
    E = prim_to_cons(W, gamma)[2]
    return np.array([rho * u, rho * u**2 + p, u * (E + p)])
