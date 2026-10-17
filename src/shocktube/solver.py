"""Finite-volume solver for the 1D Euler equations (spec §3.1).

Uniform cell-centered grid on [0, 1] with NG ghost cells per side. The semi-discrete
operator is L(U)_i = -(F_{i+1/2} - F_{i-1/2}) / dx; the time integrator advances
dU/dt = L(U).
"""

from dataclasses import dataclass

import numpy as np

from shocktube.fluxes import get_flux
from shocktube.gas import GAMMA, cons_to_prim, prim_to_cons, sound_speed

NG = 2  # ghost cells per side


class PositivityError(RuntimeError):
    """Raised when a cell reaches rho <= 0 or p <= 0 (or NaN) during the run."""

    def __init__(self, message, t, cell, x, rho, p):
        super().__init__(message)
        self.t, self.cell, self.x, self.rho, self.p = t, cell, x, rho, p


@dataclass
class Result:
    x: np.ndarray            # cell centers, shape (N,)
    W: np.ndarray            # primitive state at t, shape (3, N)
    U0: np.ndarray           # conserved initial state, shape (3, N)
    U: np.ndarray            # conserved final state, shape (3, N)
    t: float
    dx: float
    n_steps: int
    boundary_flux: np.ndarray  # integral over time of (F_left_face - F_right_face), shape (3,)


def make_grid(N):
    """Cell centers x_i = (i + 1/2) dx on [0, 1], dx = 1/N."""
    dx = 1.0 / N
    return (np.arange(N) + 0.5) * dx, dx


def initial_condition(case, x):
    """Piecewise-constant Riemann data: W_L for x < x0, W_R otherwise. Returns conserved U."""
    left = x < case.x0
    W = np.where(left, np.array(case.WL)[:, None], np.array(case.WR)[:, None])
    return prim_to_cons(W, case.gamma)


def fill_ghosts(U, bc):
    """Return U padded with NG ghost cells per side.

    transmissive: zero-gradient, ghost = nearest interior cell (Toro §6.3.3).
    periodic:     ghost cells copy the cells at the opposite end.
    """
    if bc == "transmissive":
        return np.concatenate([np.repeat(U[:, :1], NG, axis=1), U,
                               np.repeat(U[:, -1:], NG, axis=1)], axis=1)
    if bc == "periodic":
        return np.concatenate([U[:, -NG:], U, U[:, :NG]], axis=1)
    raise ValueError(f"unknown boundary condition {bc!r}")


def check_positivity(U, x, t, gamma=GAMMA, stage=""):
    """Raise PositivityError naming the first cell with rho <= 0 or p <= 0 (or NaN)."""
    rho, _, p = cons_to_prim(U, gamma)
    bad = ~((rho > 0) & (p > 0))  # also catches NaN
    if np.any(bad):
        i = int(np.argmax(bad))
        raise PositivityError(
            f"non-physical state at t = {t:.6g}{stage}: cell {i} (x = {x[i]:.4f}), "
            f"rho = {rho[i]:.4g}, p = {p[i]:.4g}",
            t=t, cell=i, x=x[i], rho=rho[i], p=p[i],
        )


def spatial_operator(U, dx, flux, bc, gamma=GAMMA, flux_kwargs=None):
    """First-order semi-discrete operator L(U)_i = -(F_{i+1/2} - F_{i-1/2})/dx  (spec §3.1; Toro §6.2).

    Face states are the adjacent cell averages (piecewise-constant reconstruction).
    Returns (L, F_left, F_right), where F_left/F_right are the fluxes through the
    domain boundary faces x = 0 and x = 1 (needed for the open-system budget).
    """
    N = U.shape[1]
    W = cons_to_prim(fill_ghosts(U, bc), gamma)
    WL = W[:, NG - 1:NG + N]   # left state at faces 1/2 - 1 ... N + 1/2 - 1, i.e. N+1 faces
    WR = W[:, NG:NG + N + 1]
    F = flux(WL, WR, gamma=gamma, **(flux_kwargs or {}))
    return -(F[:, 1:] - F[:, :-1]) / dx, F[:, 0], F[:, -1]


def cfl_time_step(U, dx, cfl, gamma=GAMMA):
    """dt = C dx / max_i(|u_i| + a_i)  (spec §3.1; Toro §6.3.2)."""
    rho, u, p = cons_to_prim(U, gamma)
    return cfl * dx / np.max(np.abs(u) + sound_speed(rho, p, gamma))


def solve(case, N=400, flux="rusanov", cfl=0.8, bc="transmissive", t_end=None, flux_kwargs=None):
    """Run a Riemann case to t_end with first-order forward-Euler time stepping.

    U^{n+1} = U^n + dt L(U^n)  (spec §3.1). The last step is clipped to land on t_end.
    The time-integrated boundary flux uses the same weights as the update, so
    sum(U^n - U^0) dx == boundary_flux to round-off.
    """
    gamma = case.gamma
    t_end = case.t_end if t_end is None else t_end
    flux_fn = get_flux(flux)
    x, dx = make_grid(N)
    U0 = initial_condition(case, x)
    check_positivity(U0, x, 0.0, gamma, " (initial data)")

    U = U0.copy()
    t, n_steps = 0.0, 0
    boundary_flux = np.zeros(3)
    while t < t_end:
        dt = min(cfl_time_step(U, dx, cfl, gamma), t_end - t)
        L, F_left, F_right = spatial_operator(U, dx, flux_fn, bc, gamma, flux_kwargs)
        U = U + dt * L
        boundary_flux += dt * (F_left - F_right)
        t = t_end if t_end - t <= dt else t + dt
        n_steps += 1
        check_positivity(U, x, t, gamma)

    return Result(x=x, W=cons_to_prim(U, gamma), U0=U0, U=U, t=t, dx=dx,
                  n_steps=n_steps, boundary_flux=boundary_flux)
