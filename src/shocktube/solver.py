"""Finite-volume solver for the 1D Euler equations (spec §3.1).

Uniform cell-centered grid on [0, 1] with NG ghost cells per side. The semi-discrete
operator is L(U)_i = -(F_{i+1/2} - F_{i-1/2}) / dx with MUSCL face states; forward Euler
or SSP-RK2 advances dU/dt = L(U).
"""

from dataclasses import dataclass

import numpy as np

from shocktube.fluxes import get_flux
from shocktube.gas import GAMMA, cons_to_prim, prim_to_cons, sound_speed
from shocktube.reconstruction import get_limiter, muscl_faces

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
    n_fallback: int = 0        # cell-stages where the positivity fallback zeroed the slopes


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


def spatial_operator(U, dx, flux, bc, phi, gamma=GAMMA, flux_kwargs=None):
    """Semi-discrete operator L(U)_i = -(F_{i+1/2} - F_{i-1/2})/dx  (spec §3.1; Toro §6.2, §13.4).

    Face states come from MUSCL reconstruction of the primitive variables with limiter
    phi (phi = 0 gives the first-order, piecewise-constant face states).
    Returns (L, F_left, F_right, n_fallback), where F_left/F_right are the fluxes through
    the domain boundary faces x = 0 and x = 1 (needed for the open-system budget) and
    n_fallback counts cells whose slopes were zeroed by the positivity fallback.
    """
    W = cons_to_prim(fill_ghosts(U, bc), gamma)
    WL, WR, n_fallback = muscl_faces(W, phi)  # N+1 faces, x = 0 ... x = 1
    F = flux(WL, WR, gamma=gamma, **(flux_kwargs or {}))
    return -(F[:, 1:] - F[:, :-1]) / dx, F[:, 0], F[:, -1], n_fallback


def cfl_time_step(U, dx, cfl, gamma=GAMMA):
    """dt = C dx / max_i(|u_i| + a_i)  (spec §3.1; Toro §6.3.2)."""
    rho, u, p = cons_to_prim(U, gamma)
    return cfl * dx / np.max(np.abs(u) + sound_speed(rho, p, gamma))


# Shu-Osher form of the integrators: stage k computes
#   U^(k) = a_k U^n + b_k (U^(k-1) + dt L(U^(k-1))),  U^(0) = U^n.
# The boundary flux of stage k enters the budget with the same overall weight that
# L(U^(k-1)) has in U^{n+1}  (spec §7, conservation test).
INTEGRATORS = {
    "euler": {"stages": [(0.0, 1.0)], "flux_weights": [1.0]},
    # SSP-RK2 (Heun): U1 = U + dt L(U);  U^{n+1} = 1/2 U + 1/2 (U1 + dt L(U1))  (Gottlieb & Shu 1998)
    "ssprk2": {"stages": [(0.0, 1.0), (0.5, 0.5)], "flux_weights": [0.5, 0.5]},
}


def solve(case, N=400, flux="rusanov", limiter="none", time_integrator=None, cfl=None,
          bc="transmissive", t_end=None, flux_kwargs=None, W_init=None):
    """Run a case to t_end  (spec §3).

    limiter "none" defaults to forward Euler with CFL 0.8; any other limiter defaults
    to SSP-RK2 with CFL 0.5. The last step is clipped to land on t_end. Positivity is
    checked after every stage. The time-integrated boundary flux uses the same stage
    weights as the update, so sum(U^n - U^0) dx == boundary_flux to round-off.
    W_init(x) -> primitive (3, N) overrides the case's Riemann initial data.
    """
    gamma = case.gamma
    t_end = case.t_end if t_end is None else t_end
    second_order = limiter != "none"
    time_integrator = time_integrator or ("ssprk2" if second_order else "euler")
    cfl = cfl if cfl is not None else (0.5 if second_order else 0.8)
    flux_fn, phi = get_flux(flux), get_limiter(limiter)
    scheme = INTEGRATORS[time_integrator]

    x, dx = make_grid(N)
    U0 = initial_condition(case, x) if W_init is None else prim_to_cons(W_init(x), gamma)
    check_positivity(U0, x, 0.0, gamma, " (initial data)")

    U = U0.copy()
    t, n_steps, n_fallback = 0.0, 0, 0
    boundary_flux = np.zeros(3)
    while t < t_end:
        dt = min(cfl_time_step(U, dx, cfl, gamma), t_end - t)
        t_new = t_end if t_end - t <= dt else t + dt
        U_stage = U
        for k, ((a, b), w) in enumerate(zip(scheme["stages"], scheme["flux_weights"])):
            L, F_left, F_right, nf = spatial_operator(U_stage, dx, flux_fn, bc, phi, gamma, flux_kwargs)
            U_stage = a * U + b * (U_stage + dt * L)
            boundary_flux += w * dt * (F_left - F_right)
            n_fallback += nf
            check_positivity(U_stage, x, t_new, gamma, f" (stage {k + 1})")
        U, t = U_stage, t_new
        n_steps += 1

    return Result(x=x, W=cons_to_prim(U, gamma), U0=U0, U=U, t=t, dx=dx, n_steps=n_steps,
                  boundary_flux=boundary_flux, n_fallback=n_fallback)
