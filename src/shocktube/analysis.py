"""Error measures and discontinuity-resolution metrics against the exact Riemann solution."""

import numpy as np

from shocktube.exact import exact_solution, star_state
from shocktube.gas import sound_speed


def l1_error(result, case, var=0):
    """Discrete L1 error  sum_i |q_i - q_exact(x_i, t)| dx  (default q = rho).

    The exact solution is sampled at cell centers.
    """
    W_exact = exact_solution(result.x, result.t, case.WL, case.WR, case.x0, case.gamma)
    return np.sum(np.abs(result.W[var] - W_exact[var])) * result.dx


def observed_order(N, errors):
    """Least-squares slope p of log(error) = -p log(N) + c."""
    return -np.polyfit(np.log(N), np.log(errors), 1)[0]


def drop_across(result, x_face, var=0):
    """q_{i-1} - q_i for the two cells on either side of the face nearest x_face."""
    i = int(np.searchsorted(result.x, x_face))
    return result.W[var][i - 1] - result.W[var][i]


def sod_wave_positions(case, t):
    """Positions at time t of the fan tail, contact and shock for a left-fan / right-shock
    problem such as Sod (Toro §4.5 wave speeds)."""
    p_star, u_star = star_state(case.WL, case.WR, case.gamma)
    rhoL, uL, pL = case.WL
    rhoR, uR, pR = case.WR
    g = case.gamma
    assert p_star <= pL and p_star > pR, "expects a left rarefaction and a right shock"
    aL, aR = sound_speed(rhoL, pL, g), sound_speed(rhoR, pR, g)
    tail = u_star - aL * (p_star / pL) ** ((g - 1) / (2 * g))
    shock = uR + aR * np.sqrt((g + 1) / (2 * g) * p_star / pR + (g - 1) / (2 * g))
    return {"tail": case.x0 + tail * t, "contact": case.x0 + u_star * t,
            "shock": case.x0 + shock * t}


def cells_in_transition(q, x, window, q_a, q_b, lo=0.05, hi=0.95):
    """Number of cells in `window` = (x_min, x_max) whose value lies strictly between
    lo and hi of the jump from q_a to q_b. Overshoots outside the jump are not counted."""
    s = (q - q_a) / (q_b - q_a)
    inside = (x > window[0]) & (x < window[1])
    return int(np.count_nonzero(inside & (s > lo) & (s < hi)))


def sod_resolution(result, case):
    """Cells across the shock (measured in p) and the contact (measured in rho)  (spec §6.4).

    Shock: p between 5% and 95% of the jump p* -> p_R, in the window from the
    contact/shock midpoint to x = 1 (p is continuous across the contact).
    Contact: rho between 5% and 95% of the jump rho*_L -> rho*_R, in the window from
    the fan tail to the contact/shock midpoint.
    """
    pos = sod_wave_positions(case, result.t)
    mid = 0.5 * (pos["contact"] + pos["shock"])
    xs = np.array([pos["contact"] - 1e-9, pos["contact"] + 1e-9])
    rho_starL, rho_starR = exact_solution(xs, result.t, case.WL, case.WR, case.x0, case.gamma)[0]
    p_star, _ = star_state(case.WL, case.WR, case.gamma)
    rho, _, p = result.W
    return {
        "shock": cells_in_transition(p, result.x, (mid, 1.0), p_star, case.WR[2]),
        "contact": cells_in_transition(rho, result.x, (pos["tail"], mid), rho_starL, rho_starR),
    }
