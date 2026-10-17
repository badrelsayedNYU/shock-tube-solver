"""Exact Riemann solver for the 1D Euler equations, ideal gas (Toro ch. 4).

Left/right states are primitive triples W = (rho, u, p).
"""

import numpy as np

from shocktube.gas import GAMMA, sound_speed


def pressure_function(p, rho_K, p_K, a_K, gamma=GAMMA):
    """Wave function f_K(p) and its derivative f_K'(p) for side K = L or R  (spec §4; Toro §4.2-4.3).

    Shock (p > p_K):         f_K = (p - p_K) * sqrt(A_K / (p + B_K)),
                             A_K = 2/((gamma+1) rho_K),  B_K = (gamma-1)/(gamma+1) p_K
    Rarefaction (p <= p_K):  f_K = 2 a_K/(gamma-1) * [(p/p_K)^((gamma-1)/(2 gamma)) - 1]

    The derivatives are the analytic d/dp of these expressions (used by Newton's method).
    """
    if p > p_K:
        A = 2.0 / ((gamma + 1.0) * rho_K)
        B = (gamma - 1.0) / (gamma + 1.0) * p_K
        root = np.sqrt(A / (p + B))
        f = (p - p_K) * root
        df = root * (1.0 - 0.5 * (p - p_K) / (p + B))
    else:
        z = (gamma - 1.0) / (2.0 * gamma)
        f = 2.0 * a_K / (gamma - 1.0) * ((p / p_K) ** z - 1.0)
        df = 1.0 / (rho_K * a_K) * (p / p_K) ** (-(gamma + 1.0) / (2.0 * gamma))
    return f, df


def two_rarefaction_guess(WL, WR, gamma=GAMMA):
    """Initial guess for p* from the two-rarefaction approximation  (Toro §4.3.2).

    p_TR = [ (a_L + a_R - (gamma-1)/2 (u_R - u_L)) / (a_L/p_L^z + a_R/p_R^z) ]^(1/z),
    z = (gamma-1)/(2 gamma). Exact when both nonlinear waves are rarefactions, and
    always positive, so it is a safe Newton starting point.
    """
    rhoL, uL, pL = WL
    rhoR, uR, pR = WR
    aL, aR = sound_speed(rhoL, pL, gamma), sound_speed(rhoR, pR, gamma)
    z = (gamma - 1.0) / (2.0 * gamma)
    num = aL + aR - 0.5 * (gamma - 1.0) * (uR - uL)
    den = aL / pL**z + aR / pR**z
    return (num / den) ** (1.0 / z)


def star_state(WL, WR, gamma=GAMMA, tol=1e-8, max_iter=100):
    """Solve f(p) = f_L(p) + f_R(p) + (u_R - u_L) = 0 for p*, then get u*  (spec §4; Toro §4.3).

    Newton iteration from the two-rarefaction guess, stopped when the relative pressure
    change 2|p_new - p_old|/(p_new + p_old) < tol (Toro §4.3.1).
    u* = (u_L + u_R)/2 + [f_R(p*) - f_L(p*)]/2.

    Returns (p_star, u_star). Raises ValueError if the data generate vacuum
    (pressure positivity condition, Toro §4.2): (u_R - u_L) >= 2(a_L + a_R)/(gamma - 1).
    """
    rhoL, uL, pL = WL
    rhoR, uR, pR = WR
    aL, aR = sound_speed(rhoL, pL, gamma), sound_speed(rhoR, pR, gamma)
    du = uR - uL

    if du >= 2.0 * (aL + aR) / (gamma - 1.0):
        raise ValueError("initial data generate vacuum; not supported by this solver")

    p = two_rarefaction_guess(WL, WR, gamma)
    for _ in range(max_iter):
        fL, dfL = pressure_function(p, rhoL, pL, aL, gamma)
        fR, dfR = pressure_function(p, rhoR, pR, aR, gamma)
        p_new = p - (fL + fR + du) / (dfL + dfR)
        p_new = max(p_new, tol)  # keep pressure positive (Toro's safeguard)
        converged = 2.0 * abs(p_new - p) / (p_new + p) < tol
        p = p_new
        if converged:
            break
    else:
        raise RuntimeError(f"Newton iteration for p* did not converge in {max_iter} steps")

    fL, _ = pressure_function(p, rhoL, pL, aL, gamma)
    fR, _ = pressure_function(p, rhoR, pR, aR, gamma)
    u = 0.5 * (uL + uR) + 0.5 * (fR - fL)
    return p, u


def sample(xi, WL, WR, p_star, u_star, gamma=GAMMA):
    """Sample the self-similar exact solution W(x/t) at speeds xi = x/t  (Toro §4.5).

    Left of the contact (xi <= u*):
      shock (p* > p_L): speed S_L = u_L - a_L sqrt((gamma+1)/(2gamma) p*/p_L + (gamma-1)/(2gamma));
          rho*_L = rho_L [p*/p_L + (gamma-1)/(gamma+1)] / [(gamma-1)/(gamma+1) p*/p_L + 1]
      fan (p* <= p_L): head S_HL = u_L - a_L, tail S_TL = u* - a*_L, a*_L = a_L (p*/p_L)^((gamma-1)/(2gamma));
          rho*_L = rho_L (p*/p_L)^(1/gamma)
          inside the fan:  u = 2/(gamma+1) [a_L + (gamma-1)/2 u_L + xi]
                           c = 2/(gamma+1) [a_L + (gamma-1)/2 (u_L - xi)]
                           rho = rho_L (c/a_L)^(2/(gamma-1)),  p = p_L (c/a_L)^(2gamma/(gamma-1))
    Right of the contact (xi > u*): mirror image (Toro §4.5, right-wave formulas).

    Returns W with shape (3, len(xi)).
    """
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    rhoL, uL, pL = WL
    rhoR, uR, pR = WR
    aL, aR = sound_speed(rhoL, pL, gamma), sound_speed(rhoR, pR, gamma)

    g1 = (gamma - 1.0) / (2.0 * gamma)
    g2 = (gamma + 1.0) / (2.0 * gamma)
    g5 = 2.0 / (gamma + 1.0)
    g6 = (gamma - 1.0) / (gamma + 1.0)
    g7 = 0.5 * (gamma - 1.0)

    rho = np.empty_like(xi)
    u = np.empty_like(xi)
    p = np.empty_like(xi)

    def assign(mask, r, v, q):
        rho[mask], u[mask], p[mask] = r, v, q

    left = xi <= u_star
    right = ~left

    # ---- left of the contact ----
    if p_star > pL:  # left shock
        S_L = uL - aL * np.sqrt(g2 * p_star / pL + g1)
        rho_starL = rhoL * (p_star / pL + g6) / (g6 * p_star / pL + 1.0)
        assign(left & (xi <= S_L), rhoL, uL, pL)
        assign(left & (xi > S_L), rho_starL, u_star, p_star)
    else:  # left rarefaction
        S_HL = uL - aL
        S_TL = u_star - aL * (p_star / pL) ** g1
        rho_starL = rhoL * (p_star / pL) ** (1.0 / gamma)
        assign(left & (xi <= S_HL), rhoL, uL, pL)
        assign(left & (xi > S_TL), rho_starL, u_star, p_star)
        fan = left & (xi > S_HL) & (xi <= S_TL)
        c = g5 * (aL + g7 * (uL - xi[fan]))
        assign(fan, rhoL * (c / aL) ** (2.0 / (gamma - 1.0)),
               g5 * (aL + g7 * uL + xi[fan]),
               pL * (c / aL) ** (2.0 * gamma / (gamma - 1.0)))

    # ---- right of the contact ----
    if p_star > pR:  # right shock
        S_R = uR + aR * np.sqrt(g2 * p_star / pR + g1)
        rho_starR = rhoR * (p_star / pR + g6) / (g6 * p_star / pR + 1.0)
        assign(right & (xi >= S_R), rhoR, uR, pR)
        assign(right & (xi < S_R), rho_starR, u_star, p_star)
    else:  # right rarefaction
        S_HR = uR + aR
        S_TR = u_star + aR * (p_star / pR) ** g1
        rho_starR = rhoR * (p_star / pR) ** (1.0 / gamma)
        assign(right & (xi >= S_HR), rhoR, uR, pR)
        assign(right & (xi < S_TR), rho_starR, u_star, p_star)
        fan = right & (xi >= S_TR) & (xi < S_HR)
        c = g5 * (aR - g7 * (uR - xi[fan]))
        assign(fan, rhoR * (c / aR) ** (2.0 / (gamma - 1.0)),
               g5 * (-aR + g7 * uR + xi[fan]),
               pR * (c / aR) ** (2.0 * gamma / (gamma - 1.0)))

    return np.array([rho, u, p])


def exact_solution(x, t, WL, WR, x0=0.5, gamma=GAMMA):
    """Exact solution W(x, t) of the Riemann problem with the diaphragm at x0  (Toro ch. 4).

    Evaluates sample((x - x0)/t). At t = 0, returns the initial discontinuous data.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if t <= 0.0:
        return np.where(x <= x0, np.array(WL)[:, None], np.array(WR)[:, None])
    p_star, u_star = star_state(WL, WR, gamma)
    return sample((x - x0) / t, WL, WR, p_star, u_star, gamma)
