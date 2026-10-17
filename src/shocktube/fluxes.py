"""Numerical flux functions F_{i+1/2} = flux(WL, WR) for the 1D Euler equations.

All fluxes take primitive face states WL, WR of shape (3, n_faces) and return the
flux, shape (3, n_faces). Extra keyword arguments (e.g. gamma) are passed through.
"""

import numpy as np

from shocktube.gas import GAMMA, physical_flux, prim_to_cons, sound_speed, total_enthalpy


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


def roe_averages(WL, WR, gamma=GAMMA):
    """Roe-averaged velocity, enthalpy and sound speed  (spec §3.2b; Toro §11.2).

    With w_K = sqrt(rho_K):
      u~ = (w_L u_L + w_R u_R)/(w_L + w_R)
      H~ = (w_L H_L + w_R H_R)/(w_L + w_R)
      a~^2 = (gamma - 1)(H~ - u~^2/2)
    """
    wL, wR = np.sqrt(WL[0]), np.sqrt(WR[0])
    HL, HR = total_enthalpy(WL, gamma), total_enthalpy(WR, gamma)
    u = (wL * WL[1] + wR * WR[1]) / (wL + wR)
    H = (wL * HL + wR * HR) / (wL + wR)
    a = np.sqrt((gamma - 1.0) * (H - 0.5 * u**2))
    return u, H, a


def harten_fix(lam, delta):
    """Harten entropy fix  (spec §3.2b; Toro §11.4.2).

    |lambda| -> (lambda^2 + delta^2)/(2 delta)  where |lambda| < delta; otherwise |lambda|.
    """
    abs_lam = np.abs(lam)
    return np.where(abs_lam < delta, (lam**2 + delta**2) / (2.0 * delta), abs_lam)


def roe(WL, WR, gamma=GAMMA, entropy_fix=True, delta_factor=0.1):
    """Roe flux with optional Harten entropy fix  (spec §3.2b; Toro ch. 11).

    Eigenvalues  lambda_1 = u~ - a~,  lambda_2 = u~,  lambda_3 = u~ + a~
    Eigenvectors r_1 = [1, u~ - a~, H~ - u~ a~],  r_2 = [1, u~, u~^2/2],
                 r_3 = [1, u~ + a~, H~ + u~ a~]
    Wave strengths, with dU = U_R - U_L = [du1, du2, du3]:
      alpha_2 = (gamma-1)/a~^2 [du1 (H~ - u~^2) + u~ du2 - du3]
      alpha_1 = 1/(2 a~) [du1 (u~ + a~) - du2 - a~ alpha_2]
      alpha_3 = du1 - (alpha_1 + alpha_2)
    F = 1/2 (F_L + F_R) - 1/2 sum_i |lambda_i| alpha_i r_i

    The Harten fix (delta = delta_factor * a~) is applied to the acoustic waves
    lambda_1, lambda_3 only, never to the contact wave lambda_2.
    """
    u, H, a = roe_averages(WL, WR, gamma)
    UL, UR = prim_to_cons(WL, gamma), prim_to_cons(WR, gamma)
    FL, FR = physical_flux(WL, gamma), physical_flux(WR, gamma)
    du1, du2, du3 = UR - UL

    alpha2 = (gamma - 1.0) / a**2 * (du1 * (H - u**2) + u * du2 - du3)
    alpha1 = 1.0 / (2.0 * a) * (du1 * (u + a) - du2 - a * alpha2)
    alpha3 = du1 - (alpha1 + alpha2)

    lam1, lam2, lam3 = u - a, u, u + a
    if entropy_fix:
        delta = delta_factor * a
        abs1, abs3 = harten_fix(lam1, delta), harten_fix(lam3, delta)
    else:
        abs1, abs3 = np.abs(lam1), np.abs(lam3)
    abs2 = np.abs(lam2)

    one = np.ones_like(u)
    r1 = np.array([one, u - a, H - u * a])
    r2 = np.array([one, u, 0.5 * u**2])
    r3 = np.array([one, u + a, H + u * a])
    dissipation = abs1 * alpha1 * r1 + abs2 * alpha2 * r2 + abs3 * alpha3 * r3
    return 0.5 * (FL + FR) - 0.5 * dissipation


def roe_nofix(WL, WR, gamma=GAMMA):
    """Roe flux with the entropy fix switched off (for the §6.5 failure demos)."""
    return roe(WL, WR, gamma, entropy_fix=False)


def hllc(WL, WR, gamma=GAMMA):
    """HLLC flux  (spec §3.2c; Toro ch. 10).

    Einfeldt wave speeds: S_L = min(u_L - a_L, u~ - a~),  S_R = max(u_R + a_R, u~ + a~)
    S* = [p_R - p_L + rho_L u_L (S_L - u_L) - rho_R u_R (S_R - u_R)]
         / [rho_L (S_L - u_L) - rho_R (S_R - u_R)]
    U*_K = rho_K (S_K - u_K)/(S_K - S*) [1, S*, E_K/rho_K + (S* - u_K)(S* + p_K/(rho_K (S_K - u_K)))]
    F*_K = F_K + S_K (U*_K - U_K)
    F = F_L if 0 <= S_L;  F*_L if S_L <= 0 <= S*;  F*_R if S* <= 0 <= S_R;  F_R if S_R <= 0.
    """
    rhoL, uL, pL = WL
    rhoR, uR, pR = WR
    aL, aR = sound_speed(rhoL, pL, gamma), sound_speed(rhoR, pR, gamma)
    u_roe, _, a_roe = roe_averages(WL, WR, gamma)
    SL = np.minimum(uL - aL, u_roe - a_roe)
    SR = np.maximum(uR + aR, u_roe + a_roe)

    S_star = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / (
        rhoL * (SL - uL) - rhoR * (SR - uR))

    UL, UR = prim_to_cons(WL, gamma), prim_to_cons(WR, gamma)
    FL, FR = physical_flux(WL, gamma), physical_flux(WR, gamma)

    def star_flux(rho, u, p, U, F, S):
        coef = rho * (S - u) / (S - S_star)
        U_star = coef * np.array([np.ones_like(rho), S_star,
                                  U[2] / rho + (S_star - u) * (S_star + p / (rho * (S - u)))])
        return F + S * (U_star - U)

    with np.errstate(divide="ignore", invalid="ignore"):  # unused branches may divide by zero
        FL_star = star_flux(rhoL, uL, pL, UL, FL, SL)
        FR_star = star_flux(rhoR, uR, pR, UR, FR, SR)
    return np.select([0.0 <= SL, S_star >= 0.0, SR >= 0.0], [FL, FL_star, FR_star], default=FR)


FLUXES = {
    "rusanov": rusanov,
    "roe": roe,
    "roe-nofix": roe_nofix,
    "hllc": hllc,
}


def get_flux(name):
    """Look up a numerical flux function by name."""
    try:
        return FLUXES[name]
    except KeyError:
        raise ValueError(f"unknown flux {name!r}; choose from {sorted(FLUXES)}") from None
