"""Exact Riemann solver tests (spec §4; reference values from Toro Table 4.3)."""

import numpy as np
import pytest

from shocktube.cases import CASES
from shocktube.exact import sample, star_state
from shocktube.gas import GAMMA, cons_to_prim, physical_flux, prim_to_cons, sound_speed


# ---- star-state values required by the spec (§4) ----

def test_sod_star_state():
    case = CASES["sod"]
    p, u = star_state(case.WL, case.WR)
    assert p == pytest.approx(0.30313, abs=5e-6)
    assert u == pytest.approx(0.92745, abs=5e-6)


def test_toro_test3_star_state():
    case = CASES["blast"]
    p, u = star_state(case.WL, case.WR)
    assert p == pytest.approx(460.894, abs=5e-4)
    assert u == pytest.approx(19.5975, abs=5e-5)


def test_123_star_state():
    # Toro Table 4.3, Test 2: p* = 0.00189, u* = 0 (symmetric problem).
    case = CASES["123"]
    p, u = star_state(case.WL, case.WR)
    assert p == pytest.approx(0.00189, abs=5e-6)
    assert u == pytest.approx(0.0, abs=1e-12)


# ---- independent physics checks on the sampled solution ----

def test_gas_round_trip():
    W = np.array([[1.0, 0.125, 2.3], [0.0, -1.5, 3.0], [1.0, 0.1, 1000.0]])
    np.testing.assert_allclose(cons_to_prim(prim_to_cons(W)), W, rtol=1e-14, atol=1e-14)


def test_sod_shock_satisfies_rankine_hugoniot():
    """Across the right-moving shock: S [U] = [F(U)] for the sampled pre/post states."""
    case = CASES["sod"]
    p_star, u_star = star_state(case.WL, case.WR)
    rhoR, _, pR = case.WR
    aR = sound_speed(rhoR, pR)
    S = aR * np.sqrt((GAMMA + 1) / (2 * GAMMA) * p_star / pR + (GAMMA - 1) / (2 * GAMMA))
    W = sample([S - 1e-9, S + 1e-9], case.WL, case.WR, p_star, u_star)
    Ub, Ua = prim_to_cons(W[:, 0]), prim_to_cons(W[:, 1])
    np.testing.assert_allclose(S * (Ub - Ua), physical_flux(W[:, 0]) - physical_flux(W[:, 1]),
                               rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("name", ["sod", "123"])
def test_fan_is_continuous_and_isentropic(name):
    """Inside a rarefaction the solution is continuous at head/tail and p/rho^gamma is constant."""
    case = CASES[name]
    p_star, u_star = star_state(case.WL, case.WR)
    rhoL, uL, pL = case.WL
    aL = sound_speed(rhoL, pL)
    head = uL - aL
    tail = u_star - aL * (p_star / pL) ** ((GAMMA - 1) / (2 * GAMMA))
    eps = 1e-10
    W = sample([head - eps, head + eps, tail - eps, tail + eps], case.WL, case.WR, p_star, u_star)
    np.testing.assert_allclose(W[:, 0], W[:, 1], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(W[:, 2], W[:, 3], rtol=1e-7, atol=1e-9)

    xi = np.linspace(head, tail, 50)
    Wf = sample(xi, case.WL, case.WR, p_star, u_star)
    np.testing.assert_allclose(Wf[2] / Wf[0] ** GAMMA, pL / rhoL**GAMMA, rtol=1e-12)
    # C+ Riemann invariant: u + 2a/(gamma-1) = u_L + 2a_L/(gamma-1)  (the fan is left-facing)
    a = sound_speed(Wf[0], Wf[2])
    np.testing.assert_allclose(Wf[1] + 2 * a / (GAMMA - 1), uL + 2 * aL / (GAMMA - 1), rtol=1e-12)


def test_123_is_symmetric():
    case = CASES["123"]
    p_star, u_star = star_state(case.WL, case.WR)
    xi = np.linspace(-5, 5, 201)
    W = sample(xi, case.WL, case.WR, p_star, u_star)
    np.testing.assert_allclose(W[0], W[0][::-1], rtol=1e-10)
    np.testing.assert_allclose(W[1], -W[1][::-1], atol=1e-10)
    np.testing.assert_allclose(W[2], W[2][::-1], rtol=1e-10)


def test_far_field_returns_initial_states():
    case = CASES["blast"]
    p_star, u_star = star_state(case.WL, case.WR)
    W = sample([-1e3, 1e3], case.WL, case.WR, p_star, u_star)
    np.testing.assert_allclose(W[:, 0], case.WL)
    np.testing.assert_allclose(W[:, 1], case.WR)
