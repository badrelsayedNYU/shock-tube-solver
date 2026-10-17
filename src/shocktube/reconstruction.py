"""MUSCL reconstruction of primitive variables with slope limiters (spec §3.3; Toro ch. 13-14)."""

import numpy as np


def minmod(r):
    """phi(r) = max(0, min(1, r))  (Toro §14.3.1)."""
    return np.maximum(0.0, np.minimum(1.0, r))


def van_leer(r):
    """phi(r) = (r + |r|)/(1 + |r|)  (Toro §14.3.1)."""
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def superbee(r):
    """phi(r) = max(0, min(2r, 1), min(r, 2))  (Toro §14.3.1)."""
    return np.maximum.reduce([np.zeros_like(r), np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0)])


def no_limiter(r):
    """phi = 0: zero slopes, i.e. first-order piecewise-constant reconstruction."""
    return np.zeros_like(r)


LIMITERS = {
    "none": no_limiter,
    "minmod": minmod,
    "vanleer": van_leer,
    "superbee": superbee,
}


def get_limiter(name):
    """Look up a limiter function phi(r) by name."""
    try:
        return LIMITERS[name]
    except KeyError:
        raise ValueError(f"unknown limiter {name!r}; choose from {sorted(LIMITERS)}") from None


def limited_slopes(W, phi):
    """Limited slopes Delta_i = phi(r_i) (W_i - W_{i-1}),  r_i = (W_{i+1} - W_i)/(W_i - W_{i-1})  (spec §3.3).

    W has shape (3, M); slopes are returned for the interior cells 1..M-2, shape (3, M-2).
    Divide-by-zero guard: where W_i - W_{i-1} = 0 the slope is phi * 0 = 0 anyway, so r
    is set to 0 there.
    """
    d_minus = W[:, 1:-1] - W[:, :-2]
    d_plus = W[:, 2:] - W[:, 1:-1]
    nonzero = d_minus != 0.0
    r = np.where(nonzero, d_plus / np.where(nonzero, d_minus, 1.0), 0.0)
    return phi(r) * d_minus


def positivity_fallback(W, slopes):
    """Zero all slopes of any cell whose reconstructed rho or p is <= 0 at either face  (spec §3.3).

    W, slopes: shape (3, M) for the same cells. Returns (slopes, n_fallback_cells).
    """
    rho_faces = (W[0] - 0.5 * slopes[0], W[0] + 0.5 * slopes[0])
    p_faces = (W[2] - 0.5 * slopes[2], W[2] + 0.5 * slopes[2])
    bad = ((rho_faces[0] <= 0) | (rho_faces[1] <= 0) | (p_faces[0] <= 0) | (p_faces[1] <= 0))
    return np.where(bad, 0.0, slopes), int(np.count_nonzero(bad))


def muscl_faces(W, phi):
    """Face states for every interface between consecutive interior cells of W  (Toro §13.4).

    W has shape (3, M) including ghost cells. Cells 1..M-2 get limited slopes Delta_i and
    face values W_i -/+ Delta_i/2. Returns (WL, WR, n_fallback) where WL[:, k], WR[:, k]
    are the left/right states at the face between cells k+1 and k+2 (M-3 faces).
    """
    slopes = limited_slopes(W, phi)
    slopes, n_fallback = positivity_fallback(W[:, 1:-1], slopes)
    W_minus = W[:, 1:-1] - 0.5 * slopes  # value at each cell's left face
    W_plus = W[:, 1:-1] + 0.5 * slopes   # value at each cell's right face
    return W_plus[:, :-1], W_minus[:, 1:], n_fallback
