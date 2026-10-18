"""Steady supersonic flow over a circular cylinder: 2D Euler, finite volume on a body-fitted grid.

This extends the 1D solver to the blunt-body problem: a detached bow shock stands ahead of the
cylinder, with a subsonic pocket between the shock and the nose and supersonic flow around the
shoulders. The quantity of interest is the shock standoff distance on the stagnation line,
validated against Billig's correlation, and the stagnation pressure, validated against the exact
Rayleigh pitot formula.

Geometry and grid
  Free stream along +x, cylinder of radius R = 1 centred at the origin. Only the windward half
  is computed: theta in [-90 deg, +90 deg] measured from the -x axis, x = -r cos(theta),
  y = r sin(theta). Index i runs outward from the wall (r = 1) to the outer boundary, index j
  runs along theta. Radial spacing is stretched geometrically so that cells cluster at the wall.

Discretization
  Cell-centred finite volume; face vectors from node differences, so the geometric conservation
  law holds exactly (uniform flow stays uniform to round-off). MUSCL reconstruction of primitive
  variables in each index direction with the same limiters as the 1D code, rotated to the face
  normal for the flux. Fluxes: HLLC (Toro ch. 10) with Einfeldt wave speeds, or HLL (Einfeldt's
  HLLE), which has no contact wave and is immune to the carbuncle instability. SSP-RK2 in
  pseudo-time with local time steps (each cell at its own CFL limit), marched to steady state.

Boundary conditions
  wall (i = 0 side): slip wall, ghost state mirrors the normal velocity.
  outer circle: free stream (the outer radius is chosen so the whole shock lies inside).
  theta = +/-90 deg rays (the plane x = 0): zero-gradient outflow; the flow leaving there is
  supersonic, so no information enters.

Nondimensionalization: rho_inf = 1, a_inf = 1 (p_inf = 1/gamma), u_inf = M_inf, R = 1.
"""

from dataclasses import dataclass

import numpy as np

from shocktube.reconstruction import get_limiter

GAMMA = 1.4
NG = 2


# ---------------------------------------------------------------- grid

@dataclass
class Grid:
    xn: np.ndarray      # node x, shape (Ni+1, Nj+1)
    yn: np.ndarray
    xc: np.ndarray      # cell centres, shape (Ni, Nj)
    yc: np.ndarray
    area: np.ndarray    # cell areas, (Ni, Nj)
    Si: np.ndarray      # i-face vectors (normal * length, pointing to +i), shape (2, Ni+1, Nj)
    Sj: np.ndarray      # j-face vectors (pointing to +j), shape (2, Ni, Nj+1)
    r_nodes: np.ndarray
    theta_nodes: np.ndarray

    @property
    def shape(self):
        return self.area.shape


def make_grid(Ni=96, Nj=128, r_outer=4.0, stretch=1.03, theta_max_deg=90.0):
    """Polar O-grid sector around the unit cylinder.

    Radial node spacing grows geometrically by `stretch` from the wall to r_outer.
    """
    n = np.arange(Ni + 1)
    if stretch == 1.0:
        s = n / Ni
    else:
        s = (stretch**n - 1.0) / (stretch**Ni - 1.0)
    r = 1.0 + (r_outer - 1.0) * s
    th = np.deg2rad(np.linspace(-theta_max_deg, theta_max_deg, Nj + 1))
    R, TH = np.meshgrid(r, th, indexing="ij")
    xn, yn = -R * np.cos(TH), R * np.sin(TH)

    # i-faces (arcs of constant r): segment from node (i, j) to (i, j+1). Rotating the segment
    # (dx, dy) to (-dy, dx) gives the outward radial normal, i.e. toward increasing i.
    dx, dy = xn[:, 1:] - xn[:, :-1], yn[:, 1:] - yn[:, :-1]
    Si = np.array([-dy, dx])
    # j-faces (rays): segment from node (i, j) to (i+1, j). (dy, -dx) points toward increasing theta.
    dx, dy = xn[1:, :] - xn[:-1, :], yn[1:, :] - yn[:-1, :]
    Sj = np.array([dy, -dx])
    xc = 0.25 * (xn[:-1, :-1] + xn[1:, :-1] + xn[:-1, 1:] + xn[1:, 1:])
    yc = 0.25 * (yn[:-1, :-1] + yn[1:, :-1] + yn[:-1, 1:] + yn[1:, 1:])
    # area by the shoelace formula on each quadrilateral (nodes in order)
    x1, y1 = xn[:-1, :-1], yn[:-1, :-1]
    x2, y2 = xn[1:, :-1], yn[1:, :-1]
    x3, y3 = xn[1:, 1:], yn[1:, 1:]
    x4, y4 = xn[:-1, 1:], yn[:-1, 1:]
    area = 0.5 * np.abs((x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y4 - x4 * y3) + (x4 * y1 - x1 * y4))
    return Grid(xn, yn, xc, yc, area, Si, Sj, r, th)


# ---------------------------------------------------------------- gas

def prim_to_cons(W, g=GAMMA):
    rho, u, v, p = W
    return np.array([rho, rho * u, rho * v, p / (g - 1.0) + 0.5 * rho * (u * u + v * v)])


def cons_to_prim(U, g=GAMMA):
    rho = U[0]
    u, v = U[1] / rho, U[2] / rho
    return np.array([rho, u, v, (g - 1.0) * (U[3] - 0.5 * rho * (u * u + v * v))])


def freestream(M, g=GAMMA):
    """Primitive free-stream state with rho = 1, a = 1."""
    return np.array([1.0, M, 0.0, 1.0 / g])


# ---------------------------------------------------------------- fluxes (rotated to the face normal)

def _rotate(W, nx, ny):
    rho, u, v, p = W
    return rho, u * nx + v * ny, -u * ny + v * nx, p


def _normal_flux(rho, un, ut, p, g):
    E = p / (g - 1.0) + 0.5 * rho * (un * un + ut * ut)
    return np.array([rho * un, rho * un * un + p, rho * un * ut, un * (E + p)]), np.array([rho, rho * un, rho * ut, E])


def _wave_speeds(rL, uL, tL, pL, rR, uR, tR, pR, g):
    """Einfeldt estimates S_L = min(u_L - a_L, u~ - a~), S_R = max(u_R + a_R, u~ + a~)."""
    aL, aR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)
    wL, wR = np.sqrt(rL), np.sqrt(rR)
    HL = (g / (g - 1.0)) * pL / rL + 0.5 * (uL**2 + tL**2)
    HR = (g / (g - 1.0)) * pR / rR + 0.5 * (uR**2 + tR**2)
    u = (wL * uL + wR * uR) / (wL + wR)
    t = (wL * tL + wR * tR) / (wL + wR)
    H = (wL * HL + wR * HR) / (wL + wR)
    a = np.sqrt(np.maximum((g - 1.0) * (H - 0.5 * (u * u + t * t)), 1e-14))
    return np.minimum(uL - aL, u - a), np.maximum(uR + aR, u + a)


def hll(WL, WR, nx, ny, g=GAMMA):
    """HLL flux with Einfeldt wave speeds (HLLE), in the rotated frame. Returns normal-frame flux."""
    rL, uL, tL, pL = _rotate(WL, nx, ny)
    rR, uR, tR, pR = _rotate(WR, nx, ny)
    SL, SR = _wave_speeds(rL, uL, tL, pL, rR, uR, tR, pR, g)
    FL, UL = _normal_flux(rL, uL, tL, pL, g)
    FR, UR = _normal_flux(rR, uR, tR, pR, g)
    SLm, SRp = np.minimum(SL, 0.0), np.maximum(SR, 0.0)
    return (SRp * FL - SLm * FR + SLm * SRp * (UR - UL)) / (SRp - SLm)


def hllc(WL, WR, nx, ny, g=GAMMA):
    """HLLC flux (Toro ch. 10) with Einfeldt wave speeds; the tangential velocity is carried by the
    contact wave. Returns the flux in the rotated (normal, tangential) frame."""
    rL, uL, tL, pL = _rotate(WL, nx, ny)
    rR, uR, tR, pR = _rotate(WR, nx, ny)
    SL, SR = _wave_speeds(rL, uL, tL, pL, rR, uR, tR, pR, g)
    FL, UL = _normal_flux(rL, uL, tL, pL, g)
    FR, UR = _normal_flux(rR, uR, tR, pR, g)
    Ss = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (rL * (SL - uL) - rR * (SR - uR))

    def star(r, u, t, p, U, S):
        c = r * (S - u) / (S - Ss)
        return c * np.array([np.ones_like(r), Ss, t, U[3] / r + (Ss - u) * (Ss + p / (r * (S - u)))])

    with np.errstate(divide="ignore", invalid="ignore"):
        FsL = FL + SL * (star(rL, uL, tL, pL, UL, SL) - UL)
        FsR = FR + SR * (star(rR, uR, tR, pR, UR, SR) - UR)
    return np.select([SL >= 0.0, Ss >= 0.0, SR >= 0.0], [FL, FsL, FsR], default=FR)


FLUXES = {"hll": hll, "hllc": hllc}


def face_flux(flux, WL, WR, S, g=GAMMA):
    """Physical-frame flux through faces with area vectors S (2, ...): returns F . S (4, ...)."""
    length = np.hypot(S[0], S[1])
    nx, ny = S[0] / length, S[1] / length
    Fn = flux(WL, WR, nx, ny, g)
    return np.array([Fn[0], Fn[1] * nx - Fn[2] * ny, Fn[1] * ny + Fn[2] * nx, Fn[3]]) * length


# ---------------------------------------------------------------- spatial operator

def _ghosts(W, grid, Winf):
    """Pad primitive W (4, Ni, Nj) with NG ghost layers on every side."""
    Ni, Nj = grid.shape
    P = np.empty((4, Ni + 2 * NG, Nj + 2 * NG))
    P[:, NG:-NG, NG:-NG] = W
    # wall at i = 0: mirror cells i = k -> ghost -1-k, reflect velocity about the wall normal
    nx, ny = grid.Si[0, 0], grid.Si[1, 0]
    ln = np.hypot(nx, ny)
    nx, ny = nx / ln, ny / ln
    for k in range(NG):
        src = W[:, k, :]
        un = src[1] * nx + src[2] * ny
        P[0, NG - 1 - k, NG:-NG] = src[0]
        P[1, NG - 1 - k, NG:-NG] = src[1] - 2.0 * un * nx
        P[2, NG - 1 - k, NG:-NG] = src[2] - 2.0 * un * ny
        P[3, NG - 1 - k, NG:-NG] = src[3]
    # outer boundary: free stream
    P[:, -NG:, NG:-NG] = Winf[:, None, None]
    # theta = +/-90 deg: zero-gradient outflow (also fills the corners)
    P[:, :, :NG] = P[:, :, NG:NG + 1]
    P[:, :, -NG:] = P[:, :, -NG - 1:-NG]
    return P


def _muscl(P, phi, axis):
    """Left/right face states along `axis` (1 = i, 2 = j) for the faces between interior cells.
    P includes NG ghost layers; returns states at the N+1 faces of that direction."""
    sl = lambda a, b: tuple(slice(a, b) if ax == axis else slice(NG, -NG) for ax in (1, 2))
    n = P.shape[axis]
    Wm, W0, Wp = P[(slice(None),) + sl(0, n - 2)], P[(slice(None),) + sl(1, n - 1)], P[(slice(None),) + sl(2, n)]
    dm, dp = W0 - Wm, Wp - W0
    nz = dm != 0.0
    r = np.where(nz, dp / np.where(nz, dm, 1.0), 0.0)
    d = phi(r) * dm
    # positivity fallback on rho and p, as in the 1D code
    bad = (W0[0] - 0.5 * d[0] <= 0) | (W0[0] + 0.5 * d[0] <= 0) | (W0[3] - 0.5 * d[3] <= 0) | (W0[3] + 0.5 * d[3] <= 0)
    d = np.where(bad[None], 0.0, d)
    plus, minus = W0 + 0.5 * d, W0 - 0.5 * d     # right-face and left-face values of each cell
    take = lambda a, s: a[(slice(None),) + tuple(s if ax == axis else slice(None) for ax in (1, 2))]
    return take(plus, slice(0, -1)), take(minus, slice(1, None))


def residual(U, grid, Winf, flux, phi, g=GAMMA):
    """R(U) = -(1/A) sum over faces of F . S  (so dU/dt = R)."""
    W = cons_to_prim(U, g)
    P = _ghosts(W, grid, Winf)
    WLi, WRi = _muscl(P, phi, axis=1)
    WLj, WRj = _muscl(P, phi, axis=2)
    Fi = face_flux(flux, WLi, WRi, grid.Si, g)
    Fj = face_flux(flux, WLj, WRj, grid.Sj, g)
    return -((Fi[:, 1:, :] - Fi[:, :-1, :]) + (Fj[:, :, 1:] - Fj[:, :, :-1])) / grid.area


def local_dt(U, grid, cfl, g=GAMMA):
    """dt = CFL A / (spectral radius in i + spectral radius in j), face vectors averaged per cell."""
    rho, u, v, p = cons_to_prim(U, g)
    a = np.sqrt(g * p / rho)
    Sx_i, Sy_i = 0.5 * (grid.Si[0, 1:] + grid.Si[0, :-1]), 0.5 * (grid.Si[1, 1:] + grid.Si[1, :-1])
    Sx_j, Sy_j = 0.5 * (grid.Sj[0, :, 1:] + grid.Sj[0, :, :-1]), 0.5 * (grid.Sj[1, :, 1:] + grid.Sj[1, :, :-1])
    lam = np.abs(u * Sx_i + v * Sy_i) + a * np.hypot(Sx_i, Sy_i) + np.abs(u * Sx_j + v * Sy_j) + a * np.hypot(Sx_j, Sy_j)
    return cfl * grid.area / lam


# ---------------------------------------------------------------- driver

@dataclass
class BluntResult:
    grid: Grid
    W: np.ndarray            # primitive (4, Ni, Nj)
    M_inf: float
    iterations: int
    residuals: np.ndarray    # L2 norm of the density residual per iteration, normalized by the first
    converged: bool


def solve(M_inf, grid=None, flux="hll", limiter="vanleer", cfl=0.8, first_order_iters=1500,
          max_iters=12000, tol=1e-7, g=GAMMA, verbose=False):
    """March to steady state. A first-order phase establishes the shock, then MUSCL takes over.

    Convergence: L2 norm of the density residual has dropped by `tol` relative to its first value.
    """
    grid = grid or make_grid(r_outer=outer_radius(M_inf))
    Winf = freestream(M_inf, g)
    Ni, Nj = grid.shape
    U = np.broadcast_to(prim_to_cons(Winf, g)[:, None, None], (4, Ni, Nj)).copy()
    flux_fn, phi2, phi1 = FLUXES[flux], get_limiter(limiter), get_limiter("none")
    res_hist, r0 = [], None
    converged = False
    for it in range(1, max_iters + 1):
        phi = phi1 if it <= first_order_iters else phi2
        dt = local_dt(U, grid, cfl, g)
        R0 = residual(U, grid, Winf, flux_fn, phi, g)
        U1 = U + dt * R0
        U = 0.5 * U + 0.5 * (U1 + dt * residual(U1, grid, Winf, flux_fn, phi, g))
        rn = float(np.sqrt(np.mean(R0[0] ** 2)))
        if not np.isfinite(rn):
            raise FloatingPointError(f"solution diverged at iteration {it}")
        r0 = r0 or rn
        res_hist.append(rn / r0)
        if verbose and it % 500 == 0:
            print(f"  it {it:6d}  residual {rn / r0:.2e}")
        if it > first_order_iters + 200 and rn / r0 < tol:
            converged = True
            break
    return BluntResult(grid, cons_to_prim(U, g), M_inf, it, np.array(res_hist), converged)


# ---------------------------------------------------------------- reference results

def billig_standoff(M_inf):
    """Billig (1967) shock standoff for a circular cylinder: Delta/R = 0.386 exp(4.67/M^2)."""
    return 0.386 * np.exp(4.67 / M_inf**2)


def billig_shock_shape(M_inf, y):
    """Billig's hyperbolic bow-shock shape for a cylinder of radius R = 1 (x measured from the nose
    along the free stream, negative upstream). Returns x of the shock at height y."""
    beta = np.arcsin(1.0 / M_inf)
    Rc = 1.386 * np.exp(1.8 / (M_inf - 1.0) ** 0.75)
    delta = billig_standoff(M_inf)
    return -(1.0 + delta) + Rc / np.tan(beta) ** 2 * (np.sqrt(1.0 + (y * np.tan(beta) / Rc) ** 2) - 1.0)


def outer_radius(M_inf):
    """Outer boundary radius that keeps the whole bow shock (to theta = 90 deg) inside the grid."""
    x_shock = billig_shock_shape(M_inf, 1.0 * np.array([0.0, 1.0, 2.0, 3.0]))
    r_needed = np.max(np.hypot(x_shock, [0.0, 1.0, 2.0, 3.0]))
    return float(np.clip(1.35 * r_needed, 2.6, 8.0))


def pitot_pressure_ratio(M, g=GAMMA):
    """Rayleigh pitot formula: stagnation pressure behind a normal shock over free-stream static
    pressure, p02/p1 (exact for a perfect gas)."""
    a = ((g + 1.0) ** 2 * M**2 / (4.0 * g * M**2 - 2.0 * (g - 1.0))) ** (g / (g - 1.0))
    return a * (1.0 - g + 2.0 * g * M**2) / (g + 1.0)


def normal_shock_pressure_ratio(M, g=GAMMA):
    return 1.0 + 2.0 * g / (g + 1.0) * (M**2 - 1.0)


# ---------------------------------------------------------------- post-processing

def stagnation_line(res):
    """r and primitive state along theta = 0 (average of the two rays nearest the axis)."""
    Nj = res.grid.shape[1]
    j = Nj // 2
    W = 0.5 * (res.W[:, :, j - 1] + res.W[:, :, j]) if Nj % 2 == 0 else res.W[:, :, j]
    r = np.hypot(res.grid.xc[:, j], res.grid.yc[:, j]) if Nj % 2 else \
        0.5 * (np.hypot(res.grid.xc[:, j - 1], res.grid.yc[:, j - 1]) + np.hypot(res.grid.xc[:, j], res.grid.yc[:, j]))
    return r, W


def shock_standoff(res, g=GAMMA):
    """Standoff Delta/R: where the stagnation-line pressure crosses the midpoint between the
    free-stream pressure and the normal-shock pressure (searching inward from the outer boundary)."""
    r, W = stagnation_line(res)
    p = W[3]
    p_inf = 1.0 / g
    p_mid = 0.5 * (p_inf + p_inf * normal_shock_pressure_ratio(res.M_inf, g))
    for i in range(len(r) - 1, 0, -1):
        if p[i] < p_mid <= p[i - 1]:
            r_s = r[i] + (p_mid - p[i]) * (r[i - 1] - r[i]) / (p[i - 1] - p[i])
            return r_s - 1.0
    raise RuntimeError("no shock found on the stagnation line")


def stagnation_pressure(res, g=GAMMA):
    """Wall pressure at theta = 0, extrapolated linearly from the two cells nearest the wall,
    divided by the free-stream static pressure."""
    r, W = stagnation_line(res)
    p0 = W[3, 0] + (W[3, 0] - W[3, 1]) * (r[0] - 1.0) / (r[1] - r[0])
    return p0 / (1.0 / g)


def mach_field(res, g=GAMMA):
    rho, u, v, p = res.W
    return np.hypot(u, v) / np.sqrt(g * p / rho)


def shock_location(res, g=GAMMA):
    """Shock position along every ray (outermost crossing of the pressure midpoint between the
    free stream and the local maximum on the ray). Returns (x, y) arrays."""
    p = res.W[3]
    p_inf = 1.0 / g
    rr = np.hypot(res.grid.xc, res.grid.yc)
    xs, ys = [], []
    for j in range(p.shape[1]):
        col = p[:, j]
        mid = 0.5 * (p_inf + col.max())
        for i in range(len(col) - 1, 0, -1):
            if col[i] < mid <= col[i - 1]:
                f = (mid - col[i]) / (col[i - 1] - col[i])
                rs = rr[i, j] + f * (rr[i - 1, j] - rr[i, j])
                th = np.arctan2(res.grid.yc[i, j], -res.grid.xc[i, j])
                xs.append(-rs * np.cos(th))
                ys.append(rs * np.sin(th))
                break
    return np.array(xs), np.array(ys)
