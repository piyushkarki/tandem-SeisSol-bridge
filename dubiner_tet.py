"""Dubiner (PKDO) modal basis on the reference tetrahedron.

This is a *faithful port* of SeisSol's basis so that the modal coefficients
this module produces are bit-for-bit the same convention SeisSol stores in
`/checkpoint/lts/dofs`. Unlike the triangle code in `dubiner.py` (which only
ever maps nodal->nodal, so any spanning basis works), the volume bridge writes
*modal coefficients*, so the basis functions themselves must match SeisSol's
exactly -- including the singularity-free Jacobi recursion and the mode order.

Source of truth:
  - `src/Numerical/Functions.cpp`  (JacobiP, SingularityFreeJacobiP,
    SingularityFreeJacobiPAndDerivatives, TetraDubinerP, gradTetraDubinerP)
  - `src/Numerical/BasisFunction.h` (SampledBasisFunctions: mode ordering)
  - `src/Numerical/Transformation.cpp` (reference tet: xyz = v0 + (v1-v0) xi
    + (v2-v0) eta + (v3-v0) zeta; vertices (0,0,0),(1,0,0),(0,1,0),(0,0,1)).

The reference tetrahedron is the unit tet {xi,eta,zeta >= 0, xi+eta+zeta <= 1}.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Jacobi machinery (ported verbatim from Functions.cpp).
# --------------------------------------------------------------------------
def _singularity_free_factors(
    m: int, a: int, b: int
) -> tuple[float, float, float, float, float]:
    """SingularityFreeJacobiPFactors -> (c1, c2, c3, c4, c5)."""
    c0 = 2.0 * m + a + b
    c1 = c0 - 1.0
    c2 = float(a * a) - float(b * b)
    c3 = c0 * (c0 - 2.0)
    c4 = 2.0 * (m + a - 1.0) * (m + b - 1.0) * c0
    c5 = 2.0 * m * (m + a + b) * (c0 - 2.0)
    return c1, c2, c3, c4, c5


def _sfjp_value(n: int, a: int, b: int, x: float, y: float) -> float:
    """SingularityFreeJacobiP(n, a, b, x, y).

    This is the homogenised Jacobi polynomial P_n^{(a,b)}(x; y): for y=1 it is
    the ordinary Jacobi polynomial in x, but it stays finite as the collapsed
    coordinate y -> 0 (the Duffy singularity), exactly as SeisSol evaluates it.
    """
    if n == 0:
        return 1.0
    pm2 = 0.0
    pm1 = 1.0
    pm = (0.5 * a - 0.5 * b) * y + (1.0 + 0.5 * (a + b)) * x
    for m in range(2, n + 1):
        pm2 = pm1
        pm1 = pm
        c1, c2, c3, c4, c5 = _singularity_free_factors(m, a, b)
        pm = (c1 * (c2 * y + c3 * x) * pm1 - c4 * y * y * pm2) / c5
    return pm


def _sfjp_and_derivatives(
    n: int, a: int, b: int, x: float, y: float
) -> tuple[float, float, float]:
    """SingularityFreeJacobiPAndDerivatives -> (P, dP/dx, dP/dy)."""
    if n == 0:
        return 1.0, 0.0, 0.0
    pm2 = ddx_pm2 = ddy_pm2 = 0.0
    pm1 = 1.0
    ddx_pm1 = ddy_pm1 = 0.0
    pm = _sfjp_value(1, a, b, x, y)
    ddx_pm = 1.0 + 0.5 * (a + b)
    ddy_pm = 0.5 * (float(a) - float(b))
    for m in range(2, n + 1):
        pm2, pm1 = pm1, pm
        ddx_pm2, ddx_pm1 = ddx_pm1, ddx_pm
        ddy_pm2, ddy_pm1 = ddy_pm1, ddy_pm
        c1, c2, c3, c4, c5 = _singularity_free_factors(m, a, b)
        pm = (c1 * (c2 * y + c3 * x) * pm1 - c4 * y * y * pm2) / c5
        ddx_pm = (
            c1 * (c3 * pm1 + (c2 * y + c3 * x) * ddx_pm1) - c4 * y * y * ddx_pm2
        ) / c5
        ddy_pm = (
            c1 * (c2 * pm1 + (c2 * y + c3 * x) * ddy_pm1)
            - c4 * (2.0 * y * pm2 + y * y * ddy_pm2)
        ) / c5
    return pm, ddx_pm, ddy_pm


# --------------------------------------------------------------------------
# Tetrahedral Dubiner basis + gradient (ported from Functions.cpp).
# --------------------------------------------------------------------------
def _tetra_dubiner_p(
    ijk: tuple[int, int, int], xez: tuple[float, float, float]
) -> float:
    i, j, k = ijk
    xi, eta, zeta = xez
    r_num = 2.0 * xi - 1.0 + eta + zeta
    s_num = 2.0 * eta - 1.0 + zeta
    t = 2.0 * zeta - 1.0
    sigmatheta = 1.0 - eta - zeta
    theta = 1.0 - zeta
    ti = _sfjp_value(i, 0, 0, r_num, sigmatheta)
    tij = _sfjp_value(j, 2 * i + 1, 0, s_num, theta)
    tijk = _sfjp_value(k, 2 * i + 2 * j + 2, 0, t, 1.0)
    return ti * tij * tijk


def _grad_tetra_dubiner_p(
    ijk: tuple[int, int, int], xez: tuple[float, float, float]
) -> tuple[float, float, float]:
    i, j, k = ijk
    xi, eta, zeta = xez
    r_num = 2.0 * xi - 1.0 + eta + zeta
    s_num = 2.0 * eta - 1.0 + zeta
    t = 2.0 * zeta - 1.0
    sigmatheta = 1.0 - eta - zeta
    theta = 1.0 - zeta
    ti = _sfjp_and_derivatives(i, 0, 0, r_num, sigmatheta)
    tij = _sfjp_and_derivatives(j, 2 * i + 1, 0, s_num, theta)
    tijk = _sfjp_and_derivatives(k, 2 * i + 2 * j + 2, 0, t, 1.0)

    def ddalpha(drnum, dsigmatheta, dsnum, dtheta, dt):
        return (
            (ti[1] * drnum + ti[2] * dsigmatheta) * tij[0] * tijk[0]
            + ti[0] * (tij[1] * dsnum + tij[2] * dtheta) * tijk[0]
            + ti[0] * tij[0] * (tijk[1] * dt)
        )

    d_dxi = ddalpha(2.0, 0.0, 0.0, 0.0, 0.0)
    d_deta = ddalpha(1.0, -1.0, 2.0, 0.0, 0.0)
    d_dzeta = ddalpha(1.0, -1.0, 1.0, -1.0, 2.0)
    return d_dxi, d_deta, d_dzeta


def tet_mode_indices(order: int) -> list[tuple[int, int, int]]:
    """The (i,j,k) of each modal basis function, in SeisSol's storage order.

    Mirrors `SampledBasisFunctions` (BasisFunction.h): degree by degree,
    then k, then j, emitting (ord-j-k, j, k). For order=4 this is the 20-mode
    sequence ending the constant mode (0,0,0) first.
    """
    out: list[tuple[int, int, int]] = []
    for ordr in range(order):
        for k in range(ordr + 1):
            for j in range(ordr - k + 1):
                out.append((ordr - j - k, j, k))
    return out


def num_modes(order: int) -> int:
    return order * (order + 1) * (order + 2) // 6


def tet_vandermonde(ref_points: np.ndarray, order: int) -> np.ndarray:
    """(n_points, n_modes) matrix V[p, m] = TetraDubinerP(mode_m, ref_points[p]).

    The scalar helpers use only elementwise arithmetic, so we evaluate each
    mode across all points at once (one pass per mode, not per point).
    """
    modes = tet_mode_indices(order)
    pts = np.asarray(ref_points, dtype=float)
    xi, eta, zeta = pts[:, 0], pts[:, 1], pts[:, 2]
    v = np.empty((pts.shape[0], len(modes)), dtype=float)
    for m, ijk in enumerate(modes):
        v[:, m] = _tetra_dubiner_p(ijk, (xi, eta, zeta))
    return v


def tet_grad_vandermonde(ref_points: np.ndarray, order: int) -> np.ndarray:
    """(n_points, n_modes, 3): reference-space gradient d phi_m / d(xi,eta,zeta)."""
    modes = tet_mode_indices(order)
    pts = np.asarray(ref_points, dtype=float)
    xi, eta, zeta = pts[:, 0], pts[:, 1], pts[:, 2]
    g = np.empty((pts.shape[0], len(modes), 3), dtype=float)
    for m, ijk in enumerate(modes):
        gx, gy, gz = _grad_tetra_dubiner_p(ijk, (xi, eta, zeta))
        g[:, m, 0] = gx
        g[:, m, 1] = gy
        g[:, m, 2] = gz
    return g
