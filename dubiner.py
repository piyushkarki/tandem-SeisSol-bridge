"""Dubiner orthonormal polynomial basis on the reference triangle.

SeisSol's reference triangle is the right triangle with vertices
(0,0), (1,0), (0,1). The Dubiner (PKDO) basis is built from Jacobi
polynomials after the Duffy transform (xi,eta) -> (a,b).

This module provides:
  - dubiner_vandermonde(points, degree): (n_points, n_modes) basis matrix
  - load_seissol_stroud_points(order): SeisSol's Stroud face quad pts (vendored)
  - build_projection_matrix(deg, source_pts, target_pts): target = P @ source
  - vtk_equidistant_triangle(degree): VTK Lagrange triangle node coords
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legval

# Dubiner: phi_{i,j}(xi, eta) = c_{i,j} * (1 - eta)^i * P_i^{(0,0)}(a) * P_j^{(2i+1,0)}(b)
# where a = 2*xi/(1-eta) - 1, b = 2*eta - 1, and c_{i,j} normalizes
# to ||phi||_{L2(reference triangle, weight 1)} = 1.


def _jacobi_eval(alpha: float, beta: float, n: int, x: np.ndarray) -> np.ndarray:
    """Evaluate Jacobi polynomial P_n^{(alpha, beta)}(x) by 3-term recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 0.5 * (alpha - beta + (alpha + beta + 2.0) * x)
    p_prev = np.ones_like(x)
    p_curr = 0.5 * (alpha - beta + (alpha + beta + 2.0) * x)
    for k in range(1, n):
        a1 = 2.0 * (k + 1.0) * (k + alpha + beta + 1.0) * (2.0 * k + alpha + beta)
        a2 = (2.0 * k + alpha + beta + 1.0) * (alpha * alpha - beta * beta)
        a3 = (
            (2.0 * k + alpha + beta)
            * (2.0 * k + alpha + beta + 1.0)
            * (2.0 * k + alpha + beta + 2.0)
        )
        a4 = 2.0 * (k + alpha) * (k + beta) * (2.0 * k + alpha + beta + 2.0)
        p_next = ((a2 + a3 * x) * p_curr - a4 * p_prev) / a1
        p_prev = p_curr
        p_curr = p_next
    return p_curr


def _dubiner_normalization(i: int, j: int) -> float:
    """Constant that makes phi_{i,j} L2-orthonormal on the unit right triangle.

    K&S derives ||phi||^2 = 2 / ((2i+1)(i+j+1)) on the standard triangle with
    vertices (-1,-1), (1,-1), (-1,1) (area 2). On the unit right triangle
    with vertices (0,0), (1,0), (0,1) (area 1/2), the same expression
    evaluated at the corresponding mapped points has L2 norm scaled by
    sqrt(area_ratio) = sqrt((1/2)/2) = 1/2. The normalization constant is
    therefore 2 / sqrt(||phi||^2_KS) = sqrt(2 * (2i+1)(i+j+1)).
    """
    return np.sqrt(2.0 * (2.0 * i + 1.0) * (i + j + 1.0))


def dubiner_vandermonde(points: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate the Dubiner basis at given (xi, eta) points.

    Parameters
    ----------
    points : (n_points, 2)  reference triangle coords with vertices
             (0,0), (1,0), (0,1).
    degree : polynomial degree p; modes are pairs (i,j) with i+j <= p.

    Returns
    -------
    (n_points, n_modes) matrix; mode order is the standard triangular
    enumeration ordered by total degree then by j:
        (0,0), (0,1), (1,0), (0,2), (1,1), (2,0), ...
    NOTE: we use the same ordering everywhere we call it, so the basis
    set agrees between source and target.
    """
    xi = points[:, 0]
    eta = points[:, 1]
    eps = 1e-15
    # Duffy transform; guard the singularity at eta=1
    one_minus_eta = np.maximum(1.0 - eta, eps)
    a = 2.0 * xi / one_minus_eta - 1.0
    b = 2.0 * eta - 1.0

    n_modes = (degree + 1) * (degree + 2) // 2
    V = np.empty((points.shape[0], n_modes))
    col = 0
    for total in range(degree + 1):
        for j in range(total + 1):
            i = total - j
            Pi = _jacobi_eval(0.0, 0.0, i, a)
            Pj = _jacobi_eval(2.0 * i + 1.0, 0.0, j, b)
            # (1 - b)/2 = 1 - eta  so (1-eta)^i  = ((1-b)/2)^i
            phi = Pi * Pj * (one_minus_eta**i)
            phi *= _dubiner_normalization(i, j)
            V[:, col] = phi
            col += 1
    return V


def vtk_equidistant_triangle(degree: int) -> np.ndarray:
    """VTK Lagrange triangle equidistant nodes on the unit right triangle.

    VTK ordering for an order-p Lagrange triangle:
        - corner 0, corner 1, corner 2
        - (p-1) points along edge 0->1, edge 1->2, edge 2->0
        - then interior, recursively (one shrunken triangle)
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")
    v = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    pts: list[np.ndarray] = []
    _vtk_tri_recurse(degree, v, pts)
    return np.array(pts)


def _vtk_tri_recurse(p: int, v: np.ndarray, out: list[np.ndarray]) -> None:
    if p == 0:
        # degenerate "triangle" of degree 0 -> a single centroid point
        out.append(v.mean(axis=0))
        return
    # corners
    for k in range(3):
        out.append(v[k].copy())
    if p == 1:
        return
    # edges 0->1, 1->2, 2->0
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        for k in range(1, p):
            t = k / p
            out.append((1.0 - t) * v[a] + t * v[b])
    if p < 3:
        return
    # shrink to a similar inner triangle and recurse with degree (p - 3)
    centroid = v.mean(axis=0)
    factor = (p - 3) / p
    inner = centroid + factor * (v - centroid)
    _vtk_tri_recurse(p - 3, inner, out)


def build_projection_matrix(
    degree: int,
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """target_values = P @ source_values, with P = V_t @ V_s^{-1}.

    Both source_points and target_points are in (xi, eta) on the unit
    right triangle. The basis is Dubiner of the given degree.

    Requires len(source_points) == n_modes so V_s is square and invertible.
    """
    n_modes = (degree + 1) * (degree + 2) // 2
    if source_points.shape[0] != n_modes:
        raise ValueError(
            f"source_points ({source_points.shape[0]}) must equal n_modes ({n_modes})"
        )
    V_s = dubiner_vandermonde(source_points, degree)
    V_t = dubiner_vandermonde(target_points, degree)
    return V_t @ np.linalg.inv(V_s)


_STROUD_DATA_PATH = Path(__file__).resolve().parent / "data" / "dr_stroud_points.json"


def load_seissol_stroud_points(order: int) -> np.ndarray:
    """Return SeisSol's (chi, tau) Stroud face quadrature points for `order`.

    `order` is SeisSol's ConvergenceOrder (= polynomial degree + 1). The
    points are in (chi, tau) on the unit right triangle = (xi, eta).

    Vendored from `codegen/matrices/dr_stroud_matrices_<order>.json`'s
    `quadpoints` entry (extracted once; see `data/dr_stroud_points.json`),
    so this has no dependency on a SeisSol repo checkout at runtime.
    Available for orders 2-8 (no source data exists for higher orders).
    """
    with _STROUD_DATA_PATH.open() as f:
        table = json.load(f)
    key = str(order)
    if key not in table:
        raise ValueError(
            f"no vendored Stroud points for order {order}; available orders: "
            f"{sorted(int(k) for k in table)}"
        )
    return np.array(table[key], dtype=float)
