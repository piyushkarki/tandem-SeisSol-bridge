"""Project and rotate 2-component in-plane fault vector fields.

In-plane vector fields (slip, slip-rate, shear traction) each have two
components: one along tandem's d-axis and one along tandem's s-axis.
The bridge must:
  1. Project both components from VTK nodes to target quadrature (scalar projection).
  2. Rotate from tandem's per-face (d, s) frame to SeisSol's per-face (t1, t2) frame.
  3. Optionally flip sign when the two codes' "+" sides disagree (sign_match = -1).

The rotation matrices are pre-built by rotation.build_per_face_transforms.
This module vectorises the per-face rotation loop over all cells at once.

Convention note
---------------
Tandem's component0 is along d (= s × n) and component1 is along s (= up × n).
SeisSol's t1 = v1 - v0 (in FACE2NODES order) and t2 = n × t1.
The (3×3) rotation R = B_seissol^T @ B_tandem maps tandem (n, d, s) to
SeisSol (n, t1, t2).  Since the fault-plane components live in the (d, s)
subspace, only the t1 and t2 rows of R are needed; the n-component of the
tandem vector is always zero and drops out.
"""

from __future__ import annotations

import numpy as np

from .field_scalar import project_scalar
from .rotation import PerFaceTransform


def project_rotate_vector(
    c0: np.ndarray,
    c1: np.ndarray,
    P: np.ndarray,
    xforms: list[PerFaceTransform],
    flip_with_plus_side: bool,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project + rotate a tandem 2-component vector field to SeisSol (t1, t2).

    Parameters
    ----------
    c0, c1             : (n_cells, n_source) tandem components along d and s.
    P                  : (n_target, n_source) projection matrix (create_basis_map).
    xforms             : one PerFaceTransform per cell, in matching order.
    flip_with_plus_side: True for slip / slip-rate / shear traction whose sign
                         depends on which side is chosen as "+".  False for
                         quantities independent of orientation.
    scale              : multiply both output components (e.g. MPa -> Pa = 1e6).

    Returns
    -------
    (t1, t2) each (n_cells, n_target) in SeisSol's fault-local frame.
    """
    c0_q = project_scalar(c0, P)  # (n_cells, n_target)
    c1_q = project_scalar(c1, P)

    n_cells, n_q = c0_q.shape

    # Stack per-face rotation matrices and sign flags.
    Rs = np.stack([xf.R for xf in xforms])  # (n_cells, 3, 3)
    signs = np.array([xf.sign_match for xf in xforms])  # (n_cells,)

    # Build tandem 3-vector in (n, d, s) coordinates.
    # n-component is always 0; d-component = c0; s-component = c1.
    zeros = np.zeros((n_cells, n_q), dtype=c0_q.dtype)
    c_t = np.stack([zeros, c0_q, c1_q], axis=-1)  # (n_cells, n_q, 3)

    # Rotate: c_s[cell, quad, i] = sum_j R[cell, i, j] * c_t[cell, quad, j]
    # Equivalent to: for each cell, c_s[q] = R @ c_t[q]
    c_s = np.einsum("cij,cqj->cqi", Rs, c_t)  # (n_cells, n_q, 3)

    t1 = c_s[..., 1].copy()  # (n_cells, n_q)  -- SeisSol tangent1 component
    t2 = c_s[..., 2].copy()  # (n_cells, n_q)  -- SeisSol tangent2 component

    # Flip sign for cells where "+" sides disagree between tandem and SeisSol.
    if flip_with_plus_side:
        flip = signs < 0  # (n_cells,) bool
        t1[flip] *= -1
        t2[flip] *= -1

    if scale != 1.0:
        t1 *= scale
        t2 *= scale

    return t1, t2
