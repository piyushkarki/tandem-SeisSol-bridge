"""Project and rotate the fault stress tensor into SeisSol's fault-CS layout.

SeisSol stores initialStressInFaultCS as a (6, n_pad) array per face.  For a
traction-mode initialisation only three of the six slots carry data:

    index 0  (σ_nn)  : normal traction,  compressive-negative
    index 3  (T1)    : shear along tangent1
    index 5  (T2)    : shear along tangent2

tandem provides three scalar fields per cell per node:
    normal-stress  : +compressive scalar
    traction0      : shear component along d (vector-like, sign_match-sensitive)
    traction1      : shear component along s (vector-like, sign_match-sensitive)

project_rotate_stress does both the nodal-to-quadrature projection (via the
projection matrix from create_basis_map) and the per-face frame rotation, and
returns (Tnn, T1, T2) ready to be written into the checkpoint by patcher.py.
"""

from __future__ import annotations

import numpy as np

from .field_scalar import normal_stress_sign_convention, project_scalar
from .field_vector import project_rotate_vector
from .rotation import PerFaceTransform


def project_rotate_stress(
    traction0: np.ndarray,
    traction1: np.ndarray,
    normal_stress: np.ndarray,
    P: np.ndarray,
    xforms: list[PerFaceTransform],
    stress_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project + rotate fault tractions to SeisSol's (Tnn, T1, T2).

    Parameters
    ----------
    traction0, traction1 : (n_cells, n_source) shear components along d, s.
    normal_stress        : (n_cells, n_source) +compressive scalar field.
    P                    : (n_target, n_source) projection matrix.
    xforms               : one PerFaceTransform per cell, in matching order.
    stress_scale         : unit conversion factor (e.g. 1e6 for MPa -> Pa).

    Returns
    -------
    (Tnn, T1, T2) each (n_cells, n_target) in SeisSol's fault-CS convention [Pa
    if stress_scale is applied].  Tnn is compressive-negative; T1 and T2 are
    shear components along SeisSol's tangent1 and tangent2 respectively.
    """
    ns_q = project_scalar(normal_stress, P)
    Tnn = normal_stress_sign_convention(ns_q) * stress_scale

    T1, T2 = project_rotate_vector(
        traction0,
        traction1,
        P,
        xforms,
        flip_with_plus_side=True,
        scale=stress_scale,
    )
    return Tnn, T1, T2
