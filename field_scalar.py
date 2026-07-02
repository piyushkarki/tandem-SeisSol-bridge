"""Project and convert scalar fault fields.

A scalar field has one value per quadrature node per cell (e.g. normal-stress,
stateVariable).  The projection maps VTK equidistant source nodes to any target
quadrature via the matrix from basis_map.create_basis_map.

This module is pure mathematics (numpy only), no I/O, no geometry.
"""

from __future__ import annotations

import numpy as np


def project_scalar(values: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Project per-cell scalar nodal values from source to target nodes.

    Parameters
    ----------
    values : (n_cells, n_source)  nodal values at source points.
    P      : (n_target, n_source) projection matrix from create_basis_map.

    Returns
    -------
    (n_cells, n_target) values at target points.
    """
    return values @ P.T


def convert_psi_to_theta(
    psi: np.ndarray,
    L: float,
    V0: float,
    f0: float,
    b: float,
) -> np.ndarray:
    """Convert tandem dimensionless state variable ψ to SeisSol dimensional θ [s].

    SeisSol uses the Dieterich aging law:
        dθ/dt = 1 - V·θ/L        →  θ_ss = L / V  at steady state.

    tandem stores:    ψ = f0 + b·ln(V0·θ/L)
    Inverse:          θ = (L/V0) · exp((ψ − f0) / b)

    Parameters
    ----------
    psi : dimensionless tandem state variable (any shape).
    L   : characteristic slip distance [m].
    V0  : reference slip rate [m/s].
    f0  : reference friction coefficient (dimensionless).
    b   : RS evolution parameter b (dimensionless).

    Returns
    -------
    θ in seconds, same shape as psi.
    """
    return (L / V0) * np.exp((psi - f0) / b)


def normal_stress_sign_convention(tandem_normal_stress: np.ndarray) -> np.ndarray:
    """Convert tandem normal-stress to SeisSol's compressive-negative convention.

    tandem stores normal-stress as σ̂_n = −σ_n + σ_n_pre (positive in
    compression).  SeisSol's initialStressInFaultCS[0] stores σ_nn with
    compressive values negative.  The conversion is a sign flip.
    """
    return -tandem_normal_stress
