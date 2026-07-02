"""Build a nodal projection matrix from one polynomial node set to another.

The bridge uses the Dubiner (PKDO) orthonormal basis as the modal intermediary:

    P = V_target @ inv(V_source)

where V_source is the (n_modes x n_modes) Vandermonde at source nodes and
V_target is the (n_target x n_modes) Vandermonde at target nodes.  Applied:

    per cell:    target_values = P @ source_values         (n_target,) = (n_target, n_modes) @ (n_modes,)
    all cells:   target_values = source_values @ P.T       (n_cells, n_target)

The source node count must equal n_modes = (degree+1)*(degree+2)//2 so that
V_source is square and invertible.

This module is independent of SeisSol: it takes numpy arrays only.  The CLI
(bridge.py) loads the SeisSol-specific Stroud points and passes them in.
"""

from __future__ import annotations

import numpy as np

from .dubiner import build_projection_matrix as _dubiner_proj

_SUPPORTED_BASES = {"dubiner"}


def create_basis_map(
    degree: int,
    source_nodes: np.ndarray,
    target_nodes: np.ndarray,
    basis: str = "dubiner",
) -> np.ndarray:
    """Return P of shape (n_target, n_modes) such that target = P @ source.

    Parameters
    ----------
    degree       : polynomial degree p; n_modes = (p+1)(p+2)/2.
    source_nodes : (n_modes, 2) node coordinates on the unit right triangle
                   with vertices (0,0), (1,0), (0,1).  The count must exactly
                   equal n_modes to keep V_source square (= invertible).
    target_nodes : (n_target, 2) any evaluation points on the same triangle
                   (e.g. SeisSol Stroud quadrature, or Gauss points).
    basis        : modal polynomial basis used as the intermediary.
                   Only "dubiner" is supported currently.

    Notes
    -----
    The actual checkpoint slot is (n_pad,) where n_pad >= n_target; the caller
    is responsible for zero-padding.  This function is pure mathematics.
    """
    if basis not in _SUPPORTED_BASES:
        raise ValueError(
            f"basis {basis!r} not supported; choose from {_SUPPORTED_BASES}"
        )
    n_modes = (degree + 1) * (degree + 2) // 2
    if source_nodes.shape[0] != n_modes:
        raise ValueError(
            f"source_nodes has {source_nodes.shape[0]} rows but degree={degree} "
            f"requires n_modes={n_modes} source points to make V_source square."
        )
    return _dubiner_proj(degree, source_nodes, target_nodes)
