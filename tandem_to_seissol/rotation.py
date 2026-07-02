"""Reconstruct tandem's per-face (n, t1, t2) and rotate fields into SeisSol's basis.

Tandem's facetBasis (Curvilinear.cpp:259-294, D=3):
    n = normal.normalized()
    s = (up x n).normalized()
    d = s x n
    basis columns = (n, d, s)
    tandem 'slip0/traction0/slip-rate0' -> coordinate along d
    tandem 'slip1/traction1/slip-rate1' -> coordinate along s

ref_normal sign rule (AdapterBase.cpp:67-74):
    if dot(ref_normal, geometric_normal) < 0: flip the whole (n, d, s) basis.

SeisSol per-face basis (MeshTools::normalAndTangents):
    Order = (normal, tangent1, tangent2) where tangent1 = v1 - v0 (in
    FACE2NODES order) and tangent2 = normal x tangent1. mesh_reader.FaultFace
    already provides these unit vectors.

Sign convention difference for scalar quantities:
    normal-stress: tandem stores +compressive (sn_hat), SeisSol stores
                   sigma_nn (compressive negative); seissol_Tnn = -tandem_ns.

Sign convention difference for vector quantities (slip, slip-rate, shear traction):
    A physical slip / traction vector is the same in 3D space; only the
    sign assigned to it depends on which side is "+". If tandem's normal
    is opposite SeisSol's, the "+" sides are opposite and the vector flips.
    This is captured by sign_match = sign(dot(n_tandem, n_seissol)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .seissol_mesh_reader import FaultFace


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("cannot normalize zero vector")
    return v / n


def tandem_facet_basis(
    geom_normal: np.ndarray,
    ref_normal: np.ndarray,
    up: np.ndarray,
    colinear_tol: float = 1e-10,
) -> np.ndarray:
    """Build tandem's 3x3 facet basis (columns = n, d, s) at one point.

    geom_normal: a raw outward normal of the face (any orientation; will be
                 flipped to align with ref_normal as tandem does).
    Returns (3, 3): columns 0=n, 1=d, 2=s.
    """
    n = _normalize(geom_normal)
    if np.dot(ref_normal, n) < 0:
        n = -n
    s_unnorm = np.cross(up, n)
    s_norm = np.linalg.norm(s_unnorm)
    if s_norm < colinear_tol:
        raise ValueError(
            "up and normal are nearly colinear; cannot build tandem facet basis"
        )
    s = s_unnorm / s_norm
    d = np.cross(s, n)
    d = _normalize(d)
    return np.column_stack([n, d, s])  # (3, 3)


@dataclass
class PerFaceTransform:
    """Per-face geometric data for transforming tandem fields to SeisSol."""

    R: np.ndarray  # (3, 3) maps tandem-(n,d,s) coords -> seissol-(n,t1,t2) coords
    sign_match: float  # +1 if "+" sides agree, -1 if opposite
    tandem_basis: np.ndarray  # (3, 3) tandem (n, d, s)
    seissol_basis: np.ndarray  # (3, 3) seissol (n, t1, t2)


def build_per_face_transforms(
    faces: list[FaultFace],
    ref_normal: np.ndarray,
    up: np.ndarray,
) -> list[PerFaceTransform]:
    """One PerFaceTransform per FaultFace, in input order.

    SeisSol's mesh_reader.FaultFace already supplies the SeisSol-side basis;
    we reconstruct tandem's by replaying its facetBasis logic on the
    geometric normal (which is the same SeisSol unit normal up to sign).
    """
    ref_normal = np.asarray(ref_normal, dtype=float)
    up = np.asarray(up, dtype=float)

    out: list[PerFaceTransform] = []
    for face in faces:
        # SeisSol's face.normal is a unit outward normal (chosen by SeisSol's
        # "+" side convention). Tandem will rebuild from the raw geometric
        # normal and flip if needed; both share the same span so we can use
        # face.normal directly here.
        Bt = tandem_facet_basis(face.normal, ref_normal, up)
        Bs = np.column_stack([face.normal, face.tangent1, face.tangent2])
        R = Bs.T @ Bt
        sign_match = float(np.sign(np.dot(Bs[:, 0], Bt[:, 0])))
        if sign_match == 0:
            raise RuntimeError(
                "tandem and seissol normals are perpendicular (impossible)"
            )
        out.append(
            PerFaceTransform(
                R=R, sign_match=sign_match, tandem_basis=Bt, seissol_basis=Bs
            )
        )
    return out


def rotate_tandem_vector(
    component0: np.ndarray,  # (...,) tandem 'X0' -> coord along d
    component1: np.ndarray,  # (...,) tandem 'X1' -> coord along s
    xform: PerFaceTransform,
    flip_with_plus_side: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a tandem 2-vector field (along d, s) to SeisSol (t1, t2).

    Parameters
    ----------
    component0 : tandem component along d (matches slip0 / traction0 / etc.)
    component1 : tandem component along s
    xform      : PerFaceTransform for this face
    flip_with_plus_side : True for slip / slip-rate / shear traction
                          (sign depends on '+' side choice), False otherwise.

    Returns (seissol_t1, seissol_t2), same shape as the inputs.
    """
    # tandem 3-vector (n=0, d=c0, s=c1) in (n, d, s) coords
    # apply R to get seissol coords (n', t1', t2')
    c_t = np.stack(
        [np.zeros_like(component0), component0, component1], axis=-1
    )  # (..., 3)
    c_s = c_t @ xform.R.T  # (..., 3)
    out0 = c_s[..., 1]
    out1 = c_s[..., 2]
    if flip_with_plus_side and xform.sign_match < 0:
        out0 = -out0
        out1 = -out1
    return out0, out1


def normal_stress_to_seissol(tandem_normal_stress: np.ndarray) -> np.ndarray:
    """Tandem normal-stress is +compressive (sn_hat = -sn + sn_pre).

    SeisSol's initialStressInFaultCS[0] stores sigma_nn (compressive negative).
    The relationship is symmetric in normal direction, so no sign_match here.
    """
    return -tandem_normal_stress
