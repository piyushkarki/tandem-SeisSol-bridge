"""Match tandem fault cells to SeisSol fault faces by centroid.

The SeisSol mesh exposes 2 incidences per fault face (one from each
adjacent tet). The checkpoint stores 552 specific (element*4+side)
ids, picking the canonical "+" side. We filter the 1104 mesh
incidences to those 552 ids, then run a KD-tree query.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .seissol_mesh_reader import FaultFace


@dataclass
class FaceMatch:
    seissol_faces: list[FaultFace]  # 1 per tandem cell, same order
    seissol_face_ids: np.ndarray  # (n_cells,) uint64
    max_distance: float
    mean_distance: float


def filter_to_checkpoint_ids(
    faces: list[FaultFace],
    checkpoint_ids: np.ndarray,
) -> list[FaultFace]:
    """Keep only the FaultFace whose face_id is in checkpoint_ids.

    Errors if any checkpoint id is missing from the mesh.
    """
    ids_set = set(int(i) for i in checkpoint_ids)
    by_id = {int(f.face_id): f for f in faces if int(f.face_id) in ids_set}
    missing = ids_set - by_id.keys()
    if missing:
        raise RuntimeError(
            f"{len(missing)} checkpoint __ids have no matching mesh face "
            f"(first 5: {sorted(missing)[:5]})"
        )
    return list(by_id.values())


def match_tandem_to_seissol(
    tandem_centroids: np.ndarray,
    seissol_faces: list[FaultFace],
    tol: float | None = None,
) -> FaceMatch:
    """KD-tree match tandem cell centroids to SeisSol face centroids.

    Parameters
    ----------
    tandem_centroids : (n_cells, 3) physical coords
    seissol_faces    : already filtered to the checkpoint's 552 entries
    tol              : max accepted distance; if None, no check (just report)
    """
    if len(seissol_faces) != tandem_centroids.shape[0]:
        raise RuntimeError(
            f"tandem cells ({tandem_centroids.shape[0]}) != seissol faces ({len(seissol_faces)})"
        )
    sei_centroids = np.array([f.centroid for f in seissol_faces])  # (n, 3)
    tree = cKDTree(sei_centroids)
    dist, idx = tree.query(tandem_centroids)
    if tol is not None and dist.max() > tol:
        raise RuntimeError(
            f"max centroid distance {dist.max():.3e} exceeds tol {tol:.3e} "
            f"(mean={dist.mean():.3e})"
        )
    # Ensure 1-to-1
    if len(set(idx.tolist())) != len(idx):
        raise RuntimeError(
            "tandem -> seissol match is not 1-to-1 (duplicate seissol indices)"
        )
    matched_faces = [seissol_faces[int(i)] for i in idx]
    matched_ids = np.array([f.face_id for f in matched_faces], dtype=np.uint64)
    return FaceMatch(
        seissol_faces=matched_faces,
        seissol_face_ids=matched_ids,
        max_distance=float(dist.max()),
        mean_distance=float(dist.mean()),
    )
