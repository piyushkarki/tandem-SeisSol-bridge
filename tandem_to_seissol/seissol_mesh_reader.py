"""Read a SeisSol PUML mesh and extract per-fault-face geometry + identifier.

Boundary tags encode 4 bytes (one per PUML side) inside an int32:
  byte j = tag of PUML side j; 0=interior, 1=free surface, 3=dynamic rupture,
           5=absorbing, 6=periodic. See SeisSol BC enums.

SeisSol's internal side indexing differs from PUML's:
  PumlFaceToSeisSol = [0, 1, 3, 2]   (PUML side j -> SeisSol side PFTS[j])

SeisSol's face-local vertex pickers (in SeisSol side indexing):
  FACE2NODES = [[0,2,1], [0,1,3], [0,3,2], [1,2,3]]
  tangent1   = v[FACE2NODES[s][1]] - v[FACE2NODES[s][0]]
  normal     = (v[1]-v[0]) x (v[2]-v[0])  (NOT normalized)
  tangent2   = normal x tangent1
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

PUML_FACE_TO_SEISSOL = np.array([0, 1, 3, 2], dtype=np.int32)

# SeisSol's MeshTools::FACE2NODES (src/Geometry/MeshTools.cpp:21)
FACE2NODES = np.array(
    [
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ],
    dtype=np.int32,
)

DR_TAG = 3
FREE_SURFACE_TAG = 1


@dataclass
class FaultFace:
    element: int  # SeisSol element index (== globalId in serial PUML)
    side: int  # SeisSol side index in {0,1,2,3}
    face_id: int  # element * 4 + side  (matches checkpoint __ids)
    vertices: np.ndarray  # (3, 3) physical coords of the 3 face vertices, SeisSol order
    centroid: np.ndarray  # (3,)
    normal: np.ndarray  # (3,) unit
    tangent1: np.ndarray  # (3,) unit
    tangent2: np.ndarray  # (3,) unit


def _decode_boundary_bytes(boundary: np.ndarray) -> np.ndarray:
    """Return (n_elements, 4) of per-PUML-side tags (uint8)."""
    if boundary.dtype not in (np.int32, np.uint32):
        raise ValueError(f"boundary dtype must be (u)int32, got {boundary.dtype}")
    return boundary.view(np.uint8).reshape(-1, 4)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("cannot normalize zero vector")
    return v / n


def _face_frame(face_vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror MeshTools::normalAndTangents but normalize.

    face_vertices: (3, 3) array of 3 vertex coords in SeisSol's FACE2NODES order.
    """
    v0, v1, v2 = face_vertices
    ab = v1 - v0
    ac = v2 - v0
    n = np.cross(ab, ac)
    n = _normalize(n)
    t1 = _normalize(ab)
    t2 = _normalize(np.cross(n, t1))
    return n, t1, t2


def read_seissol_fault_faces(
    mesh_h5: str | Path,
    fault_tag: int = DR_TAG,
) -> list[FaultFace]:
    """Iterate the mesh and return every (element, side) with the fault tag.

    For a sealed interior fault, each face appears twice (once from each
    adjacent tet). Both incidences are returned; the bridge filters to
    those whose face_id matches the checkpoint's __ids.
    """
    mesh_h5 = Path(mesh_h5)
    with h5py.File(mesh_h5, "r") as f:
        connect = f["connect"][:]  # (n_el, 4) uint64
        geometry = f["geometry"][:]  # (n_v, 3)  float64
        boundary = f["boundary"][:]  # (n_el,)   int32

    if connect.shape[1] != 4:
        raise ValueError(
            f"expected tetrahedral connectivity, got shape {connect.shape}"
        )

    tag_bytes = _decode_boundary_bytes(boundary)  # (n_el, 4) puml ordering

    faces: list[FaultFace] = []
    for el in range(connect.shape[0]):
        tet_v = connect[el]  # (4,) vertex indices
        for puml_side in range(4):
            tag = int(tag_bytes[el, puml_side])
            if tag != fault_tag:
                continue
            seissol_side = int(PUML_FACE_TO_SEISSOL[puml_side])
            picker = FACE2NODES[seissol_side]
            face_vertex_idx = tet_v[picker]  # (3,)
            verts = geometry[face_vertex_idx]  # (3, 3)
            n, t1, t2 = _face_frame(verts)
            faces.append(
                FaultFace(
                    element=el,
                    side=seissol_side,
                    face_id=el * 4 + seissol_side,
                    vertices=verts,
                    centroid=verts.mean(axis=0),
                    normal=n,
                    tangent1=t1,
                    tangent2=t2,
                )
            )
    return faces


def read_checkpoint_dynrup_ids(checkpoint_h5: str | Path) -> np.ndarray:
    """Return the dynrup __ids array from a SeisSol checkpoint."""
    with h5py.File(checkpoint_h5, "r") as f:
        return f["/checkpoint/dynrup/__ids"][:]


@dataclass
class MeshElements:
    """Volume (tetrahedral) elements of a SeisSol PUML mesh.

    `element_id` is the serial-PUML global id (== row index); it is what the
    checkpoint's `/checkpoint/lts/__ids` stores. `vertices` keeps the four
    vertices in mesh-connectivity order, which is the order SeisSol's
    reference->global tet map uses
    (`tetrahedronReferenceToGlobal`: v0->(0,0,0), v1->(1,0,0),
    v2->(0,1,0), v3->(0,0,1)).
    """

    element_id: np.ndarray  # (n_el,) uint
    vertices: np.ndarray  # (n_el, 4, 3) physical coords, mesh order
    centroid: np.ndarray  # (n_el, 3)


def read_seissol_elements(mesh_h5: str | Path) -> MeshElements:
    """Read every tetrahedron of a SeisSol PUML mesh (centroid + vertices)."""
    mesh_h5 = Path(mesh_h5)
    with h5py.File(mesh_h5, "r") as f:
        connect = f["connect"][:]  # (n_el, 4) uint64
        geometry = f["geometry"][:]  # (n_v, 3)  float64
    if connect.shape[1] != 4:
        raise ValueError(
            f"expected tetrahedral connectivity, got shape {connect.shape}"
        )
    vertices = geometry[connect]  # (n_el, 4, 3)
    return MeshElements(
        element_id=np.arange(connect.shape[0], dtype=np.uint64),
        vertices=vertices,
        centroid=vertices.mean(axis=1),
    )


def read_checkpoint_lts_ids(checkpoint_h5: str | Path) -> np.ndarray:
    """Return the lts __ids array (element global ids) from a SeisSol checkpoint."""
    with h5py.File(checkpoint_h5, "r") as f:
        return f["/checkpoint/lts/__ids"][:]
