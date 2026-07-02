"""Read a tandem *domain* (volume) VTU/PVTU into per-cell arrays.

Tandem's domain writer emits one VTK_LAGRANGE_TETRAHEDRON per mesh element,
with (p+1)(p+2)(p+3)/6 points per cell at the VTK Lagrange equidistant nodes
and the displacement field as point data `u0, u1, u2` (global Cartesian).

Unlike the fault output, the domain output carries only *displacement*; there
is no velocity and no stress. The volume bridge derives stress from the
displacement gradient (Hooke's law) and -- for a quasi-dynamic -> dynamic
handoff -- takes the bulk velocity to be zero. See `volume_bridge.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np

from .fault_reader import _resolve_vtu_pieces

DISP_NAMES = ("u0", "u1", "u2")


@dataclass
class TandemDomainSnapshot:
    """One time slice of a tandem domain (volume) output."""

    time: float
    degree: int  # polynomial degree on the tet
    points_per_cell: int  # (p+1)(p+2)(p+3)/6  (20 for p=3)
    n_cells: int
    node_points: np.ndarray  # (n_cells, ppc, 3) physical coords
    displacement: np.ndarray  # (n_cells, ppc, 3) u0,u1,u2 per node
    corner_centroids: np.ndarray  # (n_cells, 3) mean of the 4 vertex nodes


def _degree_from_points_per_cell(ppc: int) -> int:
    # ppc = (p+1)(p+2)(p+3)/6 ; invert by small search
    p = 0
    while (p + 1) * (p + 2) * (p + 3) // 6 < ppc:
        p += 1
    if (p + 1) * (p + 2) * (p + 3) // 6 != ppc:
        raise ValueError(
            f"points_per_cell={ppc} is not (p+1)(p+2)(p+3)/6 for integer p"
        )
    return p


def read_tandem_domain(path: str | Path) -> TandemDomainSnapshot:
    """Load a tandem domain .pvtu (single piece) or .vtu into per-cell arrays."""
    path = Path(path)
    pieces = _resolve_vtu_pieces(path)
    if len(pieces) > 1:
        raise NotImplementedError(
            f"PVTU with {len(pieces)} pieces is not yet supported by this reader"
        )
    m = meshio.read(str(pieces[0]), file_format="vtu")

    if len(m.cells) != 1 or m.cells[0].type != "VTK_LAGRANGE_TETRAHEDRON":
        raise RuntimeError(
            f"Expected a single VTK_LAGRANGE_TETRAHEDRON cell block, got "
            f"{[(c.type, c.data.shape) for c in m.cells]}"
        )
    conn = m.cells[0].data  # (n_cells, ppc)
    n_cells, ppc = conn.shape
    deg = _degree_from_points_per_cell(ppc)

    missing = [n for n in DISP_NAMES if n not in m.point_data]
    if missing:
        raise RuntimeError(f"domain VTU is missing displacement fields: {missing}")
    disp_flat = np.column_stack(
        [np.asarray(m.point_data[n]).reshape(-1) for n in DISP_NAMES]
    )

    node_points = m.points[conn]  # (n_cells, ppc, 3)
    displacement = disp_flat[conn]  # (n_cells, ppc, 3)
    # VTK_LAGRANGE_TETRAHEDRON: the first 4 nodes are the cell corners.
    corner_centroids = node_points[:, :4, :].mean(axis=1)

    if "time" not in m.field_data:
        raise RuntimeError("domain VTU does not carry a `time` field")
    time = float(np.asarray(m.field_data["time"]).reshape(-1)[0])

    return TandemDomainSnapshot(
        time=time,
        degree=deg,
        points_per_cell=ppc,
        n_cells=n_cells,
        node_points=node_points,
        displacement=displacement,
        corner_centroids=corner_centroids,
    )
