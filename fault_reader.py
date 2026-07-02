"""Read a tandem fault VTU/PVTU into per-cell arrays.

Tandem writes one VTK_LAGRANGE_TRIANGLE cell per fault face, with
(p+1)(p+2)/2 points per cell at the VTK Lagrange equidistant nodes.
Connectivity is a plain iota, so each cell owns its own block of points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import meshio
import numpy as np

FIELD_NAMES = (
    "state",
    "slip0",
    "slip1",
    "traction0",
    "traction1",
    "slip-rate0",
    "slip-rate1",
    "normal-stress",
)


@dataclass
class TandemFaultSnapshot:
    """One time slice of a tandem fault output."""

    time: float
    degree: int  # polynomial degree on the fault
    points_per_cell: int  # (deg+1)(deg+2)/2
    n_cells: int  # number of fault triangles
    points: np.ndarray  # (n_cells, points_per_cell, 3)
    fields: dict[str, np.ndarray]  # name -> (n_cells, points_per_cell)


def _resolve_vtu_pieces(path: Path) -> list[Path]:
    """Return the list of .vtu pieces for a .pvtu, or [path] for a .vtu."""
    if path.suffix == ".vtu":
        return [path]
    if path.suffix != ".pvtu":
        raise ValueError(f"Unexpected suffix {path.suffix}: expected .vtu or .pvtu")
    tree = ET.parse(path)
    root = tree.getroot()
    pieces = []
    for piece in root.iter("Piece"):
        src = piece.get("Source")
        if src is None:
            continue
        pieces.append((path.parent / src).resolve())
    if not pieces:
        raise RuntimeError(f"No <Piece Source=...> entries in {path}")
    return pieces


def _degree_from_points_per_cell(ppc: int) -> int:
    p_float = (-3.0 + np.sqrt(1.0 + 8.0 * ppc)) / 2.0
    p = int(round(p_float))
    if (p + 1) * (p + 2) // 2 != ppc:
        raise ValueError(
            f"points_per_cell={ppc} does not match (p+1)(p+2)/2 for integer p"
        )
    return p


def read_tandem_fault(path: str | Path) -> TandemFaultSnapshot:
    """Load a tandem fault .pvtu (or .vtu) and reshape into per-cell arrays."""
    path = Path(path)
    pieces = _resolve_vtu_pieces(path)
    if len(pieces) > 1:
        raise NotImplementedError(
            f"PVTU with {len(pieces)} pieces is not yet supported by this reader"
        )
    m = meshio.read(str(pieces[0]), file_format="vtu")

    if len(m.cells) != 1 or m.cells[0].type != "VTK_LAGRANGE_TRIANGLE":
        raise RuntimeError(
            f"Expected a single VTK_LAGRANGE_TRIANGLE cell block, got "
            f"{[(c.type, c.data.shape) for c in m.cells]}"
        )
    conn = m.cells[0].data
    n_cells, ppc = conn.shape
    deg = _degree_from_points_per_cell(ppc)

    expected = np.arange(n_cells * ppc, dtype=conn.dtype).reshape(n_cells, ppc)
    if np.array_equal(conn, expected):
        points = m.points.reshape(n_cells, ppc, 3)
    else:
        points = m.points[conn]

    missing = [n for n in FIELD_NAMES if n not in m.point_data]
    if missing:
        raise RuntimeError(f"VTU is missing expected fault fields: {missing}")
    fields: dict[str, np.ndarray] = {}
    for name in FIELD_NAMES:
        raw = m.point_data[name].reshape(-1)
        if raw.size != n_cells * ppc:
            raise RuntimeError(
                f"field {name!r} has {raw.size} values, expected {n_cells * ppc}"
            )
        fields[name] = raw.reshape(n_cells, ppc)

    if "time" not in m.field_data:
        raise RuntimeError("VTU does not carry a `time` field")
    time = float(np.asarray(m.field_data["time"]).reshape(-1)[0])

    return TandemFaultSnapshot(
        time=time,
        degree=deg,
        points_per_cell=ppc,
        n_cells=n_cells,
        points=points,
        fields=fields,
    )
