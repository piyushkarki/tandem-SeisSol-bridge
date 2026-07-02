"""Fault pipeline: TandemBridge (reusable class) + CLI (main).

TandemBridge
------------
Converts a TandemFaultSnapshot into SeisSol-ready BridgedFields. Takes numpy
arrays only -- no file paths, no SeisSol repo root, no hardcoded index
offsets. Those live in this file's CLI (main, below) and in patcher.py (the
HDF5 writer). The class itself can be applied to any simulator pair that
shares the same polynomial order and fault geometry.

    from tandem_bridge.dubiner import vtk_equidistant_triangle
    from tandem_bridge.rotation import build_per_face_transforms
    from tandem_bridge.fault_bridge import TandemBridge, TandemBridgeConfig

    vtk_pts    = vtk_equidistant_triangle(degree)   # source nodes from VTU
    target_pts = <load your quadrature points>       # any (n_q, 2) on the triangle
    xforms     = build_per_face_transforms(faces, ref_normal, up)

    tb = TandemBridge(degree, vtk_pts, target_pts, xforms)
    bridged = tb.convert(snap, stress_scale=1e6, slip_scale=1.0)

TandemBridgeConfig carries SeisSol checkpoint layout constants (stress-tensor
slot indices). All defaults match SeisSol's fault-CS convention.

CLI
---
Fault pipeline only: tandem fault VTU + SeisSol mesh + checkpoint -> patched
checkpoint, patching `/checkpoint/dynrup`. For the domain (volume) pipeline,
see domain_bridge.py. To run both together against one output file, see the
top-level bridge.py orchestrator.

    python -m tandem_bridge.fault_bridge \
        --vtu        /path/to/fault_full_7.pvtu \
        --mesh       /path/to/seissol_mesh.puml.h5 \
        --checkpoint /path/to/bp7-checkpoint-0.h5 \
        --output     /path/to/bp7-restart-from-tandem.h5 \
        --order      4 \
        --ref-normal 0,-1,0 \
        --up         0,0,1 \
        [--dry-run]                   # print report, do not write output
        [--tol 1e-4]                  # max centroid distance for face match
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from .basis_map import create_basis_map
from .dubiner import load_seissol_stroud_points, vtk_equidistant_triangle
from .face_match import filter_to_checkpoint_ids, match_tandem_to_seissol
from .field_scalar import (
    convert_psi_to_theta,
    project_scalar,
)  # noqa: F401  (re-exported)
from .field_tensor import project_rotate_stress
from .field_vector import project_rotate_vector
from .patcher import BridgedFields, patch_checkpoint
from .rotation import PerFaceTransform, build_per_face_transforms
from .seissol_mesh_reader import read_checkpoint_dynrup_ids, read_seissol_fault_faces
from .tandem_fault_reader import TandemFaultSnapshot, read_tandem_fault

# ---------------------------------------------------------------------------
# TandemBridge: reusable, numpy-only fault-state converter
# ---------------------------------------------------------------------------


@dataclass
class TandemBridgeConfig:
    """SeisSol checkpoint layout parameters.

    sidx_* fix the fault-CS stress tensor index convention (order-independent).
    """

    sidx_tnn: int = 0  # initialStressInFaultCS[0] = sigma_nn
    sidx_t1: int = 3  # initialStressInFaultCS[3] = shear along t1
    sidx_t2: int = 5  # initialStressInFaultCS[5] = shear along t2
    n_stress_components: int = 6  # total components in fault-CS stress tensor


class TandemBridge:
    """Convert a TandemFaultSnapshot into SeisSol-ready BridgedFields.

    Build once per geometry (projection matrix + per-face transforms); call
    convert() for every snapshot in the time-step loop.

    Parameters
    ----------
    degree        : polynomial degree of the tandem fault output (= order - 1).
    source_nodes  : (n_modes, 2)  VTK equidistant Lagrange nodes on the unit
                    right triangle — these are the nodes tandem writes to VTU.
    target_nodes  : (n_target, 2) any quadrature on the unit right triangle
                    where SeisSol evaluates fault fields (e.g. Stroud-25).
    per_face_transforms : one PerFaceTransform per fault face, in the same
                    order as tandem's cells after centroid matching.
    config        : checkpoint layout; defaults match SeisSol's fault-CS convention.
    """

    def __init__(
        self,
        degree: int,
        source_nodes: np.ndarray,
        target_nodes: np.ndarray,
        per_face_transforms: list[PerFaceTransform],
        config: TandemBridgeConfig | None = None,
    ) -> None:
        if not per_face_transforms:
            raise ValueError("per_face_transforms must be non-empty")
        self.degree = degree
        self.n_cells = len(per_face_transforms)
        self.P = create_basis_map(degree, source_nodes, target_nodes)
        self.xforms = per_face_transforms
        self.cfg = config or TandemBridgeConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        snap: TandemFaultSnapshot,
        *,
        stress_scale: float = 1.0,
        slip_scale: float = 1.0,
        slip_rate_scale: float = 1.0,
    ) -> BridgedFields:
        """Project, rotate, and scale all tandem fault fields to SeisSol format.

        Parameters
        ----------
        snap             : one time-slice from read_tandem_fault().
        stress_scale     : multiply all traction/stress values (e.g. 1e6 for MPa→Pa).
        slip_scale       : multiply slip values (e.g. 1e-3 for mm→m).
        slip_rate_scale  : multiply slip-rate values.

        Returns
        -------
        BridgedFields with seissol_face_ids set to zeros; the caller fills in
        the centroid-matched face IDs before calling patcher.patch_checkpoint.

        State variable
        --------------
        state_variable is returned as raw tandem ψ (dimensionless).  Convert to
        SeisSol's dimensional θ [s] with field_scalar.convert_psi_to_theta if
        you want a physically consistent restart.
        """
        if snap.n_cells != self.n_cells:
            raise ValueError(
                f"snapshot has {snap.n_cells} cells but bridge was built for "
                f"{self.n_cells} faces — did you pass the wrong snapshot?"
            )
        f = snap.fields

        # Scalar field: state variable (ψ; caller converts to θ if needed)
        state = project_scalar(f["state"], self.P)

        # Vector fields: slip and slip-rate
        slip1, slip2 = project_rotate_vector(
            f["slip0"],
            f["slip1"],
            self.P,
            self.xforms,
            flip_with_plus_side=True,
            scale=slip_scale,
        )
        slip_rate1, slip_rate2 = project_rotate_vector(
            f["slip-rate0"],
            f["slip-rate1"],
            self.P,
            self.xforms,
            flip_with_plus_side=True,
            scale=slip_rate_scale,
        )

        # Tensor: fault-CS tractions (normal + shear)
        Tnn, T1, T2 = project_rotate_stress(
            f["traction0"],
            f["traction1"],
            f["normal-stress"],
            self.P,
            self.xforms,
            stress_scale,
        )

        # Accumulated slip magnitude (derived from projected/rotated slip)
        accumulated = np.hypot(slip1, slip2)

        return BridgedFields(
            seissol_face_ids=np.zeros(self.n_cells, dtype=np.uint64),
            slip1=slip1,
            slip2=slip2,
            slip_rate1=slip_rate1,
            slip_rate2=slip_rate2,
            state_variable=state,
            accumulated_slip_magnitude=accumulated,
            Tnn=Tnn,
            T1=T1,
            T2=T2,
            time=snap.time,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_vec3(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected 3 comma-separated floats, got {s!r}"
        )
    return np.array([float(p) for p in parts], dtype=float)


def bridge_fault(
    *,
    vtu: str,
    mesh: str,
    checkpoint: str,
    output: str,
    order: int = 4,
    ref_normal: np.ndarray = np.array([0.0, -1.0, 0.0]),
    up: np.ndarray = np.array([0.0, 0.0, 1.0]),
    tol: float = 1e-4,
    coord_scale: float = 1.0,
    stress_scale: float = 1.0,
    slip_scale: float = 1.0,
    slip_rate_scale: float = 1.0,
    convert_state: bool = False,
    copy_state_raw: bool = False,
    rs_L: float = 0.00053,
    rs_V0: float = 1e-6,
    rs_f0: float = 0.6,
    rs_b: float = 0.01,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, int] | None:
    """Run the fault pipeline end-to-end; patches `/checkpoint/dynrup`.

    Returns the patch_checkpoint report dict, or None if dry_run.
    """
    print(f"[1/6] Reading tandem fault VTU: {vtu}")
    snap = read_tandem_fault(vtu)
    expected_deg = order - 1
    if snap.degree != expected_deg:
        raise RuntimeError(
            f"tandem VTU polynomial degree {snap.degree} != "
            f"expected {expected_deg} for ConvergenceOrder={order}"
        )
    print(
        f"       time={snap.time:.6g}  degree={snap.degree}  "
        f"cells={snap.n_cells}  ppc={snap.points_per_cell}"
    )

    print(f"[2/6] Reading SeisSol mesh: {mesh}")
    all_faces = read_seissol_fault_faces(mesh)
    print(f"       fault-face incidences: {len(all_faces)} (expect 2 x N_unique)")
    ckp_ids = read_checkpoint_dynrup_ids(checkpoint)
    print(f"       checkpoint __ids: {len(ckp_ids)} unique fault faces")
    sei_faces = filter_to_checkpoint_ids(all_faces, ckp_ids)
    print(f"       filtered to checkpoint: {len(sei_faces)} faces")

    print(
        f"[3/6] Matching tandem cells to SeisSol faces by centroid "
        f"(coord_scale={coord_scale})"
    )
    tandem_centroids = snap.points.mean(axis=1) * coord_scale
    match = match_tandem_to_seissol(tandem_centroids, sei_faces, tol=tol)
    print(
        f"       max_distance={match.max_distance:.3e}  "
        f"mean_distance={match.mean_distance:.3e}"
    )

    print(f"[4/6] Building basis map (deg={snap.degree})")
    vtk_pts = vtk_equidistant_triangle(snap.degree)
    stroud_pts = load_seissol_stroud_points(order)
    print(f"       stroud_pts={stroud_pts.shape}  vtk_pts={vtk_pts.shape}")

    print(
        f"[5/6] Per-face basis transforms  "
        f"(ref_normal={np.asarray(ref_normal).tolist()}, up={np.asarray(up).tolist()})"
    )
    xforms = build_per_face_transforms(match.seissol_faces, ref_normal, up)
    n_flipped = sum(1 for x in xforms if x.sign_match < 0)
    print(f"       sign-flipped faces: {n_flipped} / {len(xforms)}")

    tb = TandemBridge(snap.degree, vtk_pts, stroud_pts, xforms)
    bridged = tb.convert(
        snap,
        stress_scale=stress_scale,
        slip_scale=slip_scale,
        slip_rate_scale=slip_rate_scale,
    )
    bridged.seissol_face_ids = match.seissol_face_ids

    # quick sanity prints (before optional state-variable suppression)
    print(f"       sample ranges in SeisSol frame (first cell only):")
    print(
        f"         slip1            {bridged.slip1[0].min():+.3e} .. {bridged.slip1[0].max():+.3e}"
    )
    print(
        f"         slip2            {bridged.slip2[0].min():+.3e} .. {bridged.slip2[0].max():+.3e}"
    )
    print(
        f"         slipRate1        {bridged.slip_rate1[0].min():+.3e} .. {bridged.slip_rate1[0].max():+.3e}"
    )
    print(
        f"         slipRate2        {bridged.slip_rate2[0].min():+.3e} .. {bridged.slip_rate2[0].max():+.3e}"
    )
    print(
        f"         stateVariable    {bridged.state_variable[0].min():+.4g} .. {bridged.state_variable[0].max():+.4g}"
    )
    print(
        f"         Tnn (norm.stress){bridged.Tnn[0].min():+.4g} .. {bridged.Tnn[0].max():+.4g}"
    )
    print(
        f"         T1 (shear)       {bridged.T1[0].min():+.4g} .. {bridged.T1[0].max():+.4g}"
    )
    print(
        f"         T2 (shear)       {bridged.T2[0].min():+.4g} .. {bridged.T2[0].max():+.4g}"
    )

    if convert_state:
        psi = bridged.state_variable
        theta = convert_psi_to_theta(psi, rs_L, rs_V0, rs_f0, rs_b)
        bridged.state_variable = theta
        print(
            f"       stateVariable: converted psi -> theta  "
            f"(L={rs_L}, V0={rs_V0}, f0={rs_f0}, b={rs_b})"
        )
        print(f"         psi  range: {psi.min():.4g} .. {psi.max():.4g}")
        print(f"         theta range: {theta.min():.4g} .. {theta.max():.4g} s")
    elif copy_state_raw:
        print(
            "       NOTE: writing raw tandem psi as stateVariable "
            "(NOT physically consistent — for debugging only)."
        )
    else:
        print(
            "       NOTE: stateVariable will NOT be patched (left at original-checkpoint values)."
        )
        print(
            "             Pass --convert-state to convert psi->theta using RS parameters,"
        )
        print("             or --copy-state-raw to write psi as-is (debugging only).")
        bridged.state_variable = None

    if dry_run:
        print("[6/6] dry-run; not writing output")
        return None

    print(f"[6/6] Patching dynrup -> {output}")
    report = patch_checkpoint(
        checkpoint, output, bridged, order=order, overwrite_existing=overwrite
    )
    print(
        f"       rows_patched={report['rows_patched']}  "
        f"rows_missing_in_ckp={report['rows_missing_in_ckp']}"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tandem_bridge.fault_bridge", description=__doc__)
    p.add_argument("--vtu", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--order", type=int, default=4, help="SeisSol convergence order (default: 4)"
    )
    p.add_argument(
        "--ref-normal",
        type=parse_vec3,
        default="0,-1,0",
        help="tandem ref_normal (default: 0,-1,0)",
    )
    p.add_argument(
        "--up",
        type=parse_vec3,
        default="0,0,1",
        help="tandem up vector (default: 0,0,1)",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="max centroid distance for face match, in SeisSol units (default: 1e-4)",
    )
    p.add_argument(
        "--coord-scale",
        type=float,
        default=1.0,
        help="multiply tandem coordinates by this factor before matching "
        "(common: 1000 if tandem uses km and SeisSol uses m)",
    )
    p.add_argument(
        "--stress-scale",
        type=float,
        default=1.0,
        help="multiply tandem stress/traction by this factor "
        "(common: 1e6 if tandem uses MPa and SeisSol uses Pa)",
    )
    p.add_argument(
        "--slip-scale",
        type=float,
        default=1.0,
        help="multiply tandem slip by this factor "
        "(common: 1 if both use m, 1e-3 if tandem uses mm vs SeisSol m)",
    )
    p.add_argument(
        "--slip-rate-scale",
        type=float,
        default=1.0,
        help="multiply tandem slip-rate by this factor",
    )
    p.add_argument(
        "--copy-state-raw",
        action="store_true",
        help="copy tandem psi straight into SeisSol stateVariable. "
        "WARNING: SeisSol stateVariable is dimensional theta (seconds), "
        "not the dimensionless psi tandem stores. Without conversion "
        "the restart will not be physically consistent. The proper "
        "conversion theta = L/V0 * exp((psi - f0)/b) requires per-point "
        "RS parameters which this bridge does not yet read.",
    )
    p.add_argument(
        "--convert-state",
        action="store_true",
        help="convert tandem psi to SeisSol theta via "
        "theta = (L/V0) * exp((psi - f0) / b). "
        "Requires --rs-L, --rs-V0, --rs-f0, --rs-b.",
    )
    p.add_argument(
        "--rs-L",
        type=float,
        default=0.00053,
        help="characteristic slip distance L [m] (default: 0.00053 for BP7)",
    )
    p.add_argument(
        "--rs-V0",
        type=float,
        default=1e-6,
        help="reference slip rate V0 [m/s] (default: 1e-6 for BP7)",
    )
    p.add_argument(
        "--rs-f0",
        type=float,
        default=0.6,
        help="reference friction coefficient f0 (default: 0.6 for BP7)",
    )
    p.add_argument(
        "--rs-b",
        type=float,
        default=0.01,
        help="RS evolution parameter b (default: 0.01 for BP7)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report only; do not write the patched checkpoint",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="overwrite an existing output file"
    )
    args = p.parse_args(argv)

    bridge_fault(
        vtu=args.vtu,
        mesh=args.mesh,
        checkpoint=args.checkpoint,
        output=args.output,
        order=args.order,
        ref_normal=args.ref_normal,
        up=args.up,
        tol=args.tol,
        coord_scale=args.coord_scale,
        stress_scale=args.stress_scale,
        slip_scale=args.slip_scale,
        slip_rate_scale=args.slip_rate_scale,
        convert_state=args.convert_state,
        copy_state_raw=args.copy_state_raw,
        rs_L=args.rs_L,
        rs_V0=args.rs_V0,
        rs_f0=args.rs_f0,
        rs_b=args.rs_b,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
