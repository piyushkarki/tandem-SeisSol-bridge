"""Bridge tandem *domain* (volume) state into a SeisSol `/checkpoint/lts/dofs`.

Tandem's domain output carries only the displacement field `u = (u0,u1,u2)`.
SeisSol's volume DOFs hold modal coefficients of nine elastic quantities
`[s_xx,s_yy,s_zz,s_xy,s_yz,s_xz, v1,v2,v3]` (stress + velocity) in the Dubiner
tetrahedron basis, stored flat as `index = quantity*nbf + mode`.

So the volume bridge must

  1. fit the nodal displacement to the modal basis on each tet,
  2. differentiate it (physical gradient via the affine reference->global
     Jacobian) to get strain, and apply Hooke's law sigma = lam*tr(eps)*I +
     2*mu*eps to obtain the stress field,
  3. project that stress onto SeisSol's exact modal basis, and
  4. set the velocity DOFs.

Physics conventions (see README "Caveats"):

  * Velocity: a quasi-dynamic tandem run has no inertial bulk velocity, so the
    handoff to a fully-dynamic SeisSol run starts from rest -- velocity DOFs are
    set to zero. (`--velocity zero`, the only mode currently implemented.)
  * Stress: SeisSol carries the *static prestress* on the fault
    (`initialStressInFaultCS`) and lets the bulk DOFs hold only the slip-induced
    perturbation. tandem's displacement, measured from the t=0 configuration,
    yields exactly that perturbation, so it maps onto the bulk DOFs directly.
    The bridge does NOT add a background stress to the volume.

Because displacement is degree p and stress is degree p-1, and SeisSol tets are
straight-sided (affine map, constant Jacobian), both the modal fit and the
stress projection are *exact* (no quadrature error) -- the basis spans the
field exactly, so interpolation == L2 projection.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tandem_domain_reader import TandemDomainSnapshot, read_tandem_domain
from .dubiner_tet import num_modes, tet_grad_vandermonde, tet_vandermonde
from .seissol_mesh_reader import MeshElements, read_seissol_elements
from .patcher import patch_lts_dofs

# BP7 (homogeneous): rho=2670 kg/m^3, cs=3464 m/s, nu=0.25
#   mu = rho*cs^2 = 3.2038e10 Pa ; lambda = 2*nu*mu/(1-2nu) = mu (since nu=0.25)
BP7_LAMBDA = 3.2038e10
BP7_MU = 3.2038e10

# stress component (i,j) for each of the 6 SeisSol quantities s_xx..s_xz
STRESS_IJ = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))


@dataclass
class VolumeBridgeResult:
    element_ids: np.ndarray  # (n_cells,) SeisSol element ids (== lts __ids)
    q_flat: np.ndarray  # (n_cells, 9*nbf) modal DOFs, SeisSol flat layout
    time: float
    max_distance: float
    mean_distance: float
    n_cells: int
    stress_stats: dict = field(default_factory=dict)


def compute_volume_dofs(
    snap: TandemDomainSnapshot,
    mesh: MeshElements,
    *,
    order: int = 4,
    coord_scale: float = 1.0,
    disp_scale: float = 1.0,
    lam: float = BP7_LAMBDA,
    mu: float = BP7_MU,
    tol: float = 1e-4,
    velocity: str = "zero",
) -> VolumeBridgeResult:
    """Project tandem domain displacement to SeisSol modal volume DOFs.

    coord_scale : physical metres per tandem *coordinate* unit (for matching and
                  for the reference-frame Jacobian). BP7 coordinates are in km
                  -> 1000.
    disp_scale  : physical metres per tandem *displacement* unit. This is the
                  same convention as the fault bridge's --slip-scale and is
                  independent of coord_scale. BP7 (like its velocities, in m/s)
                  outputs displacement in metres -> 1.0. Strain is computed in
                  true physical metres: eps = disp_scale * d(u_vtu)/d(x_metres),
                  where x_metres already includes coord_scale.
    lam, mu     : Lame parameters [Pa]. Stress comes out in Pa.
    """
    if snap.degree != order - 1:
        raise RuntimeError(f"domain VTU degree {snap.degree} != order-1 = {order - 1}")
    nbf = num_modes(order)
    ppc = snap.points_per_cell
    if ppc != nbf:
        raise RuntimeError(
            f"points/cell {ppc} != modes {nbf}; non-square fit unsupported"
        )
    if velocity != "zero":
        raise NotImplementedError(
            f"velocity mode {velocity!r} not implemented (only 'zero')"
        )

    n = snap.n_cells

    # --- match each domain cell to a SeisSol element by centroid ---------------
    tree = cKDTree(mesh.centroid)
    dom_centroids_m = snap.corner_centroids * coord_scale
    dist, idx = tree.query(dom_centroids_m)
    max_d, mean_d = float(dist.max()), float(dist.mean())
    if max_d > tol:
        raise RuntimeError(
            f"volume centroid match too loose: max_distance={max_d:.3e} > tol={tol:.3e}. "
            f"Check --coord-scale or whether the meshes are identical."
        )
    element_ids = mesh.element_id[idx]
    vm = mesh.vertices[idx]  # (n,4,3) metres, mesh order

    # --- affine reference->global map per cell: x = v0 + A xi ------------------
    a_mat = np.stack(
        [vm[:, 1] - vm[:, 0], vm[:, 2] - vm[:, 0], vm[:, 3] - vm[:, 0]], axis=2
    )  # (n,3,3) columns
    jinv = np.linalg.inv(a_mat)  # xi = jinv (x - v0)

    # reference coords of every domain node (dimensionless; unit-independent)
    xm = snap.node_points * coord_scale  # (n,ppc,3) metres
    rel = xm - vm[:, 0:1, :]
    xi = np.einsum("cij,cpj->cpi", jinv, rel)  # (n,ppc,3)

    # --- basis + gradient at those reference points ----------------------------
    vand = tet_vandermonde(xi.reshape(-1, 3), order).reshape(n, ppc, nbf)
    grad = tet_grad_vandermonde(xi.reshape(-1, 3), order).reshape(n, ppc, nbf, 3)

    # --- modal displacement (raw tandem units), then physical strain -----------
    cu = np.linalg.solve(vand, snap.displacement)  # (n,nbf,3) coeffs
    gu_ref = np.einsum("cpmd,cmk->cpkd", grad, cu)  # d u_k / d xi_d
    gu_phys = np.einsum("cpkd,cdj->cpkj", gu_ref, jinv)  # d u_k / d x_j  [u_t/m]
    gu_phys *= disp_scale  # -> dimensionless strain rate
    eps = 0.5 * (gu_phys + np.swapaxes(gu_phys, 2, 3))  # (n,ppc,3,3)
    trace = eps[..., 0, 0] + eps[..., 1, 1] + eps[..., 2, 2]
    sig = 2.0 * mu * eps
    for d in range(3):
        sig[..., d, d] += lam * trace

    stress6 = np.stack([sig[..., i, j] for (i, j) in STRESS_IJ], axis=-1)  # (n,ppc,6)
    cs = np.linalg.solve(vand, stress6)  # (n,nbf,6) modal stress

    # --- pack into SeisSol flat layout: index = quantity*nbf + mode ------------
    q_flat = np.zeros((n, 9 * nbf), dtype=np.float64)
    for q in range(6):
        q_flat[:, q * nbf : (q + 1) * nbf] = cs[:, :, q]
    # velocity quantities 6,7,8 remain zero

    stats = {
        name: (float(sig[..., i, j].min()), float(sig[..., i, j].max()))
        for name, (i, j) in zip(
            ("s_xx", "s_yy", "s_zz", "s_xy", "s_yz", "s_xz"), STRESS_IJ
        )
    }
    return VolumeBridgeResult(
        element_ids=element_ids,
        q_flat=q_flat,
        time=snap.time,
        max_distance=max_d,
        mean_distance=mean_d,
        n_cells=n,
        stress_stats=stats,
    )


def bridge_domain(
    *,
    domain_vtu: str,
    mesh: str,
    checkpoint: str,
    output: str,
    order: int = 4,
    coord_scale: float = 1.0,
    disp_scale: float = 1.0,
    lam: float = BP7_LAMBDA,
    mu: float = BP7_MU,
    tol: float = 1e-4,
    velocity: str = "zero",
    dry_run: bool = False,
    overwrite: bool = False,
    copy_output: bool = True,
) -> dict[str, int] | None:
    """Run the domain (volume) pipeline end-to-end; patches `/checkpoint/lts/dofs`.

    copy_output : copy `checkpoint` -> `output` first. Set False when another step
                  (e.g. fault_bridge.bridge_fault) already created `output` -- both
                  steps must never copy independently, or the second copy would
                  discard the first step's patched rows.

    Returns the patch_lts_dofs report dict, or None if dry_run.
    """
    print(f"[1/4] Reading tandem domain VTU: {domain_vtu}")
    snap = read_tandem_domain(domain_vtu)
    print(
        f"       time={snap.time:.6g}  degree={snap.degree}  "
        f"cells={snap.n_cells}  ppc={snap.points_per_cell}"
    )

    print(f"[2/4] Reading SeisSol mesh elements: {mesh}")
    mesh_elements = read_seissol_elements(mesh)
    print(f"       elements: {len(mesh_elements.element_id)}")

    print(
        f"[3/4] Projecting displacement -> modal stress DOFs "
        f"(lambda={lam:g}, mu={mu:g}, coord_scale={coord_scale:g})"
    )
    res = compute_volume_dofs(
        snap,
        mesh_elements,
        order=order,
        coord_scale=coord_scale,
        disp_scale=disp_scale,
        lam=lam,
        mu=mu,
        tol=tol,
        velocity=velocity,
    )
    print(
        f"       centroid match: max={res.max_distance:.3e}  mean={res.mean_distance:.3e}"
    )
    print(f"       stress ranges [Pa] (volume perturbation, velocity=0):")
    for name, (lo, hi) in res.stress_stats.items():
        print(f"         {name:5s} {lo:+.4g} .. {hi:+.4g}")

    if dry_run:
        print("[4/4] dry-run; not writing output")
        return None

    out = Path(output)
    if copy_output:
        if out.exists() and not overwrite:
            raise FileExistsError(f"{out} already exists; pass --overwrite")
        shutil.copy(checkpoint, out)
    print(f"[4/4] Patching lts/dofs -> {out}")
    report = patch_lts_dofs(out, res.element_ids, res.q_flat, set_time=res.time)
    print(
        f"       rows_patched={report['rows_patched']}  "
        f"rows_missing_in_ckp={report['rows_missing_in_ckp']}"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tandem_to_seissol.domain_bridge", description=__doc__)
    p.add_argument("--domain-vtu", required=True, help="tandem domain .pvtu/.vtu")
    p.add_argument("--mesh", required=True, help="SeisSol PUML .h5 mesh")
    p.add_argument("--checkpoint", required=True, help="source SeisSol checkpoint .h5")
    p.add_argument("--output", required=True, help="patched checkpoint .h5 to write")
    p.add_argument(
        "--order", type=int, default=4, help="SeisSol convergence order (default 4)"
    )
    p.add_argument(
        "--coord-scale",
        type=float,
        default=1.0,
        help="metres per tandem coordinate unit (BP7: 1000)",
    )
    p.add_argument(
        "--disp-scale",
        type=float,
        default=1.0,
        help="metres per tandem displacement unit (BP7 outputs displacement "
        "in metres, like its m/s velocities, so 1.0; same convention as "
        "the fault bridge --slip-scale, independent of --coord-scale)",
    )
    p.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=BP7_LAMBDA,
        help=f"Lame lambda [Pa] (default BP7 {BP7_LAMBDA:g})",
    )
    p.add_argument(
        "--mu",
        type=float,
        default=BP7_MU,
        help=f"shear modulus mu [Pa] (default BP7 {BP7_MU:g})",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="max centroid match distance [SeisSol units] (default 1e-4)",
    )
    p.add_argument(
        "--velocity",
        default="zero",
        choices=["zero"],
        help="bulk velocity DOFs (only 'zero' = QD->FD rest start)",
    )
    p.add_argument("--dry-run", action="store_true", help="report only; do not write")
    p.add_argument(
        "--overwrite", action="store_true", help="overwrite an existing output"
    )
    args = p.parse_args(argv)

    bridge_domain(
        domain_vtu=args.domain_vtu,
        mesh=args.mesh,
        checkpoint=args.checkpoint,
        output=args.output,
        order=args.order,
        coord_scale=args.coord_scale,
        disp_scale=args.disp_scale,
        lam=args.lam,
        mu=args.mu,
        tol=args.tol,
        velocity=args.velocity,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
