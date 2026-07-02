"""Top-level orchestrator: run the fault and/or domain bridge into one checkpoint.

Calls fault_bridge.bridge_fault() (patches `/checkpoint/dynrup`) and
domain_bridge.bridge_domain() (patches `/checkpoint/lts/dofs`) against the same
`--output` file. Either input may be omitted -- if `--vtu` or `--domain-vtu` is
missing, that half is skipped with a WARNING (not silently, and not a hard
failure, since a fault-only or domain-only restart is a legitimate use case).
At least one of the two must be given.

    python -m tandem_to_seissol.bridge \
        --vtu        /path/to/fault_full_7.pvtu \
        --domain-vtu /path/to/domain_7.pvtu \
        --mesh       /path/to/seissol_mesh.puml.h5 \
        --checkpoint /path/to/bp7-checkpoint-0.h5 \
        --output     /path/to/bp7-restart-from-tandem.h5 \
        --order      4 \
        --ref-normal 0,-1,0 --up 0,0,1 \
        --lambda 3.2038e10 --mu 3.2038e10

For single-pipeline runs (and their full flag sets), fault_bridge.py and
domain_bridge.py also each have their own standalone CLI.
"""

from __future__ import annotations

import argparse
import sys

from .domain_bridge import BP7_LAMBDA, BP7_MU, bridge_domain
from .fault_bridge import bridge_fault, parse_vec3


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tandem_to_seissol.bridge", description=__doc__)
    p.add_argument("--mesh", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--order", type=int, default=4, help="SeisSol convergence order (default: 4)"
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="max centroid distance for matching, in SeisSol units (default: 1e-4)",
    )
    p.add_argument(
        "--coord-scale",
        type=float,
        default=1.0,
        help="multiply tandem coordinates by this factor before matching "
        "(common: 1000 if tandem uses km and SeisSol uses m)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report only; do not write the patched checkpoint",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="overwrite an existing output file"
    )

    # ---- fault (dynrup) ----
    p.add_argument(
        "--vtu",
        default=None,
        help="tandem fault .pvtu/.vtu; if omitted, /checkpoint/dynrup is "
        "NOT patched (warns).",
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
        help="copy tandem psi straight into SeisSol stateVariable (debugging only; "
        "see fault_bridge.py --help for the caveat).",
    )
    p.add_argument(
        "--convert-state",
        action="store_true",
        help="convert tandem psi to SeisSol theta (requires --rs-L, --rs-V0, "
        "--rs-f0, --rs-b).",
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

    # ---- domain (lts/dofs) ----
    p.add_argument(
        "--domain-vtu",
        default=None,
        help="tandem domain (volume) .pvtu/.vtu; if omitted, /checkpoint/lts/dofs "
        "is NOT patched (warns). Must be the same simulation time as --vtu.",
    )
    p.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=BP7_LAMBDA,
        help=f"Lame lambda [Pa] for the domain bridge (default BP7 {BP7_LAMBDA:g})",
    )
    p.add_argument(
        "--mu",
        type=float,
        default=BP7_MU,
        help=f"shear modulus mu [Pa] for the domain bridge (default BP7 {BP7_MU:g})",
    )
    p.add_argument(
        "--disp-scale",
        type=float,
        default=1.0,
        help="metres per tandem displacement unit for the domain bridge "
        "(BP7: 1.0, displacement is in metres like slip; independent "
        "of --coord-scale)",
    )
    p.add_argument(
        "--velocity",
        default="zero",
        choices=["zero"],
        help="bulk velocity DOFs for the domain bridge "
        "(only 'zero' = quasi-dynamic -> dynamic rest start)",
    )
    args = p.parse_args(argv)

    if not args.vtu and not args.domain_vtu:
        p.error("at least one of --vtu or --domain-vtu is required")

    fault_report = None
    fault_wrote_output = False
    if args.vtu:
        fault_report = bridge_fault(
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
        fault_wrote_output = fault_report is not None
    else:
        print(
            "WARNING: --vtu not given; /checkpoint/dynrup (fault state) will NOT be patched."
        )

    if args.domain_vtu:
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
            # if the fault step already copied checkpoint -> output, don't copy again
            copy_output=not fault_wrote_output,
        )
    else:
        print(
            "WARNING: --domain-vtu not given; /checkpoint/lts/dofs (volume state) will NOT be patched."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
