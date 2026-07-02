"""Patch a SeisSol checkpoint .h5 with bridged tandem fault state.

Only the dynrup variables this bridge produces are overwritten; everything
else (LTS dofs, surface displacementDofs, ...) is left untouched.

The output is always a copy of the input file so the original is preserved.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

SIDX_TNN = 0  # initialStressInFaultCS index for normal traction
SIDX_T1 = 3  # ... for shear along tangent1
SIDX_T2 = 5  # ... for shear along tangent2

_PADDING_DATA_PATH = Path(__file__).resolve().parent / "data" / "dr_padding.json"


def load_dr_padding(order: int) -> int:
    """Return NumPaddedPoints for `order`, from the vendored padding table.

    dynrup arrays are padded to a multiple of the SIMD vector width (see
    SeisSol's `misc::NumPaddedPoints`, DynamicRupture/Misc.h), which depends
    on ConvergenceOrder *and* the SIMD width / precision the SeisSol binary
    was built with. `data/dr_padding.json` was generated for one specific
    build (Vectorsize=32 bytes, F64 -- see its "note" field) and is only
    valid for checkpoints from a matching build; `patch_checkpoint` cross-
    checks it against the checkpoint's actual row width and raises if they
    disagree, rather than guessing.
    """
    with _PADDING_DATA_PATH.open() as f:
        table = json.load(f)
    key = str(order)
    if key not in table["pad_by_order"]:
        raise ValueError(
            f"no vendored padding for order {order}; available orders: "
            f"{sorted(int(k) for k in table['pad_by_order'])}"
        )
    return int(table["pad_by_order"][key])


@dataclass
class BridgedFields:
    """Per-tandem-cell fields evaluated at SeisSol's Stroud points.

    All arrays are shape (n_cells, n_quad), where n_quad depends on the
    ConvergenceOrder (9/16/25/36/49/64/81 for order 2-8). A field set to
    None is skipped by the patcher (original checkpoint values are preserved).
    """

    seissol_face_ids: np.ndarray  # (n_cells,) uint64
    slip1: np.ndarray
    slip2: np.ndarray
    slip_rate1: np.ndarray
    slip_rate2: np.ndarray
    state_variable: np.ndarray | None
    accumulated_slip_magnitude: np.ndarray
    Tnn: np.ndarray
    T1: np.ndarray
    T2: np.ndarray
    time: float


def _dataset_pad_width(dset: h5py.Dataset) -> int:
    """NumPaddedPoints for a dynrup row dataset, read from the dataset itself.

    The padded width depends on both ConvergenceOrder and the SIMD width the
    writing SeisSol binary was built for, so it can't be inferred from order
    alone -- the checkpoint file is the only reliable source of truth.
    """
    return dset.dtype.shape[0] if dset.dtype.shape else dset.shape[1]


def _pad_to(arr: np.ndarray, pad: int) -> np.ndarray:
    """Pack arr's trailing (real quadrature) values into a `pad`-wide array."""
    n_quad = arr.shape[-1]
    if n_quad > pad:
        raise ValueError(
            f"bridged data has {n_quad} quadrature points but the checkpoint "
            f"row only has room for {pad} -- order mismatch between the "
            f"tandem VTU/--order and this checkpoint?"
        )
    out = np.zeros(arr.shape[:-1] + (pad,), dtype=np.float64)
    out[..., :n_quad] = arr
    return out


def patch_checkpoint(
    source_h5: str | Path,
    target_h5: str | Path,
    bridged: BridgedFields,
    *,
    order: int,
    overwrite_existing: bool = False,
) -> dict[str, int]:
    """Copy source -> target, then overwrite dynrup rows from bridged data.

    `order` looks up the expected NumPaddedPoints from the vendored padding
    table (`load_dr_padding`); this is cross-checked against the checkpoint's
    actual row width so a build with a different SIMD width/precision fails
    with a clear message instead of writing misaligned data.

    Returns a small report dict (rows touched, rows missing).
    """
    source_h5 = Path(source_h5)
    target_h5 = Path(target_h5)
    if target_h5.exists() and not overwrite_existing:
        raise FileExistsError(
            f"{target_h5} already exists; pass overwrite_existing=True"
        )
    shutil.copy(source_h5, target_h5)

    report = {"rows_patched": 0, "rows_missing_in_ckp": 0}
    with h5py.File(target_h5, "r+") as f:
        dr = f["/checkpoint/dynrup"]
        ids = dr["__ids"][:]
        id_to_row = {int(v): i for i, v in enumerate(ids)}

        pad = load_dr_padding(order)
        actual_pad = _dataset_pad_width(dr["slip1"])
        if actual_pad != pad:
            raise ValueError(
                f"vendored padding for order={order} is {pad}, but this checkpoint's "
                f"dynrup rows are {actual_pad} wide -- it was likely built with a "
                f"different SIMD width or precision than data/dr_padding.json assumes. "
                f"Regenerate dr_padding.json for this build, or pass the checkpoint's "
                f"own width explicitly."
            )

        for i, sei_id in enumerate(bridged.seissol_face_ids):
            row = id_to_row.get(int(sei_id))
            if row is None:
                report["rows_missing_in_ckp"] += 1
                continue

            dr["slip1"][row] = _pad_to(bridged.slip1[i], pad)
            dr["slip2"][row] = _pad_to(bridged.slip2[i], pad)
            dr["slipRate1"][row] = _pad_to(bridged.slip_rate1[i], pad)
            dr["slipRate2"][row] = _pad_to(bridged.slip_rate2[i], pad)
            if bridged.state_variable is not None:
                dr["stateVariable"][row] = _pad_to(bridged.state_variable[i], pad)
            dr["accumulatedSlipMagnitude"][row] = _pad_to(
                bridged.accumulated_slip_magnitude[i], pad
            )

            stress = np.zeros((6, pad), dtype=np.float64)
            stress[SIDX_TNN] = _pad_to(bridged.Tnn[i], pad)
            stress[SIDX_T1] = _pad_to(bridged.T1[i], pad)
            stress[SIDX_T2] = _pad_to(bridged.T2[i], pad)
            dr["initialStressInFaultCS"][row] = stress

            report["rows_patched"] += 1

        f["/checkpoint"].attrs["__time"] = bridged.time

    return report


# Number of basis functions packed per quantity in the checkpoint dofs layout.
# Confirmed unpadded for ConvergenceOrder=4 (tensor::Q::size()=180=20*9) and via
# PostProcessor.cpp accessing quantity q at dofs[NumAlignedBasisFunctions * q].
LTS_NUM_QUANTITIES = 9


def patch_lts_dofs(
    target_h5: str | Path,
    element_ids: np.ndarray,
    q_flat: np.ndarray,
    *,
    set_time: float | None = None,
) -> dict[str, int]:
    """Overwrite `/checkpoint/lts/dofs` rows in an existing checkpoint copy.

    Edits `target_h5` *in place* (no copy), so it can be chained after
    `patch_checkpoint` to produce a single fault+volume restart file.

    Parameters
    ----------
    element_ids : (n,) the SeisSol element global ids to patch (match lts __ids).
    q_flat      : (n, dofs_len) modal DOFs already packed in SeisSol's flat
                  layout: index = quantity * n_basis + mode, with quantities
                  [s_xx,s_yy,s_zz,s_xy,s_yz,s_xz,v1,v2,v3].
    set_time    : if given, also (re)write the /checkpoint __time attribute.
    """
    target_h5 = Path(target_h5)
    report = {"rows_patched": 0, "rows_missing_in_ckp": 0}
    with h5py.File(target_h5, "r+") as f:
        lts = f["/checkpoint/lts"]
        ids = lts["__ids"][:]
        id_to_row = {int(v): i for i, v in enumerate(ids)}
        dofs = lts["dofs"]
        dofs_len = dofs.dtype.shape[0] if dofs.dtype.shape else dofs.shape[1]
        if q_flat.shape[1] != dofs_len:
            raise ValueError(
                f"q_flat has {q_flat.shape[1]} dofs/element but checkpoint expects {dofs_len}"
            )
        for i, eid in enumerate(element_ids):
            row = id_to_row.get(int(eid))
            if row is None:
                report["rows_missing_in_ckp"] += 1
                continue
            dofs[row] = q_flat[i]
            report["rows_patched"] += 1
        if set_time is not None:
            f["/checkpoint"].attrs["__time"] = set_time
    return report
