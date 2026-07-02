"""Bridge tandem fault snapshots into a SeisSol dynrup checkpoint.

See README.md for usage. The pipeline is:

  1. Read tandem fault VTU  →  TandemFaultSnapshot  (vtu_reader)
  2. Read SeisSol mesh      →  FaultFace list        (mesh_reader)
  3. Match centroids        →  FaceMatch             (face_match)
  4. Build projection map   →  P matrix              (basis_map)
  5. Build face transforms  →  PerFaceTransform list (rotation)
  6. TandemBridge.convert   →  BridgedFields         (tandem_bridge)
  7. patch_checkpoint       →  patched .h5           (patcher)

Public API (for programmatic use outside the CLI):
"""

from .basis_map import create_basis_map
