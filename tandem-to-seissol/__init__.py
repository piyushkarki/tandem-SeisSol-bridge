"""Bridge tandem fault snapshots into a SeisSol dynrup checkpoint.

See README.md for usage. The pipeline is:

  1. Read tandem fault VTU  →  TandemFaultSnapshot  (tandem_fault_reader)
  2. Read SeisSol mesh      →  FaultFace list        (seissol_mesh_reader)
  3. Match centroids        →  FaceMatch             (face_match)
  4. Build projection map   →  P matrix              (basis_map)
  5. Build face transforms  →  PerFaceTransform list (rotation)
  6. TandemBridge.convert   →  BridgedFields         (fault_bridge)
  7. patch_checkpoint       →  patched .h5           (patcher)

Public API (for programmatic use outside the CLI):
"""

from .basis_map import create_basis_map
from .fault_bridge import TandemBridge, TandemBridgeConfig
from .field_scalar import (
    convert_psi_to_theta,
    normal_stress_sign_convention,
    project_scalar,
)
from .field_tensor import project_rotate_stress
from .field_vector import project_rotate_vector
