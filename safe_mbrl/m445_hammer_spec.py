"""M445 hammer constants: URDF, joint order, safe training-distribution box."""
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
HAMMER_URDF = os.path.join(
    _HERE, "envs", "heap_env", "rsc", "m445", "m445_hammer_w_cabin_simplified.urdf")

JOINT_NAMES = ("J_TURN", "J_BOOM", "J_STICK", "J_TELE", "J_EE_PITCH")

# joint_pos_*_safe from the 5-DOF deploy config: the ensemble is only trusted inside.
SAFE_Q_MIN = np.array([-0.3, -1.38, 0.75, 0.05, 0.18], np.float32)
SAFE_Q_MAX = np.array([0.3, -0.44, 1.8, 0.6, 1.9167], np.float32)

# URDF velocity limits; J_TURN has none (continuous) -> conservative value.
JOINT_VMAX = np.array([0.30, 0.46, 0.62, 0.64, 1.30], np.float32)

EE_BODY = "ENDEFFECTOR_CONTACT"
STRIKE_AXIS = np.array([1.0, 0.0, 0.0], np.float32)  # EE-frame x = hammer tool axis
