"""Min-jerk joint trajectories with an EE-frame speed cap."""
import numpy as np

from safe_mbrl.m445_hammer_spec import JOINT_VMAX

_PEAK_SD = 1.875  # max of ds/dtau at tau = 1/2


def min_jerk_ref(q0, q1, t_traj: float, t_step: float):
    """Quintic (zero vel + acc at both ends) on the control grid -> (q_ref, qd_ref)."""
    q0 = np.asarray(q0, np.float32)
    dq = np.asarray(q1, np.float32) - q0
    n = int(round(t_traj / t_step)) + 1
    tau = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    s = tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)
    sd = 30.0 * tau**2 * (1.0 - tau) ** 2 / t_traj
    return q0 + dq * s, dq * sd


def ee_speed_duration(q0, q1, fk_pos, v_ee_max: float = 0.20,
                      vmax_joints=JOINT_VMAX, t_step: float = 0.04,
                      n_samples: int = 64) -> float:
    """Duration T so peak EE speed <= v_ee_max (conservative product bound)."""
    q0 = np.asarray(q0, np.float32)
    dq = np.asarray(q1, np.float32) - q0
    s = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    xyz = np.asarray(fk_pos(q0 + dq * s[:, None]))
    g = np.linalg.norm(np.diff(xyz, axis=0), axis=-1) * (n_samples - 1)
    t_ee = _PEAK_SD * float(g.max()) / v_ee_max
    t_joint = _PEAK_SD * float((np.abs(dq) / np.asarray(vmax_joints)).max())
    return max(t_step, np.ceil(max(t_ee, t_joint) / t_step) * t_step)
