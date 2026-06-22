"""Minimal HeapEnv for the M445 arm.

Brax-style JAX env, identical to heap_m545 but for the M445: dynamics come from a
learned RobotEnsemble (JAX) and the end-effector pose is computed with mjx forward
kinematics from the M445 URDF. Same 4-DOF arm in the CABIN frame (boom, stick, tele,
pitch); the cabin slew (J_TURN) is fixed in the URDF since it does not affect EE-in-CABIN.
"""
import os
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from flax import nnx

from safe_mbrl.envs.base import Env, State
from safe_mbrl.utils.structs import RobotState
from safe_mbrl.models.online_trainer import _model_step
from safe_mbrl.models.robot_ensemble import RobotEnsemble

file_path = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(file_path, "heap_env/rsc/m445/m445_shovel_fixed_w_cabin.urdf")
EE_BODY, ROOT_BODY = "ENDEFFECTOR_CONTACT", "CABIN"

# M445 arm joint ranges from the URDF limits (order boom, stick, tele, pitch). Reset samples reachable.
POS_LIMIT = jnp.array([[-1.38, 0.36], [0.59, 2.73], [0.0, 1.598], [-0.66, 2.298]])


def _quat2mat(q):
    w, x, y, z = q
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def _so3_error(R, R_d):
    """SO(3) attitude error vector e_R = 0.5 * vee(R_d^T R - R^T R_d) (Lee et al.,
    geometric tracking). |e_R| ~ sin(angle) between R and R_d; zero iff R == R_d."""
    M = R_d.T @ R - R.T @ R_d
    return 0.5 * jnp.array([M[2, 1], M[0, 2], M[1, 0]])


class M445FK:
    """mjx forward kinematics: joint vector -> EE position in the CABIN frame.

    The URDF has .dae meshes mujoco can't read, so visual/collision geoms are
    stripped; `fusestatic=false` keeps the CABIN / ENDEFFECTOR_CONTACT frames
    (mujoco otherwise welds fixed-joint bodies away).
    """

    def __init__(self, urdf_path=URDF_PATH):
        root = ET.parse(urdf_path).getroot()
        for link in root.findall("link"):
            for tag in ("visual", "collision"):
                for e in link.findall(tag):
                    link.remove(e)
        ET.SubElement(ET.SubElement(root, "mujoco"), "compiler",
                      {"fusestatic": "false", "balanceinertia": "true"})
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

        self._mx = mjx.put_model(m)
        self._ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
        self._root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
        self.jnt_range = jnp.asarray(m.jnt_range)      # (nq, 2)
        self.nq = m.nq

    def ee_pos(self, q):
        """q: (nq,) -> EE position (3,) expressed in the CABIN frame."""
        d = mjx.make_data(self._mx).replace(qpos=q)
        d = mjx.kinematics(self._mx, d)
        R_root = _quat2mat(d.xquat[self._root])
        return R_root.T @ (d.xpos[self._ee] - d.xpos[self._root])

    def ee_pose(self, q):
        """q: (nq,) -> (EE position (3,), EE rotation (3,3)) in the CABIN frame."""
        d = mjx.make_data(self._mx).replace(qpos=q)
        d = mjx.kinematics(self._mx, d)
        Rt = _quat2mat(d.xquat[self._root]).T
        pos = Rt @ (d.xpos[self._ee] - d.xpos[self._root])
        R = Rt @ _quat2mat(d.xquat[self._ee])
        return pos, R

    def ee_twist(self, q, qd):
        """q, qd: (nq,) -> EXACT 6D EE twist [v; w] in the CABIN frame, via the geometric Jacobian J(q) @ qd"""
        d = mjx.make_data(self._mx).replace(qpos=q)
        d = mjx.kinematics(self._mx, d)
        d = mjx.com_pos(self._mx, d)
        jacp, jacr = mjx.jac(self._mx, d, d.xpos[self._ee], self._ee)   # (nv, 3) each, world frame
        Rt = _quat2mat(d.xquat[self._root]).T
        return jnp.concatenate([Rt @ (qd @ jacp), Rt @ (qd @ jacr)])    # (6,) twist in CABIN frame


class HeapEnv(Env):
    """Brax-style env: learned-model dynamics, mjx FK, heap-style tracking reward."""

    def __init__(self, model, cfg=None):
        self.cfg = cfg
        self.fk = M445FK()
        self._jd = model.joint_dim
        self._bd = model.buffer_dim
        self._mode = model.mode
        self._dt = model._dt
        self._action_coef = getattr(cfg, "action_penalty_coef", 0.0) if cfg is not None else 0.1
        self._twist_coef = getattr(cfg, "twist_coef", 1.0) if cfg is not None else 1.0   # exact-twist tracking
        # Tracking mode: "pose" -> task-space TF (position + rotation) + optional twist;
        # "joint" -> joint position + optional joint velocity. Selected via cfg.track_mode.
        self._track_mode = getattr(cfg, "track_mode", "joint") if cfg is not None else "joint"
        self._rot_coef = getattr(cfg, "rot_coef", 1.0) if cfg is not None else 1.0        # attitude tracking
        self._qd_coef = getattr(cfg, "qd_coef", 0.1) if cfg is not None else 1.0          # joint-velocity tracking
        self._graphdef, self._params = nnx.split(model.model)

    def reset(self, rng: jax.Array) -> State:
        rng, r_tgt, r_init = jax.random.split(rng, 3)
        lo, hi = POS_LIMIT[:, 0], POS_LIMIT[:, 1]
        q_target = jax.random.uniform(r_tgt, (self._jd,), minval=lo, maxval=hi)
        
        q0 = jax.random.uniform(r_init, (self._jd,), minval=lo, maxval=hi)

        rs = RobotState.create(q0, buffer_size=self._bd, q_dim=self._jd)

        info = {"last_action": jnp.zeros(self._jd),
                "step": jnp.zeros((), jnp.int32), "params": self._params}
        if self._track_mode == "joint":
            info["q_target_seq"] = q_target[None, :]                      # (1, jd)
        else:
            pos, R = self.fk.ee_pose(q_target)                           # CABIN-frame target pose
            info["ee_target_seq"] = jnp.eye(4).at[:3, :3].set(R).at[:3, 3].set(pos)[None]   # (1, 4, 4)

        z = jnp.zeros(())

        return State(rs, self._get_obs(rs, info), z, z, {"reward": z}, info)
    
    def make_traj_state(self, q_buf, qd_buf, act_buf, target_seq, jd, aux_seq=None):
        # Important method for real world deployment -> here we just set the inputs from the
        # Real world, as q_buf, qd_buf, and act_buf, and the reference window `target_seq`.
        # `target_seq`/`aux_seq` follow the env's track_mode:
        #   "pose"  -> target_seq = EE pose seq (T,4,4) or position seq (T,3); aux_seq = twist (T,6)
        #   "joint" -> target_seq = joint-pos seq (T,jd);                      aux_seq = joint-vel (T,jd)

        rs = RobotState(q_buffer=jnp.asarray(q_buf),
                        qd_buffer=jnp.asarray(qd_buf),
                        act_buffer=jnp.asarray(act_buf),
                        q_dim=jd)

        info = {"last_action": jnp.zeros(jd),
                "step": jnp.zeros((), jnp.int32),
                "params": self._params}
        if self._track_mode == "joint":
            info["q_target_seq"] = jnp.asarray(target_seq)
            if aux_seq is not None:
                info["qd_target_seq"] = jnp.asarray(aux_seq)
        else:
            info["ee_target_seq"] = jnp.asarray(target_seq)
            if aux_seq is not None:
                info["twist_target_seq"] = jnp.asarray(aux_seq)

        z = jnp.zeros(())

        return State(rs, self._get_obs(rs, info), z, z, {"reward": z}, info)

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)

        # merge a fresh ensemble from the carried params, then advance one BPTT-consistent step
        ens = nnx.merge(self._graphdef, state.info["params"])
        rs = _model_step(ens, state.pipeline_state, action, self._jd, self._mode, self._dt)

        i = state.info["step"]
        # tracking reward + action-rate penalty over the rollout (mirrors the sim's
        # smoothness term) -> rewards smooth, low-jitter action sequences while planning.
        reward = self._track_reward(rs, state.info, i) \
            - self._action_coef * jnp.sum((action - state.info["last_action"]) ** 2)
        info = {**state.info, "last_action": action, "step": i + 1}
        return State(rs, self._get_obs(rs, info), reward, jnp.zeros(()), state.metrics, info)

    # kept for API compatibility (non-jit callers); identical semantics to step().
    step_general = step

    def _track_reward(self, rs: RobotState, info, i) -> jax.Array:
        """Reward for tracking THIS rollout step's reference (OOB index clamps to last).
        Two modes, selected by self._track_mode:
          "joint" -> -||q - q_ref||^2 (- qd_coef ||qd - qd_ref||^2 if a qd ref is given)
          "pose"  -> EE TF error: -||pos - pos_ref||^2 (- rot_coef ||e_R||^2 when the
                     reference is a full (4,4) homogeneous transform; a (3,) reference is
                     position-only for back-compat) (- twist_coef ||twist - twist_ref||^2
                     when a twist ref is given). EE comes from the differentiable mjx FK.
        """
        if self._track_mode == "joint":
            reward = -jnp.sum((rs.get_q() - info["q_target_seq"][i]) ** 2)
            if "qd_target_seq" in info:
                reward = reward - self._qd_coef * jnp.sum((rs.get_qd() - info["qd_target_seq"][i]) ** 2)
            return reward

        target = info["ee_target_seq"][i]
        if info["ee_target_seq"].ndim == 3:                              # (T, 4, 4) full pose
            pos, R = self.fk.ee_pose(rs.get_q())
            reward = -jnp.sum((pos - target[:3, 3]) ** 2) \
                     - self._rot_coef * jnp.sum(_so3_error(R, target[:3, :3]) ** 2)
        else:                                                           # (T, 3) position only
            reward = -jnp.sum((self.fk.ee_pos(rs.get_q()) - target) ** 2)
        if "twist_target_seq" in info:
            twist = self.fk.ee_twist(rs.get_q(), rs.get_qd())
            reward = reward - self._twist_coef * jnp.sum((twist - info["twist_target_seq"][i]) ** 2)
        return reward

    def _get_obs(self, rs: RobotState, info) -> jax.Array:
        parts = [rs.ravel()]
        if self._track_mode == "joint":
            parts.append(info["q_target_seq"][0])
            if "qd_target_seq" in info:
                parts.append(info["qd_target_seq"][0])
        else:
            parts.append(jnp.ravel(info["ee_target_seq"][0]))           # (3,) or flattened (4,4)
            if "twist_target_seq" in info:
                parts.append(info["twist_target_seq"][0])
        return jnp.concatenate(parts)

    def generate_polynomial_traj(self,
                                p_start:jax.Array,
                                p_end: jax.Array,
                                p_max:jax.Array,
                                p_min:jax.Array,
                                t_traj:float = 6.,
                                ref_traj_steps: int = 100)->jax.Array:
        
        """
        Generate smooth polynomial trajectories from p_start to p_end over time t for multiple_trajectories
        
        return:
        shape(ref_traj_steps, p_start.shape[0])
        """
        t = jnp.linspace(0., t_traj, ref_traj_steps, dtype=jnp.float32)[:, None]
        v_start, v_end = jnp.zeros_like(p_start), jnp.zeros_like(p_end)
        a0, a1, a2 = p_start, v_start * t_traj, jnp.zeros_like(p_start)

        A = jnp.array([[t_traj**3, t_traj**4, t_traj**5],
                        [3*t_traj**2, 4*t_traj**3, 5*t_traj**4],
                        [6*t_traj, 12*t_traj**2, 20*t_traj**3]], dtype=jnp.float32)


        B = jnp.stack([p_end - (a0 + a1 + a2),
                       v_end - (a1 + 2*a2*t_traj),
                       jnp.zeros_like(p_start)])

        
        a3, a4, a5 = jnp.linalg.solve(A, B.reshape(3, -1)).reshape(B.shape)
        position = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
        position = jnp.clip(position, p_min, p_max)
        return position

    @property
    def observation_size(self):
        return int(self.reset(jax.random.key(0)).obs.shape[-1])

    @property
    def action_size(self) -> int:
        return self._jd

    @property
    def backend(self) -> str:
        return "mjx-fk + learned-ensemble dynamics"


if __name__ == "__main__":
    p_start = jnp.array([0.,0.1,0.2,0.04], dtype = jnp.float32)
    p_end = jnp.array([0.4,.6,0.3,0.6], dtype=jnp.float32)
    
    p_min = 0.0 
    p_max = 0.6
    jd, bd = 4, 15
    model = RobotEnsemble(joint_dim=jd, buffer_dim=bd, model_type="PE", mode="v", num_ensembles=5)
    env = HeapEnv(model)
    
    position = env.generate_polynomial_traj(p_start, p_end, p_max, p_min)
    import matplotlib.pyplot as plt
    
    plt.plot(position)
    plt.show()
    print(position.shape)
    #key = jax.random.key(0)
    
    #env_state = env.reset(key)
    

    