"""Minimal HeapEnv for the M445 arm.

Brax-style JAX env: dynamics from a learned RobotEnsemble, EE pose via mjx FK.
Two modes selected by the model's joint_dim: 4 DOF (arm only, J_TURN fixed, EE in
the CABIN frame) or 5 DOF (J_TURN made revolute, EE in the BASE frame so the slew
moves the EE; qpos order [turn, boom, stick, tele, pitch]).
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
EE_BODY = "ENDEFFECTOR_CONTACT"
ROOT_BODY_4DOF, ROOT_BODY_5DOF = "CABIN", "BASE"

# M445 arm joint ranges from the URDF limits (order boom, stick, tele, pitch). Reset samples reachable.
POS_LIMIT = jnp.array([[-1.38, 0.36], [0.59, 2.73], [0.0, 1.598], [-0.66, 2.298]])
# J_TURN range for the 5-DOF mode. TODO: real safe slew range before hardware use.
TURN_LIMIT = jnp.array([[-3.1416, 3.1416]])




class M445FK:
    """mjx FK: joint vector -> EE position in the root frame (CABIN for 4 DOF,
    BASE with slew=True, which rewrites J_TURN to a revolute z-joint).

    The URDF has .dae meshes mujoco can't read, so visual/collision geoms are
    stripped; `fusestatic=false` keeps the CABIN / ENDEFFECTOR_CONTACT frames
    (mujoco otherwise welds fixed-joint bodies away).
    """

    def __init__(self, urdf_path=URDF_PATH, slew=False):
        root = ET.parse(urdf_path).getroot()
        for link in root.findall("link"):
            for tag in ("visual", "collision"):
                for e in link.findall(tag):
                    link.remove(e)
        if slew:
            j = root.find("joint[@name='J_TURN']")
            j.set("type", "revolute")
            ET.SubElement(j, "axis", {"xyz": "0 0 1"})
            ET.SubElement(j, "limit", {"lower": str(float(TURN_LIMIT[0, 0])),
                                       "upper": str(float(TURN_LIMIT[0, 1])),
                                       "effort": "100000", "velocity": "10"})
        ET.SubElement(ET.SubElement(root, "mujoco"), "compiler",
                      {"fusestatic": "false", "balanceinertia": "true"})
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

        self._mx = mjx.put_model(m)
        self._ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
        self._root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                       ROOT_BODY_5DOF if slew else ROOT_BODY_4DOF)
        self.jnt_range = jnp.asarray(m.jnt_range)      # (nq, 2)
        self.nq = m.nq

    def ee_pos(self, q):
        """q: (nq,) -> EE position (3,) expressed in the root frame."""
        d = mjx.make_data(self._mx).replace(qpos=q)
        d = mjx.kinematics(self._mx, d)
        R_root = _quat2mat(d.xquat[self._root])
        return R_root.T @ (d.xpos[self._ee] - d.xpos[self._root])

    def ee_pose(self, q):
        """q: (nq,) -> (EE position (3,), EE rotation (3,3)) in the root frame."""
        d = mjx.make_data(self._mx).replace(qpos=q)
        d = mjx.kinematics(self._mx, d)
        Rt = _quat2mat(d.xquat[self._root]).T
        pos = Rt @ (d.xpos[self._ee] - d.xpos[self._root])
        R = Rt @ _quat2mat(d.xquat[self._ee])
        return pos, R

    def ee_twist(self, q, qd):
        """q, qd: (nq,) -> EXACT 6D EE twist [v; w] in the root frame, via the geometric Jacobian J(q) @ qd"""
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
        self._jd = model.joint_dim
        if self._jd not in (4, 5):
            raise ValueError(f"HeapEnv supports 4 (arm) or 5 (J_TURN + arm) joints, got {self._jd}")
        self._slew = self._jd == 5
        self.fk = M445FK(slew=self._slew)
        self._pos_limit = (jnp.concatenate([TURN_LIMIT, POS_LIMIT])
                           if self._slew else POS_LIMIT)
        self._input_idx = getattr(model, "_input_idx", None)
        self._bd = model.buffer_dim
        self._mode = model.mode
        self._dt = model._dt
        self._action_coef = getattr(cfg, "action_penalty_coef", 0.0) if cfg is not None else 0.1
        self._twist_coef = getattr(cfg, "twist_coef", 1.0) if cfg is not None else 1.0   # exact-twist tracking
        # Tracking mode: "pose" -> task-space TF (position + rotation) + optional twist;
        # "joint" -> joint position + optional joint velocity;
        # "mpcc"  -> projection-variant MPC contouring over the FULL path
        #            (time-decoupled; see _mpcc_reward). Selected via cfg.track_mode.
        self._track_mode = getattr(cfg, "track_mode", "joint") if cfg is not None else "joint"
        if self._track_mode == "mpcc":
            m = getattr(cfg, "mpcc", None)
            if m is None or getattr(m, "qd_limit", None) is None:
                raise ValueError(
                    "track_mode 'mpcc' needs cfg.mpcc with a per-joint qd_limit "
                    "(the soft velocity cap) — add the mpcc: section to the YAML")
            qd_lim = jnp.asarray(m.qd_limit, jnp.float32)
            if qd_lim.shape != (self._jd,):
                raise ValueError(
                    f"mpcc.qd_limit needs {self._jd} entries, got {list(m.qd_limit)}")
            self._mpcc_w_ee = float(m.w_ee)
            self._mpcc_w_q = float(m.w_q)
            self._mpcc_delta = int(m.delta_max)
            self._mpcc_rho = float(m.rho)
            self._mpcc_sigma = float(m.sigma)
            self._mpcc_qd_lim = qd_lim
            self._mpcc_qd_lim_coef = float(m.qd_limit_coef)
            self._mpcc_ee_accel_coef = float(m.ee_accel_coef)
        # Velocity damping / joint-accel penalties (mpcc mode; 0 = disabled).
        self._accel_coef = getattr(cfg, "accel_penalty_coef", 0.0) if cfg is not None else 0.0
        self._rot_coef = getattr(cfg, "rot_coef", 1.0) if cfg is not None else 1.0        # attitude tracking
        self._qd_coef = getattr(cfg, "qd_coef", 0.1) if cfg is not None else 1.0          # joint-velocity tracking
        # Combined joint-mode loss: joint_weight * ||q - q_ref||^2 + ee_weight *
        # ||ee_xyz(q) - ee_xyz(q_ref)||^2. ee_weight = 0 -> pure joint tracking.
        self._joint_weight = getattr(cfg, "joint_weight", 1.0) if cfg is not None else 1.0
        self._ee_weight = getattr(cfg, "ee_weight", 0.0) if cfg is not None else 0.0
        # Per-joint scale for the joint-space cost: q/qd errors are divided by it
        # (the runner passes the safe range q_max - q_min), so every joint is
        # tracked with equal RELATIVE importance regardless of its range.
        # None -> ones (legacy raw-radians cost).
        scale = getattr(cfg, "q_cost_scale", None) if cfg is not None else None
        self._q_scale = (jnp.ones(self._jd) if scale is None
                         else jnp.asarray(scale, jnp.float32))
        # Per-joint multipliers on the joint-space error (q AND qd terms), on
        # top of the range normalization — emphasize joints whose tracking
        # matters more (e.g. J_TURN). None -> equal weights.
        w = getattr(cfg, "q_track_weight", None) if cfg is not None else None
        self._q_w = (jnp.ones(self._jd) if w is None
                     else jnp.asarray(w, jnp.float32))
        # Output EMA modeled INSIDE the rollout: the dynamics and the rate
        # penalty see the filtered action a = alpha * a_plan + (1 - alpha) *
        # a_applied_prev, mirroring the deployment-side filter, so the planner
        # optimizes knowing its raw plan will be smoothed. 1.0 = no filter.
        # Scalar or per-joint (jd,) array; the branch flag is static for jit.
        _ema = getattr(cfg, "action_ema_alpha", 1.0) if cfg is not None else 1.0
        self._ema_alpha = jnp.broadcast_to(jnp.asarray(_ema, jnp.float32), (self._jd,))
        self._use_ema = bool(jnp.any(self._ema_alpha < 1.0))
        # EE xyz of the joint reference, computed ONCE per reference window (jitted),
        # not per MPPI sample inside the rollout.
        self._ee_ref_fk = jax.jit(jax.vmap(self.fk.ee_pos))
        self._graphdef, self._params = nnx.split(model.model)

    def reset(self, rng: jax.Array) -> State:
        rng, r_tgt, r_init = jax.random.split(rng, 3)
        lo, hi = self._pos_limit[:, 0], self._pos_limit[:, 1]
        q_target = jax.random.uniform(r_tgt, (self._jd,), minval=lo, maxval=hi)
        
        q0 = jax.random.uniform(r_init, (self._jd,), minval=lo, maxval=hi)

        rs = RobotState.create(q0, buffer_size=self._bd, q_dim=self._jd)

        info = {"last_action": jnp.zeros(self._jd),
                "step": jnp.zeros((), jnp.int32), "params": self._params}
        if self._track_mode == "mpcc":
            # Degenerate 1-point path (reset is not used by the deployment
            # runners; this keeps obs shapes/introspection working).
            info.update(self._mpcc_path_info(
                q_target[None, :], self.fk.ee_pos(q_target)[None], 0))
        elif self._track_mode == "joint":
            info["q_target_seq"] = q_target[None, :]                      # (1, jd)
            if self._ee_weight > 0.0:
                info["ee_target_seq"] = self.fk.ee_pos(q_target)[None]    # (1, 3)
        else:
            pos, R = self.fk.ee_pose(q_target)                           # CABIN-frame target pose
            info["ee_target_seq"] = jnp.eye(4).at[:3, :3].set(R).at[:3, 3].set(pos)[None]   # (1, 4, 4)

        z = jnp.zeros(())

        return State(rs, self._get_obs(rs, info), z, z, {"reward": z}, info)
    
    def _mpcc_path_info(self, q_path, ee_path, path_idx):
        """MPCC info entries: paths padded with delta_max copies of the last
        point, so the projection window (dynamic_slice at theta of static size
        delta_max + 1) never clamps its start — clamping would let the window
        slide BACKWARD near the path end and break monotonicity. At the end
        theta stalls on the last real index (argmin ties -> first occurrence),
        progress stops paying, and the contour term holds the endpoint."""
        q_path = jnp.asarray(q_path, jnp.float32)                     # (T, jd)
        ee_path = jnp.asarray(ee_path, jnp.float32)                   # (T, 3)
        pad = self._mpcc_delta
        return {
            "q_path": jnp.concatenate([q_path, jnp.tile(q_path[-1:], (pad, 1))]),
            "ee_path": jnp.concatenate([ee_path, jnp.tile(ee_path[-1:], (pad, 1))]),
            "path_idx": jnp.asarray(path_idx, jnp.int32),
            "ee_prev": jnp.zeros(3),      # never read before step >= 2 (gated)
            "ee_prev2": jnp.zeros(3),
        }

    def make_traj_state(self, q_buf, qd_buf, act_buf, target_seq, jd, aux_seq=None,
                        last_action=None, path_idx=0):
        # Important method for real world deployment -> here we just set the inputs from the
        # Real world, as q_buf, qd_buf, and act_buf, and the reference window `target_seq`.
        # `target_seq`/`aux_seq` follow the env's track_mode:
        #   "pose"  -> target_seq = EE pose seq (T,4,4) or position seq (T,3); aux_seq = twist (T,6)
        #   "joint" -> target_seq = joint-pos seq (T,jd);                      aux_seq = joint-vel (T,jd)
        #   "mpcc"  -> target_seq = FULL joint path (T,jd) — not a window —
        #              aux_seq = its EE positions (T,3) precomputed once per
        #              trajectory; path_idx = theta_0, the UNCONSTRAINED
        #              joint-space projection of q_now (a mid-path replan must
        #              not catch up from index 0 through the monotone window).
        # `last_action` anchors the action-rate penalty of the FIRST rollout step to
        # the command actually applied last tick (None -> zeros, legacy behavior).

        rs = RobotState(q_buffer=jnp.asarray(q_buf),
                        qd_buffer=jnp.asarray(qd_buf),
                        act_buffer=jnp.asarray(act_buf),
                        q_dim=jd)

        info = {"last_action": (jnp.zeros(jd) if last_action is None
                                else jnp.asarray(last_action)),
                "step": jnp.zeros((), jnp.int32),
                "params": self._params}
        if self._track_mode == "mpcc":
            if aux_seq is None:
                raise ValueError("mpcc mode needs aux_seq = EE positions of the path")
            info.update(self._mpcc_path_info(target_seq, aux_seq, path_idx))
        elif self._track_mode == "joint":
            info["q_target_seq"] = jnp.asarray(target_seq)
            if aux_seq is not None:
                info["qd_target_seq"] = jnp.asarray(aux_seq)
            if self._ee_weight > 0.0:
                info["ee_target_seq"] = self._ee_ref_fk(jnp.asarray(target_seq))  # (T, 3)
        else:
            info["ee_target_seq"] = jnp.asarray(target_seq)
            if aux_seq is not None:
                info["twist_target_seq"] = jnp.asarray(aux_seq)

        z = jnp.zeros(())

        return State(rs, self._get_obs(rs, info), z, z, {"reward": z}, info)

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        # Apply the modeled output EMA: from here on `action` is the APPLIED
        # action (what the machine would receive), and info["last_action"]
        # carries the filter memory through the rollout.
        if self._use_ema:
            action = (self._ema_alpha * action
                      + (1.0 - self._ema_alpha) * state.info["last_action"])

        # merge a fresh ensemble from the carried params, then advance one BPTT-consistent step
        ens = nnx.merge(self._graphdef, state.info["params"])
        rs = _model_step(ens, state.pipeline_state, action, self._jd, self._mode, self._dt,
                         self._input_idx)

        i = state.info["step"]
        # tracking reward + action-rate penalty over the rollout (mirrors the sim's
        # smoothness term) -> rewards smooth, low-jitter action sequences while planning.
        if self._track_mode == "mpcc":
            reward, mpcc_updates = self._mpcc_reward(rs, state.info)
        else:
            reward, mpcc_updates = self._track_reward(rs, state.info, i), {}
        reward = reward \
            - self._action_coef * jnp.sum((action - state.info["last_action"]) ** 2)
        info = {**state.info, "last_action": action, "step": i + 1, **mpcc_updates}
        return State(rs, self._get_obs(rs, info), reward, jnp.zeros(()), state.metrics, info)

    # kept for API compatibility (non-jit callers); identical semantics to step().
    step_general = step

    def _mpcc_reward(self, rs: RobotState, info):
        """Projection-variant MPCC step reward -> (reward, info updates).

        theta (info["path_idx"]) is DERIVED, never optimized: the windowed
        argmin of the weighted contour metric
            d_i = w_ee ||p - p_path_i||^2 + w_q ||q - q_path_i||^2
        over [theta, theta + delta_max] — monotone (never slides backward)
        and rate-bounded (delta_max caps the rewarded path speed). Each MPPI
        sample carries its own theta / EE history through info. Reward:
            -c^2                          contouring (c^2 = d at the projection)
            + rho * dtheta * exp(-c^2/sigma^2)   proximity-gated progress:
                                          off the path the ONLY way to improve
                                          is to return — corner-cutting cannot
                                          buy progress (quasi-lexicographic)
            - qd_limit hinge              per-joint soft speed limit
            - ee_accel                    2nd-order-FD EE smoothness (task
                                          space: J(q) qd amplifies at reach,
                                          joint smoothness does not imply EE
                                          smoothness); gated for step < 2
            - qd damping / joint accel    optional (qd_coef / accel_penalty_coef)
        The action-rate term stays in step() (shared with the other modes).
        The exact projection is nonsmooth — MPPI only evaluates rollouts, so
        this variant is incompatible with derivative-based solvers.
        """
        q, qd = rs.get_q(), rs.get_qd()
        p = self.fk.ee_pos(q)                       # shared by contour + smoothness
        th_prev = info["path_idx"]
        w = self._mpcc_delta + 1
        q_win = jax.lax.dynamic_slice(info["q_path"], (th_prev, 0), (w, self._jd))
        p_win = jax.lax.dynamic_slice(info["ee_path"], (th_prev, 0), (w, 3))
        d = (self._mpcc_w_ee * jnp.sum((p - p_win) ** 2, axis=-1)
             + self._mpcc_w_q * jnp.sum((q - q_win) ** 2, axis=-1))
        off = jnp.argmin(d)
        c2 = d[off]
        n_path = info["q_path"].shape[0] - self._mpcc_delta          # T (static)
        # Clamp to the last REAL index: without it theta drifts into the pad
        # and the padded copies would pay up to delta_max of bogus progress.
        theta = jnp.minimum(th_prev + off, n_path - 1)
        dtheta = (theta - th_prev).astype(jnp.float32) / n_path

        reward = -c2 + self._mpcc_rho * dtheta * jnp.exp(-c2 / self._mpcc_sigma ** 2)

        over = jnp.maximum(0.0, jnp.abs(qd) - self._mpcc_qd_lim)
        reward = reward - self._mpcc_qd_lim_coef * jnp.sum(over ** 2)

        ee_acc = (p - 2.0 * info["ee_prev"] + info["ee_prev2"]) / self._dt ** 2
        reward = reward - jnp.where(
            info["step"] >= 2,                       # no EE history before that
            self._mpcc_ee_accel_coef * jnp.sum(ee_acc ** 2), 0.0)

        if self._qd_coef > 0.0:                      # velocity damping (off by default)
            reward = reward - self._qd_coef * jnp.sum(qd ** 2)
        if self._accel_coef > 0.0:                   # joint accel (off by default)
            qd_prev = rs.qd_buffer[-2 * self._jd:-self._jd]
            reward = reward - self._accel_coef * jnp.sum(((qd - qd_prev) / self._dt) ** 2)

        return reward, {"path_idx": theta, "ee_prev": p, "ee_prev2": info["ee_prev"]}

    def _track_reward(self, rs: RobotState, info, i) -> jax.Array:
        """Reward for tracking THIS rollout step's reference (OOB index clamps to last).
        Two modes, selected by self._track_mode:
          "joint" -> -joint_weight sum(w ((q - q_ref) / s)^2) - ee_weight ||ee_xyz - ee_xyz_ref||^2
                     (- qd_coef sum(w ((qd - qd_ref) / s)^2) if a qd ref is given), with s
                     the per-joint q_cost_scale (safe range) so all joints weigh equally
                     in RELATIVE terms, and w the per-joint q_track_weight multipliers
                     (emphasize e.g. J_TURN; default ones). The EE xyz term (mjx FK of q
                     vs FK of q_ref) is active when ee_weight > 0.
          "pose"  -> EE TF error: -||pos - pos_ref||^2 (- rot_coef ||e_R||^2 when the
                     reference is a full (4,4) homogeneous transform; a (3,) reference is
                     position-only for back-compat) (- twist_coef ||twist - twist_ref||^2
                     when a twist ref is given). EE comes from the differentiable mjx FK.
        """
        if self._track_mode == "joint":
            q_err = (rs.get_q() - info["q_target_seq"][i]) / self._q_scale
            reward = -self._joint_weight * jnp.sum(self._q_w * q_err ** 2)
            if "qd_target_seq" in info:
                qd_err = (rs.get_qd() - info["qd_target_seq"][i]) / self._q_scale
                reward = reward - self._qd_coef * jnp.sum(self._q_w * qd_err ** 2)
            if "ee_target_seq" in info:
                reward = reward - self._ee_weight * jnp.sum(
                    (self.fk.ee_pos(rs.get_q()) - info["ee_target_seq"][i]) ** 2)
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
        if self._track_mode == "mpcc":
            # The learned dynamics read the raw buffers, not obs; append the
            # current projection's path point so obs stays well-defined.
            parts.append(info["q_path"][info["path_idx"]])
        elif self._track_mode == "joint":
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
    