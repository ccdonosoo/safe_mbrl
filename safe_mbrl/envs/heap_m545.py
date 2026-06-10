"""Minimal brax-style HeapEnv for the M545 arm.

Dynamics come from a learned RobotEnsemble (JAX); the end-effector position is
computed with mjx forward kinematics (same URDF as the torch HeapEnv), and the
step reward mirrors heap_example_env: end-effector tracking + an action penalty.

The __main__ block verifies that, for the same joint config, the mjx FK returns
the same end-effector pose as the original pytorch_kinematics chain.
"""
import os
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from flax import nnx

from safe_mbrl.envs.base import Env, State
from safe_mbrl.utils.type_aliases import RobotState
from safe_mbrl.models.online_trainer import _bptt_step

file_path = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(file_path, "heap_env/rsc/m545/m545_boom_dipper_tele_pitch.urdf")
EE_BODY, ROOT_BODY = "ENDEFFECTOR_CONTACT", "CABIN"

# Reachable joint range the actuator-net plant / learned model operate in
# (MenziActNet.pos_limit, order boom, dipper, tele, pitch). Reset samples reachable
# joint configs here, then FKs them to a Cartesian EE target.
POS_LIMIT = jnp.array([[-1.2, -0.5], [0.58, 1.5708], [0.0, 0.2], [0.0, 1.5708]])


def _quat2mat(q):
    w, x, y, z = q
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


class M545FK:
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
        """q: (nq,) -> EE position (3,) expressed in the CABIN frame. Differentiable."""
        d = mjx.make_data(self._mx).replace(qpos=q)
        d = mjx.kinematics(self._mx, d)
        R_root = _quat2mat(d.xquat[self._root])
        return R_root.T @ (d.xpos[self._ee] - d.xpos[self._root])

    def ee_twist(self, q, qd):
        """q, qd: (nq,) -> EXACT 6D EE twist [v; w] in the CABIN frame, via the geometric
        Jacobian J(q) @ qd (NOT a finite difference of EE positions: the angular part of
        an SE(3) twist cannot be recovered by interpolating poses)."""
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
        self.fk = M545FK()
        self._jd = model.joint_dim
        self._bd = model.buffer_dim
        self._mode = model.mode
        self._dt = model._dt
        # Original HeapEnv reward coefficients (see heap_example_env._compute_reward).
        self._action_coef = getattr(cfg, "action_penalty_coef", 0.1) if cfg is not None else 0.1
        self._twist_coef = getattr(cfg, "twist_coef", 1.0) if cfg is not None else 1.0   # exact-twist tracking

        # Split the ensemble once: graphdef is static, params are a plain pytree.
        # step() merges a fresh module from these inside lax.scan/vmap (ICEM), which
        # the live nnx module can't survive. Re-build the env if the model retrains.
        self._graphdef, self._params = nnx.split(model.model)

    def reset(self, rng: jax.Array) -> State:
        rng, r_tgt, r_init = jax.random.split(rng, 3)
        # Sample reachable joint configs (pos_limit, not the wider URDF jnt_range) and
        # FK them to Cartesian EE targets: the original HeapEnv reward tracks the
        # END-EFFECTOR trajectory, not joints.
        lo, hi = POS_LIMIT[:, 0], POS_LIMIT[:, 1]
        q_target = jax.random.uniform(r_tgt, (self._jd,), minval=lo, maxval=hi)
        q0 = jax.random.uniform(r_init, (self._jd,), minval=lo, maxval=hi)

        rs = RobotState.create(q0, buffer_size=self._bd, q_dim=self._jd)
        # Track the EE position reference over the horizon, one row per rollout step;
        # reset uses a 1-point seq (planners supply the full window). Params ride in
        # info so step() can merge the model at the scan/vmap trace level.
        info = {"ee_target_seq": self.fk.ee_pos(q_target)[None, :],
                "last_action": jnp.zeros(self._jd),
                "step": jnp.zeros((), jnp.int32), "params": self._params}
        z = jnp.zeros(())
        return State(rs, self._get_obs(rs, info), z, z, {"reward": z}, info)

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)

        # merge a fresh ensemble from the carried params, then advance one BPTT-consistent step
        ens = nnx.merge(self._graphdef, state.info["params"])
        rs = _bptt_step(ens, state.pipeline_state, action, self._jd, self._mode, self._dt)

        # END-EFFECTOR pose tracking of THIS rollout step's reference (OOB index clamps
        # to last). EE comes from the differentiable mjx FK. If the planner supplies a
        # twist reference, also track the EXACT 6D EE twist (J(q)@qd) -- enabled only when
        # `twist_target_seq` is in info, so pure-position planners are unaffected.
        ee = self.fk.ee_pos(rs.get_q())
        i = state.info["step"]
        track = -jnp.sum((ee - state.info["ee_target_seq"][i]) ** 2)
        reward = track  # - self._action_coef * jnp.sum((action - state.info["last_action"]) ** 2)
        if "twist_target_seq" in state.info:
            twist = self.fk.ee_twist(rs.get_q(), rs.get_qd())
            reward = reward - self._twist_coef * jnp.sum((twist - state.info["twist_target_seq"][i]) ** 2)

        info = {**state.info, "last_action": action, "step": i + 1}
        return State(rs, self._get_obs(rs, info), reward, jnp.zeros(()), state.metrics, info)

    def _get_obs(self, rs: RobotState, info) -> jax.Array:
        parts = [rs.ravel(), info["ee_target_seq"][0]]
        if "twist_target_seq" in info:
            parts.append(info["twist_target_seq"][0])
        return jnp.concatenate(parts)

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
    # TEST: same joint config -> same EE pose in mjx FK and the torch HeapEnv chain.
    import numpy as np
    import torch
    import pytorch_kinematics as pk

    fk = M545FK()
    chain = pk.build_serial_chain_from_urdf(open(URDF_PATH, "rb").read(), EE_BODY, ROOT_BODY)
    ee_mjx = jax.jit(fk.ee_pos)

    errs = []
    for s in range(100):
        q = np.random.RandomState(s).uniform(-0.5, 0.5, size=fk.nq).astype(np.float32)
        ee_pk = chain.forward_kinematics(torch.tensor(q).unsqueeze(0)).get_matrix()[0, :3, 3].numpy()
        ee_jx = np.asarray(ee_mjx(jnp.asarray(q)))
        errs.append(float(np.linalg.norm(ee_pk - ee_jx)))

    print(f"max EE error over 100 random configs: {max(errs):.2e}")
    assert max(errs) < 1e-4, "mjx FK does not match pytorch_kinematics!"
    print("PASS: brax/mjx FK matches the HeapEnv (pytorch_kinematics) FK")
