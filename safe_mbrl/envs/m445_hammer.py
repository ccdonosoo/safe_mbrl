"""HeapEnv on the M445 HAMMER URDF: 5-DOF, EE in BASE frame, safe-box range."""
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from safe_mbrl.envs.heap_m445 import HeapEnv, _quat2mat
from safe_mbrl.m445_hammer_spec import (
    EE_BODY, HAMMER_URDF, SAFE_Q_MAX, SAFE_Q_MIN, STRIKE_AXIS)

ROOT_BODY = "BASE"


class M445HammerFK:
    """mjx FK: q (5,) -> EE quantities in the BASE frame."""

    def __init__(self, urdf_path: str = HAMMER_URDF):
        root = ET.parse(urdf_path).getroot()
        for link in root.findall("link"):
            for tag in ("visual", "collision"):
                for e in link.findall(tag):
                    link.remove(e)
        # J_TURN is continuous here (shovel URDF has it fixed): keep <axis>,
        # rewrite type + add a limit; the planning constraint is the safe box.
        j = root.find("joint[@name='J_TURN']")
        j.set("type", "revolute")
        ET.SubElement(j, "limit", {"lower": "-3.1416", "upper": "3.1416",
                                   "effort": "1e6", "velocity": "10"})
        ET.SubElement(ET.SubElement(root, "mujoco"), "compiler",
                      {"fusestatic": "false", "balanceinertia": "true"})
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

        self._mx = mjx.put_model(m)
        self._ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
        self._root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
        self._axis_local = jnp.asarray(STRIKE_AXIS)
        self.jnt_range = jnp.asarray(m.jnt_range)
        self.nq = m.nq

    def _kin(self, q):
        d = mjx.make_data(self._mx).replace(qpos=q)
        return mjx.kinematics(self._mx, d)

    def ee_pos(self, q):
        d = self._kin(q)
        Rt = _quat2mat(d.xquat[self._root]).T
        return Rt @ (d.xpos[self._ee] - d.xpos[self._root])

    def ee_pose(self, q):
        d = self._kin(q)
        Rt = _quat2mat(d.xquat[self._root]).T
        pos = Rt @ (d.xpos[self._ee] - d.xpos[self._root])
        return pos, Rt @ _quat2mat(d.xquat[self._ee])

    def ee_axis(self, q):
        _, R = self.ee_pose(q)
        return R @ self._axis_local

    def ee_twist(self, q, qd):
        d = self._kin(q)
        d = mjx.com_pos(self._mx, d)
        jacp, jacr = mjx.jac(self._mx, d, d.xpos[self._ee], self._ee)
        Rt = _quat2mat(d.xquat[self._root]).T
        return jnp.concatenate([Rt @ (qd @ jacp), Rt @ (qd @ jacr)])


class HammerEnv(HeapEnv):
    """HeapEnv with the hammer FK and the safe box as joint range."""

    def __init__(self, model, cfg=None):
        if model.joint_dim != 5:
            raise ValueError(f"HammerEnv is 5-DOF only, got {model.joint_dim}")
        super().__init__(model, cfg)
        self.fk = M445HammerFK()
        self._pos_limit = jnp.stack(
            [jnp.asarray(SAFE_Q_MIN), jnp.asarray(SAFE_Q_MAX)], axis=1)
        self._ee_ref_fk = jax.jit(jax.vmap(self.fk.ee_pos))


def make_fk_batch(fk: M445HammerFK):
    """(fk_pos_axis, fk_pos): jitted+vmapped, numpy in/out."""
    pos_axis = jax.jit(jax.vmap(lambda q: (fk.ee_pos(q), fk.ee_axis(q))))
    pos_only = jax.jit(jax.vmap(fk.ee_pos))

    def fk_pos_axis(q):
        p, a = pos_axis(jnp.asarray(q))
        return jax.device_get(p), jax.device_get(a)

    def fk_pos(q):
        return jax.device_get(pos_only(jnp.asarray(q)))

    return fk_pos_axis, fk_pos
