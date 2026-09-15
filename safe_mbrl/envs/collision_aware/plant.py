"""Identified valve-to-joint dynamics of the real M445 hammer.

The plant is the probabilistic ensemble trained online on the machine
(real_5dof_hammer_fast_experiment, episode_073; copied to ./model). Every tick
it maps the q / qd / valve-command history to the next joint velocity, and the
joint position is integrated from it. No physics runs anywhere.

Actions are the pre-polarity normalized valve commands in [-1, 1] the model was
trained on (+ means q increases on every joint). Sampling picks how the
ensemble's predictive distribution is used: "mean" (deterministic nominal
plant), "member" (a random ensemble member, epistemic only) or "normal"
(member + aleatoric noise, the most realistic).
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orbax
from flax import nnx

from safe_mbrl.models.robot_ensemble import RobotEnsemble
from safe_mbrl.utils.structs import RobotState

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


def load_ensemble(model_dir: str = MODEL_DIR) -> RobotEnsemble:
    """RobotEnsemble with the orbax weights restored into its inner nnx module."""
    with open(os.path.join(model_dir, "config.json")) as f:
        ensemble = RobotEnsemble(**json.load(f))
    _, _, state = nnx.split(ensemble.model, nnx.RngState, ...)
    sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    restore_args = jax.tree.map(lambda _: orbax.ArrayRestoreArgs(sharding=sharding), state)
    state = orbax.PyTreeCheckpointer().restore(os.path.join(model_dir, "state"), item=state,
                                               restore_args=restore_args)
    nnx.update(ensemble.model, state)
    return ensemble


class IdentifiedPlant:
    """Stateful wrapper: reset(q0) then step(action) -> (q, qd) at the model's dt."""

    def __init__(self, q_min, q_max, model_dir: str = MODEL_DIR, sampling: str = "mean", seed: int = 0):
        if sampling not in ("mean", "member", "normal"):
            raise ValueError(f"sampling must be mean|member|normal, got {sampling!r}")
        self.ensemble = load_ensemble(model_dir)
        self.joint_dim, self.buffer_dim, self.dt = (self.ensemble.joint_dim, self.ensemble.buffer_dim,
                                                    self.ensemble._dt)
        self.q_min, self.q_max = jnp.asarray(q_min, jnp.float32), jnp.asarray(q_max, jnp.float32)
        self._key = jax.random.key(seed)
        self._graphdef, self._params = nnx.split(self.ensemble.model)
        self._step = jax.jit(self._make_step(sampling))
        self.state: RobotState | None = None

    def _make_step(self, sampling: str):
        jd, dt, mode = self.joint_dim, self.dt, self.ensemble.mode
        input_idx, n_members = self.ensemble._input_idx, self.ensemble.model.num_ensembles
        q_min, q_max = self.q_min, self.q_max

        def step(params, rs: RobotState, action, key):
            net = nnx.merge(self._graphdef, params)
            x = rs.ravel() if input_idx is None else rs.ravel()[input_idx]
            mu, sigma = jnp.split(net(x), 2, axis=-1)                    # (members, jd) each
            k_member, k_noise = jax.random.split(key)
            m = jax.random.randint(k_member, (), 0, n_members)
            if sampling == "mean":
                v = mu.mean(axis=0)
            elif sampling == "member":
                v = mu[m]
            else:
                v = mu[m] + sigma[m] * jax.random.normal(k_noise, (jd,))

            qd = rs.get_qd() + v if mode == "dv" else v
            q = jnp.clip(rs.get_q() + qd * dt, q_min, q_max)
            qd = (q - rs.get_q()) / dt                                   # velocity after the clamp
            return rs.replace(q_buffer=jnp.roll(rs.q_buffer, -jd).at[-jd:].set(q),
                              qd_buffer=jnp.roll(rs.qd_buffer, -jd).at[-jd:].set(qd),
                              act_buffer=jnp.roll(rs.act_buffer, -jd).at[-jd:].set(action))
        return step

    def reset(self, q0) -> None:
        """Start at rest at q0 with an empty command history."""
        self.state = RobotState.create(jnp.asarray(q0, jnp.float32), buffer_size=self.buffer_dim,
                                       q_dim=self.joint_dim)

    def step(self, action) -> tuple[np.ndarray, np.ndarray]:
        action = jnp.clip(jnp.asarray(action, jnp.float32), -1.0, 1.0)
        self._key, key = jax.random.split(self._key)
        self.state = self._step(self._params, self.state, action, key)
        return self.q, self.qd

    @property
    def q(self) -> np.ndarray:
        return np.asarray(self.state.get_q())

    @property
    def qd(self) -> np.ndarray:
        return np.asarray(self.state.get_qd())
