"""Consistency test for the brax planning env (heap_m545).

Question: is the env's reward the *correct* negative EE tracking error, so that the
best-tracking trajectory is the best-reward trajectory? `reward = -||ee - target||²`
by definition, but that's only meaningful if (a) the target is aligned to the right
rollout step, (b) the FK is right, and (c) the sign/sum are right.

Method: roll K random action sequences through the env (recording summed reward + the
predicted joint trajectory), then INDEPENDENTLY recompute the tracking cost with
pytorch_kinematics (a different FK than the env's mjx) against the SAME target window,
and check  -reward == cost  and that ranking-by-reward == ranking-by-(-cost).
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch
import jax
import jax.numpy as jnp
from types import SimpleNamespace
import pytorch_kinematics as pk

from safe_mbrl.models.robot_ensemble import RobotEnsemble
from safe_mbrl.envs.heap_m545 import HeapEnv as PlanEnv, URDF_PATH, EE_BODY, ROOT_BODY
from safe_mbrl.envs.base import State
from safe_mbrl.safe_mbrl.utils.structs import RobotState

jd, bd, H, K = 4, 10, 20, 64
cfg = SimpleNamespace(action_penalty_coef=0.0)        # pure tracking reward
model = RobotEnsemble(joint_dim=jd, buffer_dim=bd, model_type="PE", mode="dv", num_ensembles=5)
penv = PlanEnv(model, cfg=cfg)

# An arbitrary EE reference window (reachable joint targets FK'd to EE).
key = jax.random.key(0)
lo, hi = penv.fk.jnt_range[:, 0], penv.fk.jnt_range[:, 1]
qrefs = jax.random.uniform(key, (H, jd), minval=lo, maxval=hi)
ee_target_seq = np.stack([np.asarray(penv.fk.ee_pos(q)) for q in qrefs])     # (H, 3)

# Initial planning state.
z = jnp.zeros(jd * bd)
rs = RobotState(q_buffer=z, qd_buffer=z, act_buffer=z, q_dim=jd)
info = {"ee_target_seq": jnp.asarray(ee_target_seq), "last_action": jnp.zeros(jd),
        "step": jnp.zeros((), jnp.int32), "params": penv._params}
st0 = State(rs, penv._get_obs(rs, info), jnp.zeros(()), jnp.zeros(()), {"reward": jnp.zeros(())}, info)


@jax.jit
def rollout(acts):
    def body(s, a):
        s = penv.step(s, a)
        return s, (s.reward, s.pipeline_state.get_q())
    _, (rews, qs) = jax.lax.scan(body, st0, acts)
    return rews.sum(), qs                                                    # scalar, (H, jd)


chain = pk.build_serial_chain_from_urdf(open(URDF_PATH, "rb").read(), EE_BODY, ROOT_BODY)

def independent_cost(qs):
    """Tracking cost recomputed with pytorch_kinematics on the rolled joint trajectory."""
    ee = chain.forward_kinematics(torch.tensor(np.asarray(qs))).get_matrix()[:, :3, 3].numpy()  # (H, 3)
    return float(np.sum((ee - ee_target_seq) ** 2))


rng = np.random.RandomState(0)
env_reward, indep = [], []
for _ in range(K):
    acts = jnp.asarray(np.clip(rng.randn(H, jd) * 0.5, -1.0, 1.0).astype(np.float32))
    r, qs = rollout(acts); r.block_until_ready()
    env_reward.append(float(r))
    indep.append(independent_cost(qs))
env_reward, indep = np.array(env_reward), np.array(indep)

print(f"K={K} random action sequences, horizon={H}")
print(f"max |(-reward) - independent_cost| : {np.max(np.abs(-env_reward - indep)):.3e}")
print(f"corr( reward , -cost )             : {np.corrcoef(env_reward, -indep)[0, 1]:.6f}")
same = np.argmax(env_reward) == np.argmin(indep)
print(f"best-reward seq == best-tracking seq: {'SAME' if same else 'DIFFER'} "
      f"(reward argmax={np.argmax(env_reward)}, cost argmin={np.argmin(indep)})")
print("CONSISTENT" if np.max(np.abs(-env_reward - indep)) < 1e-3 and same
      else "INCONSISTENT — reward does not match independent tracking cost!")
