"""Collect an offline dataset with online MPPI-MBRL on HeapEnv.

Runs N_ROUNDS episodes; each episode tracks the same objective (the reference EE
trajectory) with MPPI through the learned model, executes in the real torch sim,
and trains the dynamics model online so the planner improves round to round.

The point of this script is the DATA, not the model: at the end it does NOT save the
model — it dumps the full collection of per-episode (q, qd, act) rollout datasets to
a .npz file, so a fresh model can be trained offline (BPTT) on the same data later.
Use `load_datasets(path, jd)` to rebuild the list[Dataset] for the offline trainer.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import time
from types import SimpleNamespace

import numpy as np
import torch
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from safe_mbrl.envs.heap_env.heap_example_env import HeapEnv as SimEnv      # real simulator (torch)
from safe_mbrl.envs.heap_m545 import HeapEnv as PlanEnv                     # learned-model env (mjx FK)
from safe_mbrl.envs.base import State
from safe_mbrl.models.robot_ensemble import RobotEnsemble
from safe_mbrl.models.online_trainer import OnlineTrainer
from safe_mbrl.utils.type_aliases import RobotState, Dataset
from safe_mbrl.mpc.mppi import MPPI


class ListLogger:
    """Captures OnlineTrainer's per-epoch scalars so we can read the latest NLL."""
    def __init__(self):
        self.data = {}

    def scalar_summary(self, tag, value, step):
        self.data.setdefault(tag, []).append((int(step), float(value)))


def save_datasets(train_ds, val_ds, path):
    """Dump the collected per-episode datasets for offline BPTT training. Each row is a
    ravel([q, qd, act]) input; targets are unused by BPTT. We store concatenated inputs
    plus per-episode lengths so episode boundaries are preserved (rollout windows must
    not cross them)."""
    def pack(ds_list):
        inputs = np.concatenate([np.asarray(d.input, np.float32) for d in ds_list], axis=0)
        lengths = np.array([len(d) for d in ds_list], dtype=np.int32)
        return inputs, lengths
    tr_in, tr_len = pack(train_ds)
    va_in, va_len = pack(val_ds)
    np.savez_compressed(path, train_inputs=tr_in, train_lengths=tr_len,
                        val_inputs=va_in, val_lengths=va_len)


def load_datasets(path, jd):
    """Rebuild (train_ds, val_ds) lists of Datasets from a saved .npz for offline training."""
    z = np.load(path)
    def unpack(inputs, lengths):
        out, off = [], 0
        for L in lengths:
            L = int(L)
            out.append(Dataset(input=jnp.asarray(inputs[off:off + L]),
                               target=jnp.zeros((L, jd), jnp.float32)))
            off += L
        return out
    return unpack(z["train_inputs"], z["train_lengths"]), unpack(z["val_inputs"], z["val_lengths"])


def make_plan_fn(penv, mppi):
    """Jitted MPPI planner: (plan_state, warm_start_actions, rng) -> (action_seq, value)."""
    @jax.jit
    def plan(state, init_actions, rng):
        def sum_rewards(seq):                                  # (H, jd) -> scalar
            def body(s, a):
                s = penv.step(s, a)
                return s, s.reward
            return jax.lax.scan(body, state, seq)[1].sum()
        return mppi.optimize(sum_rewards, rng, init_actions, (-1.0, 1.0))
    return plan


def plan_state(penv, q_buf, qd_buf, act_buf, ee_target_seq, jd):
    """Build a planning State from the simulator's current joint buffers + the EE
    reference window to track over the horizon (one row per planned step).
    Params are carried in info so step() can merge the model inside scan/vmap."""
    rs = RobotState(q_buffer=jnp.asarray(q_buf), qd_buffer=jnp.asarray(qd_buf),
                    act_buffer=jnp.asarray(act_buf), q_dim=jd)
    info = {"ee_target_seq": jnp.asarray(ee_target_seq), "last_action": jnp.zeros(jd),
            "step": jnp.zeros((), jnp.int32), "params": penv._params}
    z = jnp.zeros(())
    return State(rs, penv._get_obs(rs, info), z, z, {"reward": z}, info)


def ref_window(ee_ref, g, horizon):
    """The next `horizon` EE reference points from the FULL pre-generated episode
    reference (shifted +1 so plan-step i tracks the state after action i). Because
    `ee_ref` spans all sub-trajectories, the window looks AHEAD across sub-trajectory
    boundaries (the planner anticipates the next segment); it only pads at the very
    end of the episode."""
    idx = np.minimum(np.arange(g + 1, g + 1 + horizon), len(ee_ref) - 1)
    return ee_ref[idx]                                                   # (horizon, 3)


def cur_joints(sim):
    return (sim.dof_pos_history.squeeze(0).cpu().numpy()[:, 0],
            sim.dof_vel_history.squeeze(0).cpu().numpy()[:, 0])


def build_episode_ref(sim, n_traj):
    """Pre-generate the WHOLE episode reference up front: `n_traj` chained rest-to-rest
    quintics, each starting at the previous one's endpoint (sub-traj 0 is the one from
    reset). Returns the per-sub-traj joint segments (n_traj, no_dof, ref_steps) and the
    concatenated EE reference (T, 3), T = n_traj*(ref_steps-1) + 1. Having the full
    reference up front is what lets the look-ahead window cross sub-traj boundaries."""
    no_dof = sim.actnet.no_dof
    lo, hi = sim.actnet.pos_limit[:, 0], sim.actnet.pos_limit[:, 1]
    segs = [sim.ref_traj_joint[0].clone()]                              # (no_dof, ref_steps) from reset
    for _ in range(n_traj - 1):
        start01 = ((segs[-1][:, -1] - lo) / (hi - lo)).clamp(0.0, 1.0)
        goal01 = torch.rand(no_dof, device=sim.device)
        s01 = sim.generate_polynomial_traj(start01, goal01, 1.0, 0.0).reshape(no_dof, -1)
        segs.append(torch.einsum("jk,j->jk", s01, hi - lo) + lo[:, None])
    seg_joint = torch.stack(segs, dim=0)                               # (n_traj, no_dof, ref_steps)
    joint_cat = torch.cat([segs[0]] + [s[:, 1:] for s in segs[1:]], dim=1)            # drop shared starts
    ee_cat = sim.kinematics.forward_kinematics(joint_cat.T).get_matrix()[:, :3, 3]    # (T, 3)
    return seg_joint, ee_cat.cpu().numpy()


def set_segment(sim, seg_joint, k):
    """Point the env's per-sub-traj reference (used by render + the sim's heap reward)
    at pre-generated segment k. Resets the env step counter; keeps the plant state."""
    sim.ref_traj_joint = seg_joint[k:k + 1].clone()                    # (1, no_dof, ref_steps)
    Tee = sim.kinematics.forward_kinematics(
        sim.ref_traj_joint.transpose(1, 2).reshape(-1, sim.actnet.no_dof)).get_matrix()
    sim.ref_traj_eepos = Tee.reshape(1, sim.ref_traj_steps, 4, 4)[:, :, :3, 3]
    sim.current_step = 0


def run_episode(sim, penv, plan_fn, jd, bd, horizon, rng, n_traj, random=False, render=False):
    """One episode = `n_traj` continuously-chained reference trajectories (each
    `t_traj` long). MPPI-controlled (or random for warmup).
    With render=True, each sub-trajectory is recorded as one env render session
    (followed + desired EE), so the heap env's plot() draws all `n_traj` of them.
    Returns (Dataset, plan_times, mean_tracking_error, episode_reward, rng)."""
    sim.reset()
    seg_joint, ee_ref = build_episode_ref(sim, n_traj)          # full chained reference (look-ahead spans segments)
    q0, _ = cur_joints(sim)
    q_buf = np.tile(q0, bd).astype(np.float32)
    qd_buf = np.zeros(jd * bd, np.float32)
    act_buf = np.zeros(jd * bd, np.float32)
    init = jnp.zeros((horizon, jd))
    steps_per_traj = sim.ref_traj_steps - 1

    rows, ptimes, errs, ep_reward, g = [], [], [], 0.0, 0
    for traj_i in range(n_traj):
        set_segment(sim, seg_joint, traj_i)                     # env reference for render + sim reward
        if render:
            sim.render()                                        # seed a render session for this sub-trajectory
        for s in range(steps_per_traj):
            cur_wp = ee_ref[g]                                  # global waypoint for the error metric

            if random:
                a0 = np.clip(0.5 * np.random.randn(jd), -1.0, 1.0).astype(np.float32)
            else:
                st = plan_state(penv, q_buf, qd_buf, act_buf, ref_window(ee_ref, g, horizon), jd)  # cross-boundary window
                rng, k = jax.random.split(rng)
                t0 = time.perf_counter()
                aseq, _ = plan_fn(st, init, k)
                aseq.block_until_ready()
                ptimes.append(time.perf_counter() - t0)
                a0 = np.asarray(aseq[0])
                init = jnp.concatenate([aseq[1:], aseq[-1:]], axis=0)           # warm-start next plan

            _, rwd, _, _, _ = sim.step(a0[None, :])
            if render:
                sim.render(done=(s == steps_per_traj - 1))      # append EE; flush sub-trajectory at the end
            ep_reward += float(np.asarray(rwd).reshape(-1)[0])                  # simulator reward (heap-style)
            q_t, qd_t = cur_joints(sim)
            errs.append(float(np.linalg.norm(sim.ee_pos.squeeze(0).cpu().numpy() - cur_wp)))
            q_buf = np.roll(q_buf, -jd); q_buf[-jd:] = q_t
            qd_buf = np.roll(qd_buf, -jd); qd_buf[-jd:] = qd_t
            act_buf = np.roll(act_buf, -jd); act_buf[-jd:] = a0
            rows.append(np.concatenate([q_buf, qd_buf, act_buf]))
            g += 1

    rows = np.asarray(rows, np.float32)
    ds = Dataset(input=jnp.asarray(rows), target=jnp.asarray(np.zeros((len(rows), jd), np.float32)))
    return ds, ptimes, float(np.mean(errs)), ep_reward, rng


def main():
    jd, bd = 4, 10
    H_PLAN, N_SAMPLES = 15, 3000
    N_TRAJ, WARMUP_TRAJ, N_ROUNDS = 10, 2, 100     # episode = 10 chained refs x t_traj(6s) = 60 s

    cfg = SimpleNamespace(step_penalty_coef=0, action_penalty_coef=1,
                          accel_penalty_coef=0, accel_sign_penalty_coef=0)
    sim = SimEnv(n_envs=1, use_act_net=True, n_history_steps=bd,
                 n_ref_steps=15, t_step=0.04, t_traj=6.0, cfg=cfg)
    model = RobotEnsemble(joint_dim=jd, buffer_dim=bd, model_type="PE", mode="dv", num_ensembles=5)
    penv = PlanEnv(model, cfg=cfg)
    mppi = MPPI(penv, horizon=H_PLAN, nb_samples=N_SAMPLES,
                temperature=0.02, init_std=0.15, nb_steps=1)
    plan_fn = make_plan_fn(penv, mppi)

    logger = ListLogger()
    train_ds, val_ds = [], []
    trainer = OnlineTrainer(model, optax.adam(1e-4), train_ds, val_ds,
                            batch_size=256, horizon=10, horizon_val=15,
                            nb_epochs=2, early_stopping_patience=10_000, logger=logger)
    rng = jax.random.key(0)

    # --- warmup: random episodes to seed the dynamics model before planning ---
    print("=== warmup: 2 random episodes + train ===")
    for _ in range(2):
        ds, _, _, _, rng = run_episode(sim, penv, plan_fn, jd, bd, H_PLAN, rng, WARMUP_TRAJ, random=True)
        train_ds.append(ds)
    val_ds.append(train_ds.pop())
    trainer.train_model_bptt(seed=0, verbose=False)
    penv._params = nnx.split(model.model)[1]

    # compile the planner once (excluded from the reported Hz)
    zeros = np.zeros(jd * bd, np.float32)
    plan_fn(plan_state(penv, zeros, zeros, zeros, np.zeros((H_PLAN, 3), np.float32), jd),
            jnp.zeros((H_PLAN, jd)), rng)[0].block_until_ready()

    # --- online MPPI loop: plan -> execute -> train, accumulating on-policy data ---
    print("\n=== online MPPI: plan -> execute -> train (collecting offline dataset) ===")
    for r in range(N_ROUNDS):
        ds, ptimes, err, ep_reward, rng = run_episode(sim, penv, plan_fn, jd, bd, H_PLAN, rng, N_TRAJ)
        train_ds.append(ds)
        val_mse = float(trainer.train_model_bptt(seed=r + 1, verbose=False))   # best val MSE this round
        val_nll = logger.data["train/val_loss"][-1][1]                         # latest val NLL
        penv._params = nnx.split(model.model)[1]                # update planner with retrained model
        print(f"Round {r}: {N_TRAJ} refs / {len(ptimes)} steps (60s) | MPPI {np.mean(ptimes) * 1000:.2f} ms/step "
              f"| episode reward {ep_reward:.2f} | tracking err {err:.3f} m "
              f"| model NLL {val_nll:.3f} MSE {val_mse:.2e} | train steps {sum(len(d) for d in train_ds)}")

    # --- save ONLY the collected data (not the model) for offline training ---
    out = os.path.join(os.path.dirname(__file__), "heap_mppi_dataset.npz")
    save_datasets(train_ds, val_ds, out)
    n_tr = sum(len(d) for d in train_ds)
    n_va = sum(len(d) for d in val_ds)
    print(f"\n=== saved offline dataset -> {out} | {len(train_ds)} train eps ({n_tr} steps) "
          f"+ {len(val_ds)} val eps ({n_va} steps) ===")


if __name__ == "__main__":
    main()
