""" Minimal online MBRL with MPPI on HeapEnv: joint pos (w=8) + EE pos (w=2), EMA actions, no exploration. """
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import time
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from safe_mbrl.envs.heap_env.heap_example_env import HeapEnv as SimEnv
from safe_mbrl.envs.heap_m545 import HeapEnv as PlanEnv
from safe_mbrl.models.robot_ensemble import RobotEnsemble
from safe_mbrl.models.online_trainer import OnlineTrainer
from safe_mbrl.utils.structs import Dataset
from safe_mbrl.mpc.mppi import MPPI

JD, BD = 4, 15
H = 30
EMA = 0.18
N_TRAJ, N_ROUNDS = 10, 300


def make_plan_fn(penv, mpc):
    @jax.jit
    def plan(state, init_actions, rng):
        def sum_rewards(seq):
            def body(s, a):
                s = penv.step(s, a)
                return s, s.reward
            return jax.lax.scan(body, state, seq)[1].sum()
        return mpc.optimize(sum_rewards, rng, init_actions, (-1.0, 1.0))
    return plan


def ref_window(seg, s):
    idx = np.minimum(np.arange(s + 1, s + 1 + H), len(seg) - 1)
    return seg[idx]


def run_episode(sim, penv, plan_fn, rng):
    q0, _ = sim.cur_joints()
    q_buf = np.tile(q0, BD).astype(np.float32)
    qd_buf = np.zeros(JD * BD, np.float32)
    act_buf = np.zeros(JD * BD, np.float32)
    a_prev = np.zeros(JD, np.float32)
    init = jnp.zeros((H, JD))

    rows, ptimes, errs, ep_reward, start_q = [], [], [], 0.0, None
    for traj_i in range(N_TRAJ):
        sim.reset(start_q=start_q, reset_model=traj_i == 0)
        seg = sim.ref_traj_joint[0].T.cpu().numpy()               # (ref_steps, JD)
        sim.render()

        for s in range(sim.ref_traj_steps - 1):
            st = penv.make_traj_state(q_buf, qd_buf, act_buf, ref_window(seg, s), JD,
                                      last_action=a_prev)
            rng, k = jax.random.split(rng)
            t0 = time.perf_counter()
            aseq, _ = plan_fn(st, init, k)
            aseq.block_until_ready()
            ptimes.append(time.perf_counter() - t0)
            init = jnp.concatenate([aseq[1:], aseq[-1:]], axis=0)
            # same EMA filter the planner models in its rollout
            a0 = (EMA * np.asarray(aseq[0]) + (1.0 - EMA) * a_prev).astype(np.float32)
            a_prev = a0

            _, rwd, _, _, _ = sim.step(a0[None, :])
            sim.render(done=(s == sim.ref_traj_steps - 2))

            ep_reward += float(np.asarray(rwd).reshape(-1)[0])
            errs.append(float(np.linalg.norm(sim.ee_pos.squeeze(0).cpu().numpy()
                                             - sim.ref_traj_eepos[0, sim.current_step].cpu().numpy())))
            q_t, qd_t = sim.cur_joints()
            q_buf = np.roll(q_buf, -JD); q_buf[-JD:] = q_t
            qd_buf = np.roll(qd_buf, -JD); qd_buf[-JD:] = qd_t
            act_buf = np.roll(act_buf, -JD); act_buf[-JD:] = a0
            rows.append(np.concatenate([q_buf, qd_buf, act_buf]))

        start_q = q_t

    rows = np.asarray(rows, np.float32)
    ds = Dataset(input=jnp.asarray(rows), target=jnp.asarray(np.zeros((len(rows), JD), np.float32)))
    return ds, ptimes, float(np.mean(errs)), ep_reward, rng


def plot_metrics(hist, path):
    fig, axes = plt.subplots(1, len(hist), figsize=(4 * len(hist), 3))
    for ax, (k, v) in zip(axes, hist.items()):
        ax.plot(v, marker=".")
        ax.set_title(k)
        ax.set_xlabel("round")
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    cfg = SimpleNamespace(step_penalty_coef=0,
                          action_penalty_coef=0.01,
                          accel_penalty_coef=0,
                          accel_sign_penalty_coef=0,
                          track_mode="joint",
                          joint_weight=8.0,
                          ee_weight=2.0,
                          action_ema_alpha=EMA)

    sim = SimEnv(n_envs=1, use_act_net=True, n_history_steps=BD,
                 n_ref_steps=H, t_step=0.04, t_traj=6.0, cfg=cfg)
    model = RobotEnsemble(joint_dim=JD, buffer_dim=BD, model_type="PE", mode="v",
                          features=(256, 256), num_ensembles=5)
    penv = PlanEnv(model, cfg=cfg)
    mppi = MPPI(penv, horizon=H, nb_samples=2000, temperature=0.05, init_std=0.5, nb_steps=3)
    plan_fn = make_plan_fn(penv, mppi)

    train_ds = []
    trainer = OnlineTrainer(model, optax.adamw(1e-4, weight_decay=1e-4), train_ds, [],
                            batch_size=128, horizon=10, nb_epochs=3)

    rng = jax.random.key(0)

    plot_dir = os.path.join(os.path.dirname(__file__), "episode_plots")
    os.makedirs(plot_dir, exist_ok=True)
    hist_lists = (sim.predicted_trajectories, sim.predicted_velocities,
                  sim.joint_pos_histories, sim.joint_vel_histories, sim.reward_histories,
                  sim.target_trajectories, sim.target_joint_pos_histories)
    hist = {"tracking err [m]": [], "episode reward": [], "train NLL": []}

    for r in range(N_ROUNDS):
        for lst in hist_lists:
            lst.clear()
        ds, ptimes, err, ep_reward, rng = run_episode(sim, penv, plan_fn, rng)
        sim.render(mode="plot", save_dir=plot_dir, file_name=f"episode_{r:02d}")
        train_ds.append(ds)
        nll = float(trainer.train_model_bptt(seed=r + 1, verbose=False))
        penv._params = nnx.split(model.model)[1]

        hist["tracking err [m]"].append(err)
        hist["episode reward"].append(ep_reward)
        hist["train NLL"].append(nll)
        plot_metrics(hist, os.path.join(plot_dir, "metrics.png"))

        print(f"Round {r}: MPPI {np.mean(ptimes) * 1000:.2f} ms/step ({1.0 / np.mean(ptimes):.0f} Hz) "
              f"| reward {ep_reward:.2f} | tracking err {err:.3f} m | train NLL {nll:.3f} "
              f"| train steps {sum(len(d) for d in train_ds)}")


if __name__ == "__main__":
    main()
