""" Online MBRL with MPPI on HeapEnv. """
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
from safe_mbrl.utils.structs import RobotState, Dataset
from safe_mbrl.mpc.icem import ICEM


class ListLogger:
    """Captures OnlineTrainer's per-epoch scalars so we can read the latest NLL."""
    def __init__(self):
        self.data = {}

    def scalar_summary(self, tag, value, step):
        self.data.setdefault(tag, []).append((int(step), float(value)))


def make_plan_fn(penv, mpc):
    """Jitted MPC planner: (plan_state, warm_start_actions, rng) -> (action_seq, value)."""
    @jax.jit
    def plan(state, init_actions, rng):
        def sum_rewards(seq):                                  # (H, jd) -> scalar
            def body(s, a):
                s = penv.step(s, a)
                return s, s.reward
            return jax.lax.scan(body, state, seq)[1].sum()
        return mpc.optimize(sum_rewards, rng, init_actions, (-1.0, 1.0))
    return plan


def ref_window(seg, s, horizon):
    """ This just give the reference window, for the mpc, with tiling at the end"""
    idx = np.minimum(np.arange(s + 1, s + 1 + horizon), len(seg) - 1)
    return seg[idx]


def build_refs(sim, penv, jd):
    """CABIN-frame reference sequences for the CURRENT sim.ref_traj_joint, matched to
    penv._track_mode -> (target_seq, aux_seq), each (ref_steps, ...), indexed along time:
      "joint" -> joint-pos (ref_steps, jd),  joint-vel (ref_steps, jd)
      "pose"  -> EE pose   (ref_steps, 4, 4), EE twist  (ref_steps, 6)
    The EE pose + twist go through penv.fk so they live in the SAME frame the env measures
    in (NOT the world-frame sim.ref_traj_Tee, which caused the original concat crash)."""
    q_ref = jnp.asarray(sim.ref_traj_joint[0].T.cpu().numpy())          # (ref_steps, jd)
    # EXACT reference joint velocity (no finite differencing). Each joint's reference is a
    # single rest-to-rest quintic (min-jerk): q(tau) = q0 + (q1-q0)(10 tau^3 - 15 tau^4 +
    # 6 tau^5), tau = t / t_traj in [0,1]. Its analytic derivative is
    #   qd(t) = (q1 - q0) / t_traj * 30 tau^2 (1 - tau)^2   (zero at both endpoints).
    tau = jnp.linspace(0.0, 1.0, q_ref.shape[0])[:, None]               # (ref_steps, 1)
    qd_ref = (q_ref[-1] - q_ref[0]) * (30.0 * tau ** 2 * (1.0 - tau) ** 2) / sim.t_traj
    if penv._track_mode == "joint":
        return np.asarray(q_ref), np.asarray(qd_ref)
    pos, R = jax.vmap(penv.fk.ee_pose)(q_ref)                           # (T,3), (T,3,3)
    Tee = jnp.broadcast_to(jnp.eye(4), (q_ref.shape[0], 4, 4))
    Tee = Tee.at[:, :3, :3].set(R).at[:, :3, 3].set(pos)               # (ref_steps, 4, 4)
    twist = jax.vmap(penv.fk.ee_twist)(q_ref, qd_ref)                  # (ref_steps, 6) exact SE(3) twist
    return np.asarray(Tee), np.asarray(twist)


def run_episode(sim, penv, plan_fn, jd, bd, horizon, rng, n_traj, random=False, render=False):

    
    # Buffer of model initialization
    q0, _ = sim.cur_joints()
    q_buf = np.tile(q0, bd).astype(np.float32)
    qd_buf = np.zeros(jd * bd, np.float32)
    act_buf = np.zeros(jd * bd, np.float32)
    init = jnp.zeros((horizon, jd))
    steps_per_traj = sim.ref_traj_steps - 1
    
    start_q, reset_model = None, True

    # Logging
    rows, ptimes, errs, ep_reward = [], [], [], 0.0
    for traj_i in range(n_traj):
        reset_model = True if traj_i == 0 else False
        sim.reset(start_q=start_q, reset_model=reset_model)
        seg, seg_aux = build_refs(sim, penv, jd)   # CABIN-frame refs matched to penv._track_mode

        if render:
            sim.render()

        for s in range(steps_per_traj):

            if random:
                a0 = np.clip(0.5 * np.random.randn(jd), -1.0, 1.0).astype(np.float32)
            else:
                st = penv.make_traj_state(q_buf,
                                          qd_buf,
                                          act_buf,
                                          ref_window(seg, s, horizon),
                                          jd,
                                          ref_window(seg_aux, s, horizon))
                
                # mpc inference step
                rng, k = jax.random.split(rng)
                t0 = time.perf_counter()
                aseq, _ = plan_fn(st, init, k)
                aseq.block_until_ready()
                ptimes.append(time.perf_counter() - t0)
                a0 = np.asarray(aseq[0])
                init = jnp.concatenate([aseq[1:], aseq[-1:]], axis=0)           

            # heap env plant simulation step
            _, rwd, _, _, _ = sim.step(a0[None, :])
            
            if render:
                sim.render(done=(s == steps_per_traj - 1))     
            
            # logging
            ep_reward += float(np.asarray(rwd).reshape(-1)[0])
            # continuous per-step EE tracking error (m), mode-independent -> real "tracking err"
            ee_now = sim.ee_pos.squeeze(0).cpu().numpy()
            ee_ref = sim.ref_traj_eepos[0, sim.current_step].cpu().numpy()
            errs.append(float(np.linalg.norm(ee_now - ee_ref)))
            q_t, qd_t = sim.cur_joints()
            q_buf = np.roll(q_buf, -jd); q_buf[-jd:] = q_t
            qd_buf = np.roll(qd_buf, -jd); qd_buf[-jd:] = qd_t
            act_buf = np.roll(act_buf, -jd); act_buf[-jd:] = a0
            rows.append(np.concatenate([q_buf, qd_buf, act_buf]))

        start_q = q_t
        
    rows = np.asarray(rows, np.float32)
    ds = Dataset(input=jnp.asarray(rows), target=jnp.asarray(np.zeros((len(rows), jd), np.float32)))
    return ds, ptimes, float(np.mean(errs)), ep_reward, rng


def main():
    jd, bd = 4, 15
    H_PLAN, N_SAMPLES = 15, 2000
    N_TRAJ, WARMUP_TRAJ, N_ROUNDS = 10, 2, 300      # episode = 10 chained refs x t_traj(6s) = 60 s

    cfg = SimpleNamespace(step_penalty_coef=0,
                          action_penalty_coef=0.01,
                          accel_penalty_coef=0,
                          accel_sign_penalty_coef=0,
                          track_mode="joint",
                          qd_coef=1.0)   # MPPI tracks joint pos + vel
    
    sim = SimEnv(n_envs=1, use_act_net=True, n_history_steps=bd,
                 n_ref_steps=15, t_step=0.04, t_traj=6.0, cfg=cfg)
    
    model = RobotEnsemble(joint_dim=jd,
                          buffer_dim=bd,
                          model_type="PE",
                          mode="v",
                          features=(256, 256),
                          num_ensembles=5)
    
    penv = PlanEnv(model, cfg=cfg)
    
    mppi = ICEM(penv, horizon=H_PLAN,
                nb_samples=500, nb_steps=5, nb_elites=150, exponent=1, alpha=0.3)
    
    plan_fn = make_plan_fn(penv, mppi)

    logger = ListLogger()
    
    train_ds, val_ds = [], []
    
    trainer = OnlineTrainer(model,
                            optax.adamw(1e-4, weight_decay=1e-4),
                            train_ds,
                            val_ds,
                            batch_size=128,
                            horizon=10,
                            horizon_val=10,
                            nb_epochs=3,
                            early_stopping_patience=10_000,
                            logger=logger)
    
    rng = jax.random.key(0)


    print("=== warmup: 2 random episodes for validation data ===")
    
    for _ in range(2):
        ds, _, _, _, rng = run_episode(sim, penv, plan_fn, jd, bd, H_PLAN, rng, WARMUP_TRAJ, random=True)
        train_ds.append(ds)
        
    val_ds.append(train_ds.pop())

    penv._params = nnx.split(model.model)[1]

    zeros_buffers = np.zeros(jd * bd, np.float32)

    seg0, seg0_aux = build_refs(sim, penv, jd)   # mode-correct dummy window to trigger JIT
    plan_fn(penv.make_traj_state(zeros_buffers,
                            zeros_buffers,
                            zeros_buffers,
                            ref_window(seg0, 0, H_PLAN),
                            jd,
                            ref_window(seg0_aux, 0, H_PLAN)),
            jnp.zeros((H_PLAN, jd)), rng)[0].block_until_ready()

    plot_dir = os.path.join(os.path.dirname(__file__), "episode_plots")
    
    os.makedirs(plot_dir, exist_ok=True)
    
    hist_lists = (sim.predicted_trajectories,
                  sim.predicted_velocities,
                  sim.joint_pos_histories,
                  sim.joint_vel_histories,
                  sim.reward_histories,
                  sim.target_trajectories,
                  sim.target_joint_pos_histories)

    all_ptimes = []

    for r in range(N_ROUNDS):
        
        # logging 
        for lst in hist_lists:
            lst.clear()    # only this episode's sub-trajectories
        ds, ptimes, err, ep_reward, rng = run_episode(sim, penv, plan_fn, jd, bd, H_PLAN, rng, N_TRAJ, render=True)
        sim.render(mode="plot", save_dir=plot_dir, file_name=f"episode_{r:02d}")   # followed vs desired EE
        train_ds.append(ds)
        
        # Training
        val_mse = float(trainer.train_model_bptt(seed=r + 1, verbose=False))   
        val_nll = logger.data["train/val_loss"][-1][1]                        
        
        # Update of model state params
        penv._params = nnx.split(model.model)[1]                
        
        all_ptimes += ptimes
        
        
        print(f"Round {r}: {N_TRAJ} refs / {len(ptimes)} steps (60s) | MPPI {np.mean(ptimes) * 1000:.2f} ms/step "
              f"({1.0 / np.mean(ptimes):.0f} Hz) | episode reward {ep_reward:.2f} "
              f"| tracking err {err:.3f} m | model NLL {val_nll:.3f} MSE {val_mse:.2e} "
              f"| train steps {sum(len(d) for d in train_ds)}")


if __name__ == "__main__":
    main()
