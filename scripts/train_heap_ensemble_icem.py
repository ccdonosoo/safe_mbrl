"""Online MBRL with ICEM on HeapEnv.

Loop per round:
  1. PLAN with ICEM through the learned RobotEnsemble (heap_m545 env, mjx FK) to
     track the simulator's reference EE trajectory.
  2. EXECUTE the first action in the real simulator (torch HeapEnv), collecting a
     dataset of (q, qd, act) rollouts.
  3. TRAIN the dynamics model (BPTT) on all collected data.
  4. UPDATE the planner with the retrained model, repeat.

The ICEM planner reuses ICEM.optimize with our own scan-based rollout `func`
(avoids icem.rollout_env's rl_algorithms_lib dependency). At the end we report the
ICEM policy inference time / control frequency.
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
from safe_mbrl.utils.structs import RobotState, Dataset
from safe_mbrl.mpc.icem import ICEM


class ListLogger:
    """Captures OnlineTrainer's per-epoch scalars so we can read the latest NLL."""
    def __init__(self):
        self.data = {}

    def scalar_summary(self, tag, value, step):
        self.data.setdefault(tag, []).append((int(step), float(value)))


def make_plan_fn(penv, planner):
    """Jitted planner: (plan_state, warm_start_actions, rng) -> (action_seq, value)."""
    @jax.jit
    def plan(state, init_actions, rng):
        def sum_rewards(seq):                                  # (H, jd) -> scalar
            def body(s, a):
                s = penv.step(s, a)
                return s, s.reward
            return jax.lax.scan(body, state, seq)[1].sum()
        return planner.optimize(sum_rewards, rng, init_actions, (-1.0, 1.0))
    return plan


def plan_state(penv, q_buf, qd_buf, act_buf, ee_target_seq, twist_target_seq, jd):
    """Build a planning State from the simulator's current joint buffers + the EE pose
    and EXACT twist reference windows to track over the horizon (one row per planned
    step). Params are carried in info so step() merges the model inside scan/vmap; the
    presence of `twist_target_seq` enables the env's twist-tracking term."""
    rs = RobotState(q_buffer=jnp.asarray(q_buf), qd_buffer=jnp.asarray(qd_buf),
                    act_buffer=jnp.asarray(act_buf), q_dim=jd)
    info = {"ee_target_seq": jnp.asarray(ee_target_seq), "twist_target_seq": jnp.asarray(twist_target_seq),
            "last_action": jnp.zeros(jd), "step": jnp.zeros((), jnp.int32), "params": penv._params}
    z = jnp.zeros(())
    return State(rs, penv._get_obs(rs, info), z, z, {"reward": z}, info)


def ref_window(seg_ee, seg_twist, s, horizon):
    """The next `horizon` EE position + EXACT twist references WITHIN the current segment
    (shifted +1 so plan-step i tracks the state after action i). Near the segment end the
    index clamps to the last point, so the windows PAD with the segment goal (a held pose
    -> zero twist) -- telling the planner to reach and HOLD the final pose. There is no
    cross-segment look-ahead: the next segment isn't generated until this one ends (it
    starts wherever the plant actually lands), so e.g. at step 145/150 with horizon 20 the
    window is 5 real points + 15 padded copies of the goal (H = 5 + N)."""
    idx = np.minimum(np.arange(s + 1, s + 1 + horizon), len(seg_ee) - 1)
    return seg_ee[idx], seg_twist[idx]


def cur_joints(sim):
    return (sim.dof_pos_history.squeeze(0).cpu().numpy()[:, 0],
            sim.dof_vel_history.squeeze(0).cpu().numpy()[:, 0])


def gen_segment(sim, penv, start_q, margin=0.05):
    """Generate ONE rest-to-rest quintic segment ON THE FLY, starting from the actual
    joint config `start_q` (no_dof,) the plant is currently in and ending at a fresh
    random goal sampled STRICTLY INSIDE the joint limits (a `margin` fraction of each
    joint's range is kept clear of both ends). Points the env's per-sub-traj reference
    (render + the sim's heap reward) at it and resets the step counter; keeps the plant
    state. Returns the segment's EE position reference (ref_steps, 3) and the EXACT SE(3)
    twist reference (ref_steps, 6) = J(q_ref) @ qd_ref (qd_ref a JOINT-space finite
    difference; the twist itself goes through the geometric Jacobian, no EE-pose
    interpolation) for the planner windows.

    Unlike the old precompute-everything-up-front scheme (each segment chained off the
    previous one's IDEALIZED endpoint), every segment now begins where the previous one
    ACTUALLY ended -- tracking error and all -- so the reference stays continuous with the
    real plant and never jumps back onto an idealized path.

    The margin matters: the real plant HARD-CLIPS joint position to pos_limit
    (MenziActNet.advance), so a reference that reaches a limit makes the plant clip there
    -> the joint velocity dies abruptly -> a corner/discontinuity in the followed EE path.
    Sampling goals in [margin, 1-margin] keeps the whole reference off that wall."""
    no_dof = sim.actnet.no_dof
    lo, hi = sim.actnet.pos_limit[:, 0], sim.actnet.pos_limit[:, 1]
    start01 = ((start_q - lo) / (hi - lo)).clamp(0.0, 1.0)
    goal01 = margin + (1.0 - 2.0 * margin) * torch.rand(no_dof, device=sim.device)   # strictly inside limits
    s01 = sim.generate_polynomial_traj(start01, goal01, 1.0, 0.0).reshape(no_dof, -1)
    seg = torch.einsum("jk,j->jk", s01, hi - lo) + lo[:, None]         # (no_dof, ref_steps)
    sim.ref_traj_joint = seg[None]                                     # (1, no_dof, ref_steps)
    Tee = sim.kinematics.forward_kinematics(
        sim.ref_traj_joint.transpose(1, 2).reshape(-1, no_dof)).get_matrix()
    sim.ref_traj_eepos = Tee.reshape(1, sim.ref_traj_steps, 4, 4)[:, :, :3, 3]
    sim.current_step = 0

    q_ref = jnp.asarray(seg.T.cpu().numpy())                          # (ref_steps, jd)
    qd_ref = jnp.concatenate([(q_ref[1:] - q_ref[:-1]) / sim.t_step, jnp.zeros((1, no_dof))], axis=0)
    ee_ref = jax.vmap(penv.fk.ee_pos)(q_ref)                           # (ref_steps, 3)
    twist_ref = jax.vmap(penv.fk.ee_twist)(q_ref, qd_ref)            # (ref_steps, 6) exact SE(3) twist
    return np.asarray(ee_ref), np.asarray(twist_ref)


def run_episode(sim, penv, plan_fn, jd, bd, horizon, rng, n_traj, random=False, render=False):
    """One episode = `n_traj` continuously-chained reference segments, each GENERATED ON
    THE FLY starting from where the plant actually is when the previous segment ends.
    Tracks EE pose + exact twist. With render=True, each sub-trajectory is one env render
    session. Returns (Dataset, plan_times, mean_tracking_error, episode_reward, rng)."""
    sim.reset()
    q0, _ = cur_joints(sim)
    q_buf = np.tile(q0, bd).astype(np.float32)
    qd_buf = np.zeros(jd * bd, np.float32)
    act_buf = np.zeros(jd * bd, np.float32)
    init = jnp.zeros((horizon, jd))
    steps_per_traj = sim.ref_traj_steps - 1

    rows, ptimes, errs, ep_reward = [], [], [], 0.0
    for traj_i in range(n_traj):
        # Each segment starts from the plant's CURRENT joint config (segment 0 from the
        # reset pose), so the reference is continuous with where the plant really ended up.
        seg_ee, seg_twist = gen_segment(sim, penv, sim.dof_pos_history[0, :, 0])
        if render:
            sim.render()                                        # seed a render session for this sub-trajectory
        for s in range(steps_per_traj):
            cur_wp = seg_ee[s]                                  # current-segment EE waypoint for the error metric

            if random:
                a0 = np.clip(0.5 * np.random.randn(jd), -1.0, 1.0).astype(np.float32)
            else:
                ee_w, tw_w = ref_window(seg_ee, seg_twist, s, horizon)        # padded pose + exact-twist windows
                st = plan_state(penv, q_buf, qd_buf, act_buf, ee_w, tw_w, jd)
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
            errs.append(float(np.linalg.norm(sim.ee_pos.squeeze(0).cpu().numpy() - cur_wp)))  # EE error (m)

            q_buf = np.roll(q_buf, -jd); q_buf[-jd:] = q_t
            qd_buf = np.roll(qd_buf, -jd); qd_buf[-jd:] = qd_t
            act_buf = np.roll(act_buf, -jd); act_buf[-jd:] = a0
            rows.append(np.concatenate([q_buf, qd_buf, act_buf]))

    rows = np.asarray(rows, np.float32)
    ds = Dataset(input=jnp.asarray(rows), target=jnp.asarray(np.zeros((len(rows), jd), np.float32)))
    return ds, ptimes, float(np.mean(errs)), ep_reward, rng


def main():
    jd, bd = 4, 10
    H_PLAN, N_SAMPLES, N_STAGES = 20, 500, 10      # ICEM: horizon 30, 12 CEM sampling stages
    N_TRAJ, WARMUP_TRAJ, N_ROUNDS = 10, 2, 200      # episode = 10 chained refs x t_traj(6s) = 60 s

    cfg = SimpleNamespace(step_penalty_coef=0, action_penalty_coef=1,
                          accel_penalty_coef=0, accel_sign_penalty_coef=0, twist_coef=1.0)
    sim = SimEnv(n_envs=1, use_act_net=True, n_history_steps=bd,
                 n_ref_steps=15, t_step=0.04, t_traj=6.0, cfg=cfg)
    model = RobotEnsemble(joint_dim=jd, buffer_dim=bd, model_type="PE", mode="v", num_ensembles=5)
    penv = PlanEnv(model, cfg=cfg)
    icem = ICEM(penv, horizon=H_PLAN, nb_samples=N_SAMPLES, nb_elites=60,
                init_std=0.5, nb_steps=N_STAGES, exponent=1.0, alpha=0.15)     # colored-noise iCEM
    plan_fn = make_plan_fn(penv, icem)

    logger = ListLogger()
    train_ds, val_ds = [], []
    trainer = OnlineTrainer(model, optax.adamw(1e-4, weight_decay=1e-4), train_ds, val_ds,
                            batch_size=64, horizon=10, horizon_val=15,
                            nb_epochs=3, early_stopping_patience=10_000, logger=logger)
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
    plan_fn(plan_state(penv, zeros, zeros, zeros, np.zeros((H_PLAN, 3), np.float32),
                       np.zeros((H_PLAN, 6), np.float32), jd),
            jnp.zeros((H_PLAN, jd)), rng)[0].block_until_ready()

    # --- online ICEM loop ---
    plot_dir = os.path.join(os.path.dirname(__file__), "episode_plots")
    os.makedirs(plot_dir, exist_ok=True)
    hist_lists = (sim.predicted_trajectories, sim.predicted_velocities, sim.joint_pos_histories,
                  sim.joint_vel_histories, sim.reward_histories, sim.target_trajectories,
                  sim.target_joint_pos_histories)

    all_ptimes = []
    print("\n=== online ICEM: plan -> execute -> train ===")
    for r in range(N_ROUNDS):
        for lst in hist_lists:
            lst.clear()                                          # only this episode's sub-trajectories
        ds, ptimes, err, ep_reward, rng = run_episode(sim, penv, plan_fn, jd, bd, H_PLAN, rng, N_TRAJ, render=True)
        sim.render(mode="plot", save_dir=plot_dir, file_name=f"episode_{r:02d}")   # followed vs desired EE
        train_ds.append(ds)
        val_mse = float(trainer.train_model_bptt(seed=r + 1, verbose=False))   # best val MSE this round
        val_nll = logger.data["train/val_loss"][-1][1]                         # latest val NLL
        penv._params = nnx.split(model.model)[1]                # update planner with retrained model
        all_ptimes += ptimes
        print(f"Round {r}: {N_TRAJ} refs / {len(ptimes)} steps (60s) | ICEM {np.mean(ptimes) * 1000:.2f} ms/step "
              f"({1.0 / np.mean(ptimes):.0f} Hz) | episode reward {ep_reward:.2f} "
              f"| tracking err {err:.3f} m | model NLL {val_nll:.3f} MSE {val_mse:.2e} "
              f"| train steps {sum(len(d) for d in train_ds)}")

    t = np.mean(all_ptimes)
    print(f"\n=== ICEM policy inference: {t * 1000:.2f} ms/step -> {1.0 / t:.0f} Hz "
          f"(horizon={H_PLAN}, {N_SAMPLES} samples, {N_STAGES} stages, ensemble dynamics) ===")


if __name__ == "__main__":
    main()
