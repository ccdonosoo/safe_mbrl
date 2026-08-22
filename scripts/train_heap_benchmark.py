"""Online MBRL (ensemble + MPPI) on HeapEnv, aligned with the TD-MPC2 / DreamerV3 heap benchmark.

One comparison unit ("episode index") = one round = --n-envs (10) parallel fresh-reset episodes
of 150 steps stepped as a single internally-batched HeapEnv (like the TD-MPC2 baseline)
= 1500 transitions. Per-round metrics mirror tdmpc2's train/ keys (episode_reward meaned over
the batch, err_term_*); eval runs on a fixed reference trajectory like the DreamerV3 runs.

Algorithm usage per colleague: the ensemble is retrained every round with 3 epochs over a
sliding window of the last --data-window rounds of data (default 60 rounds = 600 episodes);
MPPI is vmapped over the parallel envs and the per-batch training is fused into one jit call.

Planner objectives (--objective):
  tuned    - the safe_mbrl tuning: joint-space tracking (w=8) + EE pos (w=2), action penalty 0.01
  matching - the benchmark reward: -||ee_pos - ref||^2 - 1.0*||a_t - a_{t-1}||^2 (same as sim)
  woema    - ablation: identical to `tuned` but with the action EMA filter turned off
             (alpha=1.0), both on the executed action and inside the planner's rollout model
The scored (sim) reward uses the benchmark coefficients (0, 1, 0, 0) in all variants.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import csv
import json
import pickle
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from safe_mbrl.envs.base import State
from safe_mbrl.envs.heap_env.heap_example_env import HeapEnv as SimEnv
from safe_mbrl.envs.heap_m545 import HeapEnv as PlanEnv
from safe_mbrl.models.robot_ensemble import RobotEnsemble
from safe_mbrl.models.online_trainer import OnlineTrainer
from safe_mbrl.utils.structs import Dataset, RobotState
from safe_mbrl.mpc.mppi import MPPI

JD = 4
ERR_KEYS = ("err_ee_pos", "err_ee_rot", "err_ee_vel", "err_j_pos", "err_j_vel")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objective", choices=["tuned", "matching", "woema"], required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-rounds", type=int, default=400)
    p.add_argument("--n-envs", type=int, default=10, help="parallel envs (batched inside HeapEnv)")
    p.add_argument("--data-window", type=int, default=60,
                   help="sliding training window in rounds (1 round = n-envs episodes = 1500 steps)")
    p.add_argument("--exp-name", default=None)
    p.add_argument("--logdir", default=None)
    # scored (sim) reward, benchmark coefficients
    p.add_argument("--step-penalty-coef", type=float, default=0.0)
    p.add_argument("--action-penalty-coef", type=float, default=1.0)
    p.add_argument("--accel-penalty-coef", type=float, default=0.0)
    p.add_argument("--accel-sign-penalty-coef", type=float, default=0.0)
    # planner objective (tuned variant)
    p.add_argument("--joint-weight", type=float, default=8.0)
    p.add_argument("--ee-weight", type=float, default=2.0)
    p.add_argument("--plan-action-penalty-coef", type=float, default=None,
                   help="default: 0.01 for tuned, --action-penalty-coef for matching")
    # planner / model (safe_mbrl defaults)
    p.add_argument("--plan-horizon", type=int, default=30)
    p.add_argument("--nb-samples", type=int, default=2000)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--mppi-steps", type=int, default=3)
    p.add_argument("--init-std", type=float, default=0.5)
    p.add_argument("--action-ema", type=float, default=None,
                   help="EMA alpha on actions (1.0 = off); default 0.18, forced to 1.0 for --objective woema")
    p.add_argument("--buffer-dim", type=int, default=15)
    p.add_argument("--num-ensembles", type=int, default=5)
    p.add_argument("--features", type=int, nargs="+", default=[256, 256])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--nb-epochs", type=int, default=3)
    p.add_argument("--bptt-horizon", type=int, default=10)
    # eval / output
    p.add_argument("--eval-every", type=int, default=5, help="rounds between evals; 0 disables")
    # Fixed 3320-step reference trajectory (dict with ref_pos/ref_vel, (3320, 4) joint-space)
    # shared by the TD-MPC2 / DreamerV3 / MBPO benchmark evals. Bundled in the repo, so the
    # default works from a fresh clone; point --eval-traj elsewhere to eval on a different one.
    p.add_argument("--eval-traj",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "data", "ref_traj_eval.pt"))
    p.add_argument("--save-data", action=argparse.BooleanOptionalAction, default=True,
                   help="save every round's raw transitions to data/round_XXX.npy (~1.1 MB/round)")
    p.add_argument("--ckpt-every", type=int, default=1,
                   help="save ensemble params every N rounds (0 disables; ~2.3 MB each)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="heap_td_mpc")
    p.add_argument("--wandb-entity", default="fangnan")   # only used with --wandb; set to your own entity
    return p.parse_args()


def make_plan_fn(penv, mpc, objective):
    """Batched planner: vmap MPPI over the parallel envs' states. Ensemble params are an
    explicit argument (threaded into each lane's state.info) so updates never go stale
    inside the jit cache."""
    track_joint = objective in ("tuned", "woema")

    @jax.jit
    def plan(params, q_buf, qd_buf, act_buf, target_seq, last_action, init_actions, rng):
        rngs = jax.random.split(rng, q_buf.shape[0])

        def plan_one(q, qd, act, tgt, last_a, init, r):
            # mirrors PlanEnv.make_traj_state, with params injected instead of read from self
            rs = RobotState(q_buffer=q, qd_buffer=qd, act_buffer=act, q_dim=JD)
            info = {"last_action": last_a, "step": jnp.zeros((), jnp.int32), "params": params}
            if track_joint:
                info["q_target_seq"] = tgt
                if penv._ee_weight > 0.0:
                    info["ee_target_seq"] = penv._ee_ref_fk(tgt)
            else:
                info["ee_target_seq"] = tgt
            z = jnp.zeros(())
            st = State(rs, penv._get_obs(rs, info), z, z, {"reward": z}, info)

            def sum_rewards(seq):
                def body(s, a):
                    s = penv.step(s, a)
                    return s, s.reward
                return jax.lax.scan(body, st, seq)[1].sum()

            return mpc.optimize(sum_rewards, r, init, (-1.0, 1.0))

        return jax.vmap(plan_one)(q_buf, qd_buf, act_buf, target_seq, last_action, init_actions, rngs)

    return plan


def clear_render_history(sim):
    for lst in (sim.predicted_trajectories, sim.predicted_velocities,
                sim.joint_pos_histories, sim.joint_vel_histories, sim.reward_histories,
                sim.target_trajectories, sim.target_joint_pos_histories):
        lst.clear()


def apply_fixed_eval_traj(sim, traj):
    """Overwrite the freshly-reset sim with the fixed reference trajectory
    (same procedure as the DreamerV3 benchmark's eval envs)."""
    ref = traj["ref_pos"].to(sim.device).float()            # (T, JD)
    T = int(ref.shape[0])
    sim.ref_traj_joint = ref.transpose(0, 1).unsqueeze(0)
    sim.ref_traj_Tee = sim.kinematics.forward_kinematics(
        sim.ref_traj_joint.transpose(1, 2).reshape(T, -1)).get_matrix().reshape(1, T, 4, 4)
    sim.ref_traj_eepos = sim.ref_traj_Tee[:, :, :3, 3]
    sim.ref_traj_eerot = sim.ref_traj_Tee[:, :, :3, :3]
    sim.actnet.reset_static_pos(sim.ref_traj_joint[:, :, 0])
    sim.dof_pos_history = (sim.actnet.pos_buffer[:, 0] * sim.actnet.posStds
                           + sim.actnet.posMeans).unsqueeze(-1).repeat(1, 1, sim.n_history_steps)
    sim.dof_vel_history = (sim.actnet.vel_buffer[:, 0] * sim.actnet.velStds
                           + sim.actnet.velMeans).unsqueeze(-1).repeat(1, 1, sim.n_history_steps)
    sim.ee_pos = sim.kinematics.forward_kinematics(sim.dof_pos_history[:, :, 0]).get_matrix()[:, :3, 3]
    sim.ee_rot = sim.kinematics.forward_kinematics(sim.dof_pos_history[:, :, 0]).get_matrix()[:, :3, :3]
    sim.ee_vel = sim.kinematics.jacobian(sim.dof_pos_history[:, :, 0]).matmul(
        sim.dof_vel_history[:, :, 0].unsqueeze(-1)).squeeze(-1)
    sim.accel = (sim.ee_vel - torch.zeros_like(sim.ee_vel)) / sim.t_step
    sim.current_step = 0
    sim.ref_traj_steps = T
    sim.action = torch.zeros(sim.n_envs, sim.act_dim, device=sim.device)
    sim.last_action = sim.action.clone()


def run_round(sim, penv, plan_fn, params, rng, args, fixed_traj=None, render=False, collect=True):
    """One batched fresh-reset rollout over all sim.n_envs parallel envs.
    Returns (per_env_rows, ptimes, ep_rewards (E,), err_means, rng)."""
    sim.reset()
    if fixed_traj is not None:
        apply_fixed_eval_traj(sim, fixed_traj)
    if render:
        sim.render()

    E = sim.n_envs
    bd, H = args.buffer_dim, args.plan_horizon
    q0 = sim.dof_pos_history[:, :, 0].cpu().numpy()          # (E, JD)
    q_buf = np.tile(q0, (1, bd)).astype(np.float32)          # newest at the end, as RobotState.create
    qd_buf = np.zeros((E, JD * bd), np.float32)
    act_buf = np.zeros((E, JD * bd), np.float32)
    a_prev = np.zeros((E, JD), np.float32)
    init = jnp.zeros((E, H, JD))

    if args.objective in ("tuned", "woema"):
        seg = sim.ref_traj_joint.permute(0, 2, 1).cpu().numpy()   # (E, ref_steps, JD)
    else:
        seg = sim.ref_traj_eepos.cpu().numpy()                    # (E, ref_steps, 3)

    step_rows, ptimes = [], []
    ep_rewards = np.zeros(E)
    err_sums = {k: 0.0 for k in ERR_KEYS}
    n_steps = sim.ref_traj_steps - 1

    for s in range(n_steps):
        idx = np.minimum(np.arange(s + 1, s + 1 + H), seg.shape[1] - 1)
        tgt = jnp.asarray(seg[:, idx])                            # (E, H, dim)
        rng, k = jax.random.split(rng)
        t0 = time.perf_counter()
        aseq, _ = plan_fn(params, jnp.asarray(q_buf), jnp.asarray(qd_buf), jnp.asarray(act_buf),
                          tgt, jnp.asarray(a_prev), init, k)      # (E, H, JD)
        aseq.block_until_ready()
        ptimes.append(time.perf_counter() - t0)
        init = jnp.concatenate([aseq[:, 1:], aseq[:, -1:]], axis=1)
        # same EMA filter the planner models in its rollout
        a0 = (args.action_ema * np.asarray(aseq[:, 0]) + (1.0 - args.action_ema) * a_prev).astype(np.float32)
        a_prev = a0

        _, rwd, _, _, info = sim.step(a0)
        if render:
            sim.render(done=(s == n_steps - 1))

        ep_rewards += np.asarray(rwd).reshape(-1)
        for kk in ERR_KEYS:
            err_sums[kk] += float(info["err_terms"][kk])

        q_t = sim.dof_pos_history[:, :, 0].cpu().numpy()
        qd_t = sim.dof_vel_history[:, :, 0].cpu().numpy()
        q_buf = np.roll(q_buf, -JD, axis=1); q_buf[:, -JD:] = q_t
        qd_buf = np.roll(qd_buf, -JD, axis=1); qd_buf[:, -JD:] = qd_t
        act_buf = np.roll(act_buf, -JD, axis=1); act_buf[:, -JD:] = a0
        if collect:
            step_rows.append(np.concatenate([q_buf, qd_buf, act_buf], axis=1).copy())

    err_means = {k: v / n_steps for k, v in err_sums.items()}
    per_env_rows = None
    if collect:
        arr = np.asarray(step_rows, np.float32)                   # (T, E, 3*JD*bd)
        per_env_rows = [np.ascontiguousarray(arr[:, e, :]) for e in range(E)]
    return per_env_rows, ptimes, ep_rewards, err_means, rng


class RunLogger:
    """CSV + JSONL (+ optional wandb) with tdmpc2-style keys."""

    def __init__(self, logdir, args):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self._files = {}
        self._writers = {}
        self._wandb = None
        if args.wandb:
            import wandb
            self._wandb = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                                     name=args.exp_name, group=f"heap-eetracking-{args.objective}",
                                     dir=logdir, config=vars(args))
        self._jsonl = open(os.path.join(logdir, "metrics.jsonl"), "a")

    def log(self, row, category):
        path = os.path.join(self.logdir, f"{category}.csv")
        if category not in self._writers:
            new = not os.path.exists(path) or os.path.getsize(path) == 0
            f = open(path, "a", newline="")
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            self._files[category], self._writers[category] = f, w
        self._writers[category].writerow(row)
        self._files[category].flush()
        self._jsonl.write(json.dumps({"category": category, **row}) + "\n")
        self._jsonl.flush()
        if self._wandb is not None:
            self._wandb.log({f"{category}/{k}": v for k, v in row.items() if k != "step"},
                            step=int(row["step"]))

    def finish(self):
        for f in self._files.values():
            f.close()
        self._jsonl.close()
        if self._wandb is not None:
            self._wandb.finish()


def main():
    args = parse_args()
    if args.exp_name is None:
        args.exp_name = f"{args.objective}_seed{args.seed}"
    if args.logdir is None:
        args.logdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "logs", "heap-eetracking", args.objective, str(args.seed))
    if args.plan_action_penalty_coef is None:
        args.plan_action_penalty_coef = args.action_penalty_coef if args.objective == "matching" else 0.01
    if args.objective == "woema":
        args.action_ema = 1.0
    elif args.action_ema is None:
        args.action_ema = 0.18

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # scored reward: benchmark coefficients, identical across variants and to the TD-MPC2/Dreamer runs
    sim_cfg = SimpleNamespace(step_penalty_coef=args.step_penalty_coef,
                              action_penalty_coef=args.action_penalty_coef,
                              accel_penalty_coef=args.accel_penalty_coef,
                              accel_sign_penalty_coef=args.accel_sign_penalty_coef)
    # planner objective: what MPPI optimizes through the learned model
    if args.objective in ("tuned", "woema"):
        plan_cfg = SimpleNamespace(track_mode="joint",
                                   joint_weight=args.joint_weight,
                                   ee_weight=args.ee_weight,
                                   action_penalty_coef=args.plan_action_penalty_coef,
                                   action_ema_alpha=args.action_ema)
    else:
        plan_cfg = SimpleNamespace(track_mode="ee",
                                   action_penalty_coef=args.plan_action_penalty_coef,
                                   action_ema_alpha=args.action_ema)

    sim = SimEnv(n_envs=args.n_envs, use_act_net=True, n_history_steps=args.buffer_dim,
                 n_ref_steps=args.plan_horizon, t_step=0.04, t_traj=6.0, cfg=sim_cfg)
    fixed_traj = None
    sim_eval = None
    if args.eval_every > 0:
        fixed_traj = torch.load(args.eval_traj, map_location=sim.device)
        sim_eval = SimEnv(n_envs=1, use_act_net=True, n_history_steps=args.buffer_dim,
                          n_ref_steps=args.plan_horizon, t_step=0.04, t_traj=6.0, cfg=sim_cfg)

    model = RobotEnsemble(joint_dim=JD, buffer_dim=args.buffer_dim, model_type="PE", mode="v",
                          features=tuple(args.features), num_ensembles=args.num_ensembles,
                          key=jax.random.key(args.seed))
    penv = PlanEnv(model, cfg=plan_cfg)
    mppi = MPPI(penv, horizon=args.plan_horizon, nb_samples=args.nb_samples,
                temperature=args.temperature, init_std=args.init_std, nb_steps=args.mppi_steps)
    plan_fn = make_plan_fn(penv, mppi, args.objective)

    train_ds = []
    all_rows = []                                            # host-side copy for the hoisted concat
    trainer = OnlineTrainer(model, optax.adamw(args.lr, weight_decay=args.weight_decay),
                            train_ds, [], batch_size=args.batch_size,
                            horizon=args.bptt_horizon, nb_epochs=args.nb_epochs)

    rng = jax.random.fold_in(jax.random.key(args.seed), 1)

    logger = RunLogger(args.logdir, args)
    plot_dir = os.path.join(args.logdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.logdir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    data_dir = os.path.join(args.logdir, "data")
    if args.save_data:
        os.makedirs(data_dir, exist_ok=True)

    def save_ckpt(r):
        with open(os.path.join(ckpt_dir, f"params_{r:03d}.pkl"), "wb") as f:
            pickle.dump(jax.device_get(nnx.split(model.model)[1]), f)

    if args.ckpt_every > 0:
        save_ckpt(0)                                     # untrained init
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         text=True).strip()
    except Exception:
        commit = "unknown"
    with open(os.path.join(args.logdir, "config.json"), "w") as f:
        json.dump({**vars(args), "git_commit": commit,
                   "plan_cfg": vars(plan_cfg), "sim_cfg": vars(sim_cfg)}, f, indent=2)

    ep_steps = sim.ref_traj_steps - 1                        # 150
    t_start = time.time()

    def run_eval(r, rng):
        clear_render_history(sim_eval)
        _, _, ep_rewards, err_means, rng = run_round(
            sim_eval, penv, plan_fn, penv._params, rng, args,
            fixed_traj=fixed_traj, render=True, collect=False)
        sim_eval.render(mode="plot", save_dir=plot_dir, file_name=f"eval_{r:03d}")
        np.savez(os.path.join(plot_dir, f"eval_{r:03d}_data.npz"),
                 ee_pos=np.asarray(sim_eval.predicted_trajectories[-1]),
                 ee_pos_ref=np.asarray(sim_eval.target_trajectories[-1]))
        row = {"step": r * ep_steps, "episode_reward": float(ep_rewards[0]),
               **{f"err_term_{k}": v for k, v in err_means.items()}}
        logger.log(row, "eval")
        print(f"[eval @ round {r}] reward {row['episode_reward']:.2f}", flush=True)
        return rng

    if args.eval_every > 0:
        rng = run_eval(0, rng)

    for r in range(1, args.n_rounds + 1):
        t_round = time.time()
        per_env_rows, ptimes, ep_rewards, err_means, rng = run_round(
            sim, penv, plan_fn, penv._params, rng, args)

        if args.save_data:
            # (n_envs, 150, 3*JD*buffer_dim): full buffers per step -> any window/metric recomputable
            np.save(os.path.join(data_dir, f"round_{r:03d}.npy"), np.stack(per_env_rows))

        for ep_rows in per_env_rows:
            all_rows.append(ep_rows)
            # one Dataset per episode: BPTT windows must not cross reset boundaries
            train_ds.append(Dataset(input=jnp.asarray(ep_rows),
                                    target=jnp.asarray(np.zeros((len(ep_rows), JD), np.float32))))
        max_eps = args.data_window * args.n_envs
        if len(train_ds) > max_eps:
            del train_ds[:-max_eps]      # in place: the trainer holds a reference to this list
            del all_rows[:-max_eps]

        t_fit = time.time()
        train_concat = Dataset(input=jnp.asarray(np.concatenate(all_rows)),
                               target=jnp.zeros((sum(len(x) for x in all_rows), JD), jnp.float32))
        fit = trainer.train_model_bptt_jit(seed=args.seed * 100000 + r, verbose=False,
                                           train_concat=train_concat)
        nll = fit["nll"]
        penv._params = nnx.split(model.model)[1]
        if args.ckpt_every > 0 and r % args.ckpt_every == 0:
            save_ckpt(r)

        row = {"step": r * ep_steps,
               "episode": r,
               "env_steps": r * args.n_envs * ep_steps,
               "episode_reward": float(np.mean(ep_rewards)),
               "episode_reward_std": float(np.std(ep_rewards)),
               **{f"err_term_{k}": err_means[k] for k in ERR_KEYS},
               "model_loss": nll,                    # Gaussian NLL (the optimized objective)
               "model_mse": fit["mse"],              # discounted H-step joint-position MSE [rad^2]
               "model_mse_vel": fit["mse_vel"],      # one-step velocity MSE [rad^2/s^2] (MBPO's "Model Err")
               "mppi_ms": float(np.mean(ptimes) * 1000),
               "fit_s": time.time() - t_fit,
               "round_s": time.time() - t_round,
               "total_time": time.time() - t_start}
        logger.log(row, "train")
        print(f"Round {r}/{args.n_rounds}: reward {row['episode_reward']:.2f}±{row['episode_reward_std']:.2f} "
              f"| err_ee_pos {row['err_term_err_ee_pos']:.4f} | NLL {nll:.3f} "
              f"| MSE {fit['mse']:.2e}/{fit['mse_vel']:.2e} "
              f"| MPPI {row['mppi_ms']:.1f} ms/step | fit {row['fit_s']:.1f}s | round {row['round_s']:.1f}s",
              flush=True)

        if args.eval_every > 0 and r % args.eval_every == 0:
            rng = run_eval(r, rng)

    logger.finish()
    print(f"Done: {args.n_rounds} rounds in {(time.time() - t_start) / 3600:.2f} h -> {args.logdir}")


if __name__ == "__main__":
    main()
