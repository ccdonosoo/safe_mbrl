"""Strike tracking in sim: (p, n) -> batched IK -> min-jerk (EE speed cap) ->
MPPI (deploy params) on the m445_plant ensemble as both planner model and
plant. Prints heap-style EE tracking metrics and saves a plot."""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as orbax
from flax import nnx

from safe_mbrl.envs.m445_hammer import HammerEnv, M445HammerFK, make_fk_batch
from safe_mbrl.ik.newton_ik import NewtonBatchIK, select_best
from safe_mbrl.m445_hammer_spec import SAFE_Q_MAX, SAFE_Q_MIN
from safe_mbrl.models.robot_ensemble import RobotEnsemble
from safe_mbrl.mpc.mppi import MPPI
from safe_mbrl.traj.min_jerk import ee_speed_duration, min_jerk_ref
from safe_mbrl.utils.structs import RobotState

MODEL_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "safe_mbrl", "envs", "m445_plant", "model"))

# deploy MPPI / tracking parameters (perceptive 5-DOF YAML)
T_STEP = 0.04
HORIZON = 30
NB_SAMPLES = 2000
TEMPERATURE = 0.05
INIT_STD = 0.5
NB_STEPS = 3
EMA_ALPHA = 0.18
Q_TRACK_WEIGHT = np.array([4.0, 1.0, 1.0, 1.0, 1.0], np.float32)

# mechanical range (URDF): the plant clips here, not at the safe box
HAMMER_Q_LO = np.array([-3.1416, -1.41, 0.62, 0.0, -0.30], np.float32)
HAMMER_Q_HI = np.array([3.1416, 0.44, 2.8, 1.604, 2.65], np.float32)


def load_ensemble(model_dir=MODEL_DIR, weights: str = "ckpt") -> RobotEnsemble:
    """Ensemble from the orbax checkpoint; weights="random" skips the restore."""
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    model = RobotEnsemble(**cfg)
    if weights == "random":
        print("WARNING: --model random - untrained ensemble, tracking numbers "
              "are NOT representative")
        return model
    _, _, state = nnx.split(model.model, nnx.RngState, ...)
    sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    restore_args = jax.tree.map(
        lambda _: orbax.ArrayRestoreArgs(sharding=sharding), state)
    state = orbax.PyTreeCheckpointer().restore(
        os.path.join(model_dir, "state"), item=state, restore_args=restore_args)
    nnx.update(model.model, state)
    return model


class EnsemblePlant:
    """The ensemble as the simulated machine (same stepping as M445NewtonPlant)."""

    def __init__(self, ensemble: RobotEnsemble, sampling: str = "normal",
                 seed: int = 0):
        if sampling not in ("normal", "member", "mean"):
            raise ValueError(sampling)
        self.jd, self.bd, self.dt = (ensemble.joint_dim, ensemble.buffer_dim,
                                     ensemble._dt)
        graphdef, self._params = nnx.split(ensemble.model)
        input_idx = ensemble._input_idx
        mode, dt, jd = ensemble.mode, self.dt, self.jd
        n_ens = ensemble.model.num_ensembles
        lo, hi = jnp.asarray(HAMMER_Q_LO), jnp.asarray(HAMMER_Q_HI)

        @jax.jit
        def step_fn(params, rs, action, key):
            net = nnx.merge(graphdef, params)
            x = rs.ravel() if input_idx is None else rs.ravel()[input_idx]
            mu, sig = jnp.split(net(x), 2, axis=-1)
            k_member, k_noise = jax.random.split(key)
            m = jax.random.randint(k_member, (), 0, n_ens)
            if sampling == "mean":
                v = mu.mean(axis=0)
            elif sampling == "member":
                v = mu[m]
            else:
                v = mu[m] + sig[m] * jax.random.normal(k_noise, (jd,))
            qd_next = rs.get_qd() + v if mode == "dv" else v
            q_next = jnp.clip(rs.get_q() + qd_next * dt, lo, hi)
            qd_next = (q_next - rs.get_q()) / dt
            return rs.replace(
                q_buffer=jnp.roll(rs.q_buffer, -jd).at[-jd:].set(q_next),
                qd_buffer=jnp.roll(rs.qd_buffer, -jd).at[-jd:].set(qd_next),
                act_buffer=jnp.roll(rs.act_buffer, -jd).at[-jd:].set(action))

        self._step_fn = step_fn
        self._key = jax.random.key(seed)

    def reset(self, q0):
        self.rs = RobotState.create(jnp.asarray(q0, jnp.float32),
                                    buffer_size=self.bd, q_dim=self.jd)

    def step(self, action):
        self._key, k = jax.random.split(self._key)
        self.rs = self._step_fn(self._params, self.rs,
                                jnp.clip(jnp.asarray(action, jnp.float32), -1, 1), k)
        return (np.asarray(self.rs.get_q(), np.float32),
                np.asarray(self.rs.get_qd(), np.float32))


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


def ref_window(seg, s, horizon):
    idx = np.minimum(np.arange(s + 1, s + 1 + horizon), len(seg) - 1)
    return seg[idx]


def sample_reachable_target(fk_pos_axis, rng):
    while True:
        q = rng.uniform(SAFE_Q_MIN, SAFE_Q_MAX, (64, 5)).astype(np.float32)
        pos, axis = fk_pos_axis(q)
        keep = axis[:, 2] < -0.05
        if keep.any():
            i = int(np.argmax(keep))
            return pos[i], -axis[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", type=float, nargs=3, default=None,
                    help="target xyz in BASE frame (default: sampled reachable)")
    ap.add_argument("--normal", type=float, nargs=3, default=None)
    ap.add_argument("--target-seed", type=int, default=3)
    ap.add_argument("--v-ee", type=float, default=0.20, help="EE speed cap [m/s]")
    ap.add_argument("--criterion", default="ee_path",
                    choices=("ee_path", "joint_dist"))
    ap.add_argument("--sampling", default="normal",
                    choices=("normal", "member", "mean"))
    ap.add_argument("--ik-seeds", type=int, default=256)
    ap.add_argument("--settle", type=float, default=1.0,
                    help="extra seconds holding the strike pose ref")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="ckpt", choices=("ckpt", "random"))
    args = ap.parse_args()

    rng_np = np.random.default_rng(args.target_seed)
    ensemble = load_ensemble(weights=args.model)
    fk = M445HammerFK()
    fk_pos_axis, fk_pos = make_fk_batch(fk)

    if args.point is not None:
        p, n = np.asarray(args.point, np.float32), np.asarray(args.normal, np.float32)
    else:
        p, n = sample_reachable_target(fk_pos_axis, rng_np)
    q0 = 0.5 * (SAFE_Q_MIN + SAFE_Q_MAX)
    print(f"target p = {np.round(p, 3)}  n = {np.round(n, 3)}  (BASE frame)")

    ik = NewtonBatchIK(n_targets=1, n_seeds=args.ik_seeds)
    t0 = time.perf_counter()
    cand = ik.solve(p[None], n[None], q_now=q0, rng_seed=args.seed)
    best = select_best(cand, p[None], n[None], q0, fk_pos_axis,
                       criterion=args.criterion)
    print(f"IK ({args.ik_seeds} seeds, {1e3 * (time.perf_counter() - t0):.0f} ms): "
          f"ok={bool(best['ok'][0])}  n_feasible={int(best['n_feasible'][0])}  "
          f"pos_err={1e3 * best['pos_err'][0]:.2f} mm  "
          f"axis_err={best['axis_err_deg'][0]:.2f} deg")
    if not best["ok"][0]:
        raise SystemExit("no feasible in-box IK solution - aborting")
    q1 = best["q"][0]
    print(f"q0 = {np.round(q0, 3)}\nq1 = {np.round(q1, 3)}")

    T = ee_speed_duration(q0, q1, fk_pos, v_ee_max=args.v_ee, t_step=T_STEP)
    q_ref, qd_ref = min_jerk_ref(q0, q1, T, T_STEP)
    ee_ref = fk_pos(q_ref)
    ref_speed = np.linalg.norm(np.diff(ee_ref, axis=0), axis=-1) / T_STEP
    print(f"duration T = {T:.2f} s ({len(q_ref)} steps)  "
          f"reference peak EE speed = {ref_speed.max():.3f} m/s (cap {args.v_ee})")

    cfg = SimpleNamespace(
        track_mode="joint", joint_weight=8.0, ee_weight=4.0, qd_coef=0.0,
        action_penalty_coef=0.05, action_ema_alpha=EMA_ALPHA,
        q_cost_scale=(SAFE_Q_MAX - SAFE_Q_MIN) / Q_TRACK_WEIGHT)
    penv = HammerEnv(ensemble, cfg=cfg)
    mppi = MPPI(penv, horizon=HORIZON, nb_samples=NB_SAMPLES,
                temperature=TEMPERATURE, init_std=INIT_STD, nb_steps=NB_STEPS)
    plan_fn = make_plan_fn(penv, mppi)
    plant = EnsemblePlant(ensemble, sampling=args.sampling, seed=args.seed)
    plant.reset(q0)

    jd, bd = 5, ensemble.buffer_dim
    q_buf = np.tile(q0, bd).astype(np.float32)
    qd_buf = np.zeros(jd * bd, np.float32)
    act_buf = np.zeros(jd * bd, np.float32)
    init = jnp.zeros((HORIZON, jd))
    rng = jax.random.key(args.seed)

    n_steps = len(q_ref) - 1 + int(round(args.settle / T_STEP))
    q_log, ptimes = [], []
    for s in range(n_steps):
        st = penv.make_traj_state(q_buf, qd_buf, act_buf,
                                  ref_window(q_ref, s, HORIZON), jd,
                                  ref_window(qd_ref, s, HORIZON),
                                  last_action=act_buf[-jd:])
        rng, k = jax.random.split(rng)
        t0 = time.perf_counter()
        aseq, _ = plan_fn(st, init, k)
        aseq.block_until_ready()
        ptimes.append(time.perf_counter() - t0)
        init = jnp.concatenate([aseq[1:], aseq[-1:]], axis=0)
        # deploy-side EMA against the command actually applied last tick
        a0 = (EMA_ALPHA * np.asarray(aseq[0])
              + (1.0 - EMA_ALPHA) * act_buf[-jd:]).astype(np.float32)

        q_t, qd_t = plant.step(a0)
        q_buf = np.roll(q_buf, -jd); q_buf[-jd:] = q_t
        qd_buf = np.roll(qd_buf, -jd); qd_buf[-jd:] = qd_t
        act_buf = np.roll(act_buf, -jd); act_buf[-jd:] = a0
        q_log.append(q_t)

    q_log = np.asarray(q_log)
    ee_log, axis_log = fk_pos_axis(q_log)
    ref_idx = np.minimum(np.arange(1, n_steps + 1), len(q_ref) - 1)
    ee_err = np.linalg.norm(ee_log - ee_ref[ref_idx], axis=-1)
    q_err = q_log - q_ref[ref_idx]
    ee_speed = np.linalg.norm(np.diff(ee_log, axis=0), axis=-1) / T_STEP
    tgt_axis = -(n / np.linalg.norm(n))
    if tgt_axis[2] > 0:
        tgt_axis = -tgt_axis
    final_axis_err = np.degrees(np.arccos(
        np.clip(axis_log[-1] @ tgt_axis, -1, 1)))

    print("\n=== tracking (plant: ensemble '%s') ===" % args.sampling)
    print(f"EE err vs ref : mean {ee_err.mean():.4f} m | RMSE "
          f"{np.sqrt((ee_err ** 2).mean()):.4f} m | max {ee_err.max():.4f} m")
    print("joint RMSE    : " + "  ".join(
        f"{nm}={v:.4f}" for nm, v in zip(
            ("turn", "boom", "stick", "tele", "pitch"),
            np.sqrt((q_err ** 2).mean(0)))))
    print(f"EE speed      : peak {ee_speed.max():.3f} m/s (cap {args.v_ee})")
    print(f"final strike  : pos err {1e3 * np.linalg.norm(ee_log[-1] - p):.1f} mm | "
          f"axis err {final_axis_err:.2f} deg")
    print(f"MPPI          : {1e3 * np.mean(ptimes[1:]):.1f} ms/step "
          f"({1.0 / np.mean(ptimes[1:]):.0f} Hz)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(n_steps) * T_STEP
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    axs[0].plot(ee_ref[:, 0], ee_ref[:, 2], "k--", label="ref")
    axs[0].plot(ee_log[:, 0], ee_log[:, 2], "b", label="tracked")
    axs[0].scatter(*p[[0, 2]], c="r", marker="x", s=80, label="target")
    axs[0].quiver(*p[[0, 2]], *(0.5 * np.array([n[0], n[2]])), color="r", width=4e-3)
    axs[0].set_xlabel("x [m]"); axs[0].set_ylabel("z [m]")
    axs[0].set_title("EE path (BASE x-z)"); axs[0].legend(); axs[0].axis("equal")
    for i, nm in enumerate(("turn", "boom", "stick", "tele", "pitch")):
        axs[1].plot(t, q_ref[ref_idx][:, i], "--", lw=1, color=f"C{i}")
        axs[1].plot(t, q_log[:, i], color=f"C{i}", label=nm)
    axs[1].set_xlabel("t [s]"); axs[1].set_title("joints vs ref (dashed)")
    axs[1].legend(ncol=2, fontsize=8)
    axs[2].plot(t[1:], ee_speed, "b", label="tracked")
    axs[2].plot(t[:len(ref_speed)], ref_speed, "k--", label="ref")
    axs[2].axhline(args.v_ee, color="r", ls=":", label="cap")
    axs[2].set_xlabel("t [s]"); axs[2].set_ylabel("EE speed [m/s]")
    axs[2].set_title("EE speed"); axs[2].legend()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "strike_sim_out")
    os.makedirs(out, exist_ok=True)
    fname = os.path.join(out, f"strike_seed{args.target_seed}_{args.criterion}.png")
    fig.tight_layout(); fig.savefig(fname, dpi=130)
    print(f"plot: {fname}")


if __name__ == "__main__":
    main()
