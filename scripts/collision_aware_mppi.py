"""MPPI reaching over rocks with a linearised body SDF for collision avoidance.

The env is the identified real machine. Every control tick the planner
  1. linearises the neural body SDF of the arm around the current q for the
     obstacle points of the camera cloud (everything above the floor): one linear
     model per link, unit-normal gradient, min over links along the rollout,
  2. runs MPPI on the same ensemble dynamics (mean) with the cost
        w_xyz . (tip - tip*)^2 + w_q . (q - q*)^2 + W_CLEAR exp(-(clearance - CLEAR_MIN) / CLEAR_LAMBDA)
        + action terms,
     The clearance term is a smooth potential, not a hinge: it is W_CLEAR at the margin, grows
     e-fold every CLEAR_LAMBDA inside it, and still has a slope far outside it, so the
     rollouts are biased away from rocks before any of them reaches the margin.
  3. applies the first valve command through the deploy runner's EMA filter,
     a <- EMA_ALPHA * a_mppi + (1 - EMA_ALPHA) * a_prev, plus one of three safety layers (--safety):
       shield  the deploy runner's gate: below SHIELD_MARGIN every joint whose command reduces the
               (linearised) clearance is zeroed. The rollouts apply the same gate, so the planner
               sees the shield's effect on its own plans.
       cbf     a velocity-level control barrier condition in the rollouts, no gate at execution:
                   grad_q h(q) . qdot >= -CBF_ALPHA * h(q),   h(q) = clearance(q) - CLEAR_MIN,
               with grad_q h the Jacobian row of the active link/point from the tick's linearisation
               (precomputed, step 1) and qdot the rollout's joint velocity after the step. With the
               linear clearance model this is the discrete barrier condition
               h_{k+1} >= (1 - CBF_ALPHA dt) h_k. Violations enter the cost as W_CBF * relu(-residual).
       none    neither; only the clearance potential of the cost keeps the arm off the rocks.
Targets are random joint configurations q* whose hammer tip hovers over a rock; the
joint target carries the tool orientation, the tip target the Cartesian goal.

    python scripts/collision_aware_mppi.py [--targets 20] [--sampling mean|member|normal] [--headless]
                                           [--safety shield|cbf|none] [--fast]
The loop is paced to the plant's 25 Hz unless --fast: each tick waits for the wall clock before stepping.
Writes collision_aware_mppi_clearance_<safety>.png (cbf: <safety>_a<alpha>) next to this file: histogram of
the exact arm-rock clearance over all ticks, with the linearised estimate the planner used, plus a
collision_aware_mppi_<tag>.npz with the per-tick clearance, residual, q and target. The summary reports, for
every mode, how often the exact clearance broke the discrete barrier condition on the real plant.
"""
import argparse
import os
import time
from importlib.resources import files
from types import SimpleNamespace

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import brax.math as bmath
import jax
import jax.numpy as jnp
import newton
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from flax import nnx
from mujoco import mjx

from robot_sdf import RobotSDF
from safe_mbrl.envs.collision_aware import CollisionAwareEnv
from safe_mbrl.envs.collision_aware.env import TIP_OFFSET
from safe_mbrl.envs.collision_aware.plant import IdentifiedPlant
from safe_mbrl.mpc.mppi import MPPI

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_m445_plant"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)

SDF_CKPT = "/home/ccdonosoo/rsl/robot_sdf/robot_ckpt/m445_hammer"
SDF_URDF = str(files("newton_ground") / "assets/robots/m445/m445_hammer_fixed_w_cabin.urdf")
ARM = jnp.array([2, 3, 4, 5])          # SDF components: BOOM, STICK, TELE, tool (world + cabin skipped)
TOOL = 5

HORIZON, SAMPLES, INIT_STD, TEMPERATURE = 30, 512, 0.3, 0.05
CLEAR_MIN, CLEAR_LAMBDA = 0.30, 0.07    # [m] potential reference distance and decay length
MAX_POINTS = 1024                       # obstacle points handed to the planner (padded)
W_XYZ = jnp.array([10.0, 10.0, 10.0])   # tip error weights per world axis
W_Q = jnp.array([20.0, 20.0, 20.0, 20.0, 20.0])   # joint error weights, JOINT_NAMES order
W_CLEAR, W_ACT, W_RATE = 1.5, 0.05, 0.5    # W_CLEAR = clearance cost per step AT the margin
EMA_ALPHA = 0.18                        # executed-command filter of the deploy runner (per tick)
SHIELD_MARGIN, SHIELD_EPS = 0.30, 1e-3   # gate active below this clearance; |grad| below eps is not gated
CBF_ALPHA, W_CBF = 10.0, 100.0           # [1/s] barrier decay rate; cost per m/s of constraint violation per step
REACH_TOL, Q_TOL, TARGET_TIMEOUT = 0.10, 0.10, 20.0    # tip [m], joint [rad | m], per-target [s]
TARGET_CLEAR_MIN = 0.40                 # [m] minimum exact arm-rock clearance of every target configuration


class ArmSDF:
    """Neural body SDF of the arm with FK, tip pose, and a joint-space linearisation."""

    def __init__(self, spawn_height: float):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_urdf(SDF_URDF, xform=wp.transform(wp.vec3(0.0, 0.0, spawn_height)),
                         collapse_fixed_joints=True, enable_self_collisions=False)
        for d in range(len(builder.joint_effort_limit)):      # URDF effort=0 is rejected by MuJoCo
            if builder.joint_effort_limit[d] <= 0.0:
                builder.joint_effort_limit[d] = 1.0e6
        self.sdf = RobotSDF(SDF_CKPT, builder.finalize(), mode="mjx")
        self.arm_state = jax.tree_util.tree_map(lambda x: x[ARM], self.sdf.stacked_state)
        self.tip_offset = jnp.array([*TIP_OFFSET])

    def fk(self, q):  # (6, 3) positions, (6, 4) wxyz quaternions of the SDF components
        data = mjx.kinematics(self.sdf.mjx_model, self.sdf.mjx_data.replace(qpos=q))
        pos = jnp.concatenate([jnp.zeros((1, 3)), data.xpos[1:]])
        rot = jnp.concatenate([jnp.array([[1.0, 0.0, 0.0, 0.0]]), data.xquat[1:]])
        return pos, rot

    def tip_pose(self, q):  # hammer tip position and strike axis, world frame
        pos, rot = self.fk(q)
        return pos[TOOL] + bmath.rotate(self.tip_offset, rot[TOOL]), bmath.rotate(jnp.array([1.0, 0.0, 0.0]), rot[TOOL])

    def _to_links(self, q, points):  # (K, N, 3) points in every arm link frame
        pos, rot = self.fk(q)
        return jax.vmap(lambda t, r: jax.vmap(lambda x: bmath.inv_rotate(x - t, r))(points))(pos[ARM], rot[ARM])

    def _d_local(self, state, x):  # one link's network SDF at a link-frame point, metres
        net = nnx.merge(self.sdf.graphdef, state)
        return (net((x - net.center.value) / net.scale.value) * net.scale.value)[0]

    def linearize(self, q_bar, points):
        """d(q, p) ~ min_i [d0_i + J_i (q - q_bar)]: values (K, N) and Jacobians (K, N, 5)."""
        local = self._to_links(q_bar, points)
        d0 = jax.vmap(jax.vmap(self._d_local, (None, 0)))(self.arm_state, local)
        grad = jax.vmap(jax.vmap(jax.grad(self._d_local, argnums=1), (None, 0)))(self.arm_state, local)
        normal = grad / jnp.linalg.norm(grad, axis=2, keepdims=True)     # a true SDF has |grad| = 1
        return d0, jnp.einsum("kni,knij->knj", normal, jax.jacfwd(self._to_links)(q_bar, points))

    @staticmethod
    def clearance(q, q_bar, d0, jac, valid):
        d = (d0 + jac @ (q - q_bar)).min(0)                               # (N,) min over links
        return jnp.where(valid, d, 1.0e3).min()

    @staticmethod
    def clearance_and_grad(q, q_bar, d0, jac, valid):
        """Linearised clearance and its joint gradient: the Jacobian row of the active link/point."""
        d = jnp.where(valid[None], d0 + jac @ (q - q_bar), 1.0e3)          # (K, N)
        k, n = jnp.unravel_index(jnp.argmin(d), d.shape)
        return d[k, n], jac[k, n]


def shield(a, clear, grad):
    """Deploy-runner gate: below the margin, zero every joint whose command reduces clearance."""
    blocked = (clear < SHIELD_MARGIN) & (a * grad < 0.0) & (jnp.abs(grad) > SHIELD_EPS)
    return jnp.where(blocked, 0.0, a)


def obstacle_points(obs, floor_z: float):
    """Camera hits above the floor, padded to MAX_POINTS with far-away dummies."""
    hits = obs.points[(obs.valid == 1) & (obs.points[:, 2] > floor_z + 0.05)][:MAX_POINTS]
    points = np.full((MAX_POINTS, 3), 100.0, np.float32)
    points[:len(hits)] = hits
    valid = np.zeros(MAX_POINTS, bool)
    valid[:len(hits)] = True
    return jnp.asarray(points), jnp.asarray(valid)


def sample_targets(env, sdf, n: int, rng) -> list[dict]:
    """Random safe-box configurations whose tip hovers above a rock, arm-rock clearance >= TARGET_CLEAR_MIN."""
    targets = []
    while len(targets) < n:
        q = rng.uniform(env.safe_q_min, env.safe_q_max).astype(np.float32)
        tip, clearance, floor = env.check_pose(q)
        over = [r for r in env.rocks.rocks
                if np.linalg.norm(tip[:2] - r.position[:2]) < 0.6 * r.radius and r.top + 0.35 < tip[2] < r.top + 0.9]
        if over and clearance.min() >= TARGET_CLEAR_MIN and floor > 0.2:
            pos, axis = sdf.tip_pose(jnp.asarray(q))
            targets.append(dict(q=q, pos=np.asarray(pos), axis=np.asarray(axis), rock=over[0].label,
                                clearance=float(clearance.min())))
    return targets


def cbf_residual(h, grad, qd, alpha):
    """grad_q h . qdot + alpha * h: >= 0 satisfies the barrier condition, < 0 violates it [m/s]."""
    return grad @ qd + alpha * h


def make_planner(sdf: ArmSDF, plant: IdentifiedPlant, safety: str, cbf_alpha: float = CBF_ALPHA,
                 w_cbf: float = W_CBF):
    """jit-compiled MPPI step: (mean actions, key, plant state, linearisation, target) -> new mean."""
    if safety not in ("shield", "cbf", "none"):
        raise ValueError(f"safety must be shield|cbf|none, got {safety!r}")
    step = plant._make_step("mean")
    mppi = MPPI(env=SimpleNamespace(action_size=5), horizon=HORIZON, nb_samples=SAMPLES,
                temperature=TEMPERATURE, init_std=INIT_STD)

    def reward(actions, rs, params, q_bar, d0, jac, valid, target_pos, target_q, prev_action):
        def rollout(carry, a_mppi):
            rs, params, a_prev = carry             # params ride in the carry: nnx.merge needs them
            a = EMA_ALPHA * a_mppi + (1.0 - EMA_ALPHA) * a_prev          # filter, then the shield
            if safety == "shield":
                a = shield(a, *ArmSDF.clearance_and_grad(rs.get_q(), q_bar, d0, jac, valid))
            rs = step(params, rs, a, jax.random.key(0))   # at the scan's trace level, not the jit's
            q, qd = rs.get_q(), rs.get_qd()
            tip, _ = sdf.tip_pose(q)
            clear, grad = ArmSDF.clearance_and_grad(q, q_bar, d0, jac, valid)   # grad: precomputed row
            cost = (W_XYZ @ (tip - target_pos) ** 2 + W_Q @ (q - target_q) ** 2
                    + W_CLEAR * jnp.exp(-(clear - CLEAR_MIN) / CLEAR_LAMBDA) + W_ACT * jnp.sum(a ** 2))
            if safety == "cbf":
                cost = cost + w_cbf * jax.nn.relu(-cbf_residual(clear - CLEAR_MIN, grad, qd, cbf_alpha))
            return (rs, params, a), cost
        _, costs = jax.lax.scan(rollout, (rs, params, prev_action), actions)
        rate = jnp.sum(jnp.diff(jnp.concatenate([prev_action[None], actions]), axis=0) ** 2)
        return -(costs.sum() + W_RATE * rate)

    @jax.jit
    def plan(mean, key, rs, params, q_bar, d0, jac, valid, target_pos, target_q, prev_action):
        return mppi.optimize(lambda a: reward(a, rs, params, q_bar, d0, jac, valid, target_pos, target_q,
                                              prev_action), key, mean)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, default=20)
    parser.add_argument("--sampling", choices=["mean", "member", "normal"], default="mean")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--safety", choices=["shield", "cbf", "none"], default="shield")
    parser.add_argument("--cbf-alpha", type=float, default=CBF_ALPHA, help="barrier decay rate [1/s]")
    parser.add_argument("--w-cbf", type=float, default=W_CBF, help="cost per m/s of barrier violation per step")
    parser.add_argument("--fast", action="store_true", help="do not pace the loop to the plant's wall clock")
    args = parser.parse_args()

    env = CollisionAwareEnv(sampling=args.sampling, seed=args.seed)
    env.scatter_rocks(6, area=((2.5, 6.5), (-3.0, 3.0)), radius=(0.35, 0.9), seed=7)
    sdf = ArmSDF(env.spawn_height)
    planner_plant = IdentifiedPlant(env.plant.q_min, env.plant.q_max, sampling="mean")
    plan = make_planner(sdf, planner_plant, args.safety, args.cbf_alpha, args.w_cbf)
    linearize, lin_clearance = jax.jit(sdf.linearize), jax.jit(ArmSDF.clearance)
    clearance_and_grad = jax.jit(ArmSDF.clearance_and_grad)
    rng, key = np.random.default_rng(args.seed), jax.random.key(args.seed)

    targets = sample_targets(env, sdf, args.targets, rng)
    t_clear = np.array([t["clearance"] for t in targets])
    print(f"target clearance [m]: {np.round(t_clear, 2).tolist()} (min {t_clear.min():.2f}, mean {t_clear.mean():.2f})")
    print(f"{len(env.rocks)} rocks | {len(targets)} targets over {[t['rock'] for t in targets]} | "
          f"MPPI horizon {HORIZON} x {env.dt} s, {SAMPLES} samples | clearance margin {CLEAR_MIN} m | "
          f"safety {args.safety}" + (f" (alpha {args.cbf_alpha} 1/s, w {args.w_cbf})" if args.safety == "cbf" else ""))

    obs = env.reset()
    mean, action = jnp.zeros((HORIZON, 5)), jnp.zeros(5)
    idx, t_start, reached, reach_err = 0, 0.0, [], []
    wall_start = next_tick = time.perf_counter()
    clear_exact, clear_lin, residual, gated = [], [], [], 0  # per tick, for the statistics
    tick_time, tick_target, tick_q = [], [], []
    while idx < len(targets):
        target = targets[idx]
        t_tick = time.perf_counter()
        points, valid = obstacle_points(obs, env.floor_z)
        q_bar = jnp.asarray(obs.q)
        d0, jac = linearize(q_bar, points)
        key, k = jax.random.split(key)
        t0 = time.perf_counter()
        mean, value = plan(mean, k, env.plant.state, planner_plant._params, q_bar, d0, jac, valid,
                           jnp.asarray(target["pos"]), jnp.asarray(target["q"]), action)
        mean.block_until_ready()
        plan_ms = 1e3 * (time.perf_counter() - t0)
        filtered = EMA_ALPHA * mean[0] + (1.0 - EMA_ALPHA) * action        # executed = filtered, then gated
        clear0, grad0 = clearance_and_grad(q_bar, q_bar, d0, jac, valid)
        action = shield(filtered, clear0, grad0) if args.safety == "shield" else filtered
        gated += int(bool(jnp.any(action != filtered)))
        mean = jnp.roll(mean, -1, axis=0).at[-1].set(0.0)
        obs = env.step(np.asarray(action))
        if not args.headless and not env.render():
            break
        next_tick += env.dt                             # 25 Hz timer: pause until the tick is due
        if not args.fast:
            time.sleep(max(0.0, next_tick - time.perf_counter()))

        xyz_err = obs.tip - target["pos"]
        tip_err, q_err = float(np.linalg.norm(xyz_err)), float(np.abs(obs.q - target["q"]).max())
        lin_clear = float(lin_clearance(jnp.asarray(obs.q), q_bar, d0, jac, valid))
        clear_exact.append(float(obs.clearance.min()))
        clear_lin.append(lin_clear)
        tick_time.append(obs.time); tick_target.append(idx); tick_q.append(np.asarray(obs.q))
        # barrier residual on the plant: the tick's precomputed gradient, the plant's velocity, exact h
        residual.append(float(cbf_residual(clear_exact[-1] - CLEAR_MIN, np.asarray(grad0), obs.qd, args.cbf_alpha)))
        if int(round(obs.time / env.dt)) % 25 == 0:
            print(f"t={obs.time:6.2f}s target {idx} ({target['rock']}) tip err {tip_err:.2f} m q err {q_err:.2f} | "
                  f"clearance exact {obs.clearance.min():.2f} lin {lin_clear:.2f} floor {obs.floor_clearance:.2f} | "
                  f"collided={obs.collided} plan {plan_ms:.0f} ms tick {1e3 * (time.perf_counter() - t_tick):.0f} ms")
        if tip_err < REACH_TOL and q_err < Q_TOL:
            reached.append(obs.time - t_start)
            reach_err.append(tip_err)
            print(f"  reached target {idx} in {reached[-1]:.1f} s | xyz err {np.round(xyz_err, 3)} m, mean |xyz| "
                  f"{np.abs(xyz_err).mean():.3f} m | joint err {np.round(obs.q - target['q'], 2)}")
            idx, t_start = idx + 1, obs.time
        elif obs.time - t_start > TARGET_TIMEOUT:
            print(f"  target {idx} timed out (tip err {tip_err:.2f} m, joint err {np.round(obs.q - target['q'], 2)})")
            idx, t_start = idx + 1, obs.time
    print(f"[{args.safety}] reached {len(reached)}/{len(targets)} targets, mean time "
          f"{np.mean(reached) if reached else float('nan'):.1f} s, mean tip err at reach "
          f"{100 * np.mean(reach_err) if reach_err else float('nan'):.1f} cm | "
          f"sim {obs.time:.1f} s in {time.perf_counter() - wall_start:.1f} s wall")

    clear_exact, clear_lin, residual = np.array(clear_exact), np.array(clear_lin), np.array(residual)
    if args.safety == "shield":
        print(f"shield gated a joint on {gated} of {len(clear_exact)} ticks ({100 * gated / len(clear_exact):.1f} %)")
    # discrete barrier condition on the EXACT clearance, model-free: h_{k+1} >= (1 - alpha dt) h_k
    h = clear_exact - CLEAR_MIN
    broken = h[1:] < (1.0 - args.cbf_alpha * env.dt) * h[:-1]
    print(f"barrier condition (alpha {args.cbf_alpha} 1/s) on the exact clearance: broken on {100 * broken.mean():.1f} % "
          f"of ticks ({100 * (broken & (h[:-1] > 0)).mean():.1f} % while outside the margin) | "
          f"residual grad.qd + alpha h: min {residual.min():.3f} m/s, negative on {100 * (residual < 0).mean():.1f} % of ticks")
    print(f"clearance over {len(clear_exact)} ticks: min {clear_exact.min():.2f} m, p1 {np.percentile(clear_exact, 1):.2f}, "
          f"p5 {np.percentile(clear_exact, 5):.2f}, median {np.median(clear_exact):.2f} | below {CLEAR_MIN} m: "
          f"{100 * (clear_exact < CLEAR_MIN).mean():.1f} % of ticks, below 0: {100 * (clear_exact < 0).mean():.1f} % | "
          f"linearised - exact: MAE {np.abs(clear_lin - clear_exact).mean() * 100:.1f} cm, max+ {(clear_lin - clear_exact).max() * 100:.1f} cm")
    out_dir = os.path.dirname(os.path.abspath(__file__))
    tag = f"cbf_a{args.cbf_alpha:g}" if args.safety == "cbf" else args.safety
    np.savez(os.path.join(out_dir, f"collision_aware_mppi_{tag}.npz"), time=np.array(tick_time),
             target=np.array(tick_target), q=np.array(tick_q), clear_exact=clear_exact, clear_lin=clear_lin,
             residual=residual, target_q=np.array([t["q"] for t in targets]),
             target_clearance=np.array([t["clearance"] for t in targets]),
             reached_time=np.array(reached), reach_err=np.array(reach_err))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
    bins = np.arange(0.0, max(clear_exact.max(), clear_lin.max()) + 0.05, 0.05)
    ax.hist(clear_exact, bins=bins, color="#2a78d6", label="exact (mesh)")
    ax.hist(clear_lin, bins=bins, histtype="step", color="#eb6834", lw=1.5, label="linearised (planner)")
    ax.axvline(CLEAR_MIN, color="0.3", ls="--", lw=1)
    ax.set_xlabel("arm-rock clearance [m]"); ax.set_ylabel("ticks"); ax.legend(frameon=False, fontsize=8)
    ax2.hist(100 * (clear_lin - clear_exact), bins=40, color="#1baf7a")
    ax2.axvline(0, color="0.3", lw=1); ax2.set_xlabel("linearised − exact [cm]"); ax2.set_ylabel("ticks")
    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(out_dir, f"collision_aware_mppi_clearance_{tag}.png")
    fig.savefig(out, dpi=200); print(f"saved {out}")
