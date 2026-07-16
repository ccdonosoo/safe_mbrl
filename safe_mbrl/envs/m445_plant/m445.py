"""Minimal Newton interface for the M445 with two interchangeable plants.

Both drive the 5 actuated joints [J_TURN, J_BOOM, J_STICK, J_TELE,
J_EE_PITCH] of the URDF from ``heap_env/rsc/m445`` (J_TURN patched
fixed -> revolute z, same as heap_m445.M445FK(slew=True)) at a 0.04 s tick:

* ``pd`` (default, M445MujocoPlant) — real MuJoCo physics via newton
  SolverMuJoCo with a joint-VELOCITY PD controller, mirroring newton-ground's
  ShovelFloating control path: action in [-1, 1] -> desired joint velocity
  (per-joint ranges), q_desired = current q, and FeedForwardController turns
  (q_des, qd_des) into computed torques (tau = M @ qdd_des + bias, gravity
  compensated) applied through control.joint_f.
* ``nn`` (M445NewtonPlant) — the learned RobotEnsemble (probabilistic
  ensemble trained on real machine data, checkpoint in ``m445_plant/model``).
  Each tick it predicts joint velocity from the q/qd/action history buffers
  and the state evolves by sampling from that predictive distribution; no
  physics solver runs and Newton only does FK + rendering.

The reset pose and hardware polarity come from the 5-DOF baseline deploy
config (mole_planning_system safe_mbrl_train_5dof.yaml): home = midpoint of
joint_pos_min/max. The motion envelope is the FULL URDF joint range (J_TURN
patched to +-pi) - wider than the barrier ranges the model was trained in, so
outside those the ensemble extrapolates and gets less accurate.

A fixed top-down 128x128 depth camera (SensorTiledCamera) hangs 7 m above
the workspace at x = 3 m looking straight down; the machine's shapes are
excluded from its render (see TopDownCamera), so it observes the ground only.

Current polarity: the deploy config carries joint_current_sign = [1,-1,1,1,1]
(J_BOOM wired inverted). The runner stores the PRE-polarity action a0 in the
training buffers and only multiplies by the sign when publishing the actuator
command, so the ensemble acts in pre-polarity convention (+action -> q
increases on every joint; verified by a per-joint probe). The joystick
therefore feeds the model directly; ``to_hardware_current()`` is where the
inversion belongs when commanding the real machine.

Run with a gamepad:
    python m445.py [--plant pd|nn] [--sampling normal|member|mean]

Joystick layout (matches newton-ground's rock_breaker examples):
    TURN     : left stick X
    BOOM     : right stick Y (push up = positive)
    STICK    : left stick Y  (inverted)
    TELE     : LT extend / LB retract
    EE_PITCH : right stick X
    RB       : reset to the home pose
"""

import os

# JAX must not preallocate 75% of the GPU - warp, the MuJoCo solver, and the
# tiled camera share the same device (set before jax initializes its backend).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import math
import time
import xml.etree.ElementTree as ET

import numpy as np
import pygame
import warp as wp

import newton
import newton.examples
from newton.sensors import SensorTiledCamera

import jax
import jax.numpy as jnp
import orbax.checkpoint as orbax
from flax import nnx

# Persistent XLA compilation cache: jitted functions (SDF value/grad, plant
# step) are compiled once and loaded from disk on later runs.
jax.config.update("jax_compilation_cache_dir",
                  os.path.expanduser("~/.cache/jax_m445_plant"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)

from newton_ground.controllers import FeedForwardController
from newton_ground.envs.utils.kernel_utils import (
    integrate_action_kernel,
    project_depth_to_points_kernel,
)

from safe_mbrl.models.robot_ensemble import RobotEnsemble
from safe_mbrl.utils.structs import RobotState

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_HERE, "model")
URDF_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "heap_env", "rsc", "m445", "m445_shovel_fixed_w_cabin.urdf"))
# The URDF references meshes/*.dae that are not shipped with this repo; the
# first existing directory wins. With none found the machine is simulated
# anyway, just rendered without meshes.
MESH_DIRS = (
    os.path.join(os.path.dirname(URDF_PATH), "meshes"),
    "/home/ccdonosoo/rsl/newton-ground/newton-ground/assets/robots/m445/meshes",
    "/home/ccdonosoo/rsl/ros_ws/src/mole_media/robots/m445/meshes",
)

JOINT_NAMES = ("J_TURN", "J_BOOM", "J_STICK", "J_TELE", "J_EE_PITCH")
TURN_LIMIT = (-3.1416, 3.1416)
SPAWN_HEIGHT = 0.0      # wheel radius: base frame sits at axle height

# PD-plant control constants, taken from newton-ground's excavator EnvConfig
# (the defaults ShovelFloating runs with). Order follows JOINT_NAMES.
ARM_KP = (6.0, 6.0, 6.0, 1000.0, 6.0)
ARM_KD = (10.0, 10.0, 10.0, 100.0, 10.0)
JOINT_VEL_UPPER = (0.8, 0.3, 0.6, 0.4, 0.8)          # rad/s (m/s for TELE)
JOINT_VEL_LOWER = (-0.8, -0.3, -0.6, -0.4, -0.8)
EFFORT_LIMITS = ((-2e6, 2e6), (-5e6, 5e6), (-4e6, 4e6), (-5e11, 5e11), (-2e6, 2e6))
SIM_SUBSTEPS = 6

# Static procedural rocks scattered in the middle of the camera footprint.
ROCK_COUNT = 8
ROCK_AREA = ((-0.75, 6.75), (-3.75, 3.75))   # x/y ranges: footprint minus a border
ROCK_RADIUS = (0.625, 1.25)                  # per-rock generator radius [m]
ROCK_SEED = 7

# Top-down ground camera (the machine is excluded from its render).
CAM_SIZE = (64, 64)          # (width, height)
CAM_POS = (3.0, 0.0, 7.0)      # centered over the workspace at x = 3 m, 7 m up
CAM_FOV_DEG = 60.0             # ~8 m square ground footprint from 7 m altitude
CAM_MAX_DEPTH = 20.0           # rays beyond this are dropped from the point cloud
CAM_POINT_RADIUS = 0.03        # 3 cm points when drawing the cloud in the viewer

# Deep-SDF arm <-> point-cloud distance monitor.
SDF_CKPT = "/home/ccdonosoo/rsl/robot_sdf/robot_ckpt/m445_shovel"
SDF_SKIP_COMPONENTS = 2        # drop 000 (static chassis) and 001 (CABIN)
SDF_SOFTMIN_BETA = 1000.0        # softmin sharpness [1/m]; higher -> closer to hard min
SDF_PRINT_EVERY = 5            # status-line cadence [steps]
SDF_BARRIER_MARGIN = 0.30      # [m] barrier active below this sdf; grad only computed then
SDF_GRAD_EPS = 1.0e-3          # |g_i| below this: joint has no clearance authority, not gated

# 5-DOF baseline deploy config (read live so a pull in menzi_ws updates the sim).
BASELINE_5DOF_YAML = ("/home/ccdonosoo/menzi/menzi_ws/src/mole_planning_system/"
                      "safe_mbrl/config/safe_mbrl_train_5dof.yaml")
# Snapshot of that config (2026-07-07), used when the menzi workspace is absent.
BASELINE_5DOF_FALLBACK = {
    "joint_pos_min":      [0.4, -1.38, 0.80, 0.00, 0.18],
    "joint_pos_max":      [0.4, -0.95, 1.70, 0.50, 1.9667],
    "joint_pos_min_safe": [-3.0, -1.38, 0.80, 0.00, 0.18],
    "joint_pos_max_safe": [3.0, -0.95, 1.70, 0.50, 1.9167],
    "joint_current_sign": [1.0, -1.0, 1.0, 1.0, 1.0],
}


def load_baseline_cfg() -> dict:
    if os.path.exists(BASELINE_5DOF_YAML):
        import yaml
        with open(BASELINE_5DOF_YAML) as f:
            raw = yaml.safe_load(f)
        return {k: raw.get(k, v) for k, v in BASELINE_5DOF_FALLBACK.items()}
    print(f"WARNING: {BASELINE_5DOF_YAML} not found, using the snapshot values")
    return dict(BASELINE_5DOF_FALLBACK)


def load_ensemble(model_dir: str = MODEL_DIR) -> RobotEnsemble:
    """RobotEnsemble with weights restored from the orbax checkpoint.

    The checkpoint stores the state of the INNER nnx ensemble (model.model),
    so restore against that state tree (checkpointer.load_model with the
    RobotEnsemble wrapper does not match the on-disk structure).
    """
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    model = RobotEnsemble(**cfg)

    _, _, state = nnx.split(model.model, nnx.RngState, ...)
    sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    restore_args = jax.tree.map(lambda _: orbax.ArrayRestoreArgs(sharding=sharding), state)
    state = orbax.PyTreeCheckpointer().restore(
        os.path.join(model_dir, "state"), item=state, restore_args=restore_args)
    nnx.update(model.model, state)
    return model


def _load_patched_urdf() -> ET.Element:
    """URDF root with J_TURN made revolute (z) and mesh paths made absolute."""
    root = ET.parse(URDF_PATH).getroot()

    j = root.find("joint[@name='J_TURN']")
    j.set("type", "revolute")
    ET.SubElement(j, "axis", {"xyz": "0 0 1"})
    ET.SubElement(j, "limit", {"lower": str(TURN_LIMIT[0]), "upper": str(TURN_LIMIT[1]),
                               "effort": "1e6", "velocity": "10"})

    mesh_dir = next((d for d in MESH_DIRS if os.path.isdir(d)), None)
    if mesh_dir is None:
        print("WARNING: no meshes/ directory found, rendering without machine meshes")
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                mesh = el.find("geometry/mesh")
                if mesh is None:
                    continue
                if mesh_dir is None:
                    link.remove(el)
                else:
                    mesh.set("filename", os.path.join(mesh_dir, os.path.basename(mesh.get("filename"))))
    return root


def _joint_limits(root: ET.Element):
    """Full mechanical range per joint from the (patched) URDF limit tags."""
    lo, hi = [], []
    for name in JOINT_NAMES:
        lim = root.find(f"joint[@name='{name}']/limit")
        lo.append(float(lim.get("lower")))
        hi.append(float(lim.get("upper")))
    return jnp.array(lo, jnp.float32), jnp.array(hi, jnp.float32)


def _find_joint(labels, name):
    for i, lbl in enumerate(labels):
        if lbl == name or lbl.endswith("/" + name):
            return i
    raise KeyError(f"joint {name!r} not found in {labels}")


def _add_rocks(builder: newton.ModelBuilder, seed: int = ROCK_SEED) -> None:
    """Scatter ROCK_COUNT static procedural rocks over the middle of the
    camera footprint. Uses newton-ground's WarpRockGenerator (convex hull,
    like RockComponent) but as STATIC shapes resting on the ground plane.
    Labels 'rock_*' keep them visible to the top-down camera and hence in the
    point cloud / SDF barrier; in the PD plant the shovel collides with them.
    """
    from newton_ground.procedural_generation.rock_generator import WarpRockGenerator

    rng = np.random.default_rng(seed)
    gen = WarpRockGenerator(device=str(wp.get_device()))
    (x_lo, x_hi), (y_lo, y_hi) = ROCK_AREA
    for i in range(ROCK_COUNT):
        tm = gen.generate_rock(
            radius=float(rng.uniform(*ROCK_RADIUS)),
            subdivisions=2,
            warp_vector=tuple(rng.uniform(0.4, 0.8, 3)),
            noise_scale=float(rng.uniform(0.3, 0.6)),
            noise_frequency=float(rng.uniform(0.3, 0.5)),
            octaves=int(rng.choice([4, 5, 6])),
            persistence=float(rng.uniform(0.4, 0.6)),
            n_plane_clips=int(rng.integers(14, 21)),
            seed=seed + i,
        ).convex_hull
        verts = np.asarray(tm.vertices, np.float32)
        mesh = newton.Mesh(vertices=verts, indices=tm.faces.flatten().astype(np.int32))
        pos = wp.vec3(float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi)),
                      float(-verts[:, 2].min()))            # bottom resting on z = 0
        builder.add_shape_mesh(body=-1, mesh=mesh, xform=wp.transform(pos),
                               label=f"rock_{i}")


def _look_at_transform(pos, target, up_hint=(1.0, 0.0, 0.0)) -> wp.transform:
    """Camera pose for SensorTiledCamera: camera -Z points at target (same
    convention as newton-ground's look_at_quat). up_hint must not be parallel
    to the view direction - for a straight-down camera use world +x."""
    pos = np.asarray(pos, np.float64)
    f = np.asarray(target, np.float64) - pos
    f = f / np.linalg.norm(f)
    r = np.cross(f, np.asarray(up_hint, np.float64))
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    m = wp.mat33(r[0], u[0], -f[0], r[1], u[1], -f[1], r[2], u[2], -f[2])
    return wp.transform(wp.vec3(*pos), wp.quat_from_matrix(m))


class TopDownCamera:
    """Raytraced depth camera looking straight down over the workspace.

    The machine must NOT appear in the image: the raytracer only hits shapes
    that are in the shape BVH, and BVH membership is decided by the VISIBLE
    flag at build time. So the machine's shapes are flagged invisible, the
    BVH is rebuilt (enabled set = ground plane only), the sensor snapshots
    it, and the flags are restored so the GL viewer still draws the machine.
    Per-frame bvh_refit_shapes() only refits the enabled set and
    Model.collide() uses its own collision pipeline - neither re-reads flags.
    """

    def __init__(self, model: newton.Model, state: newton.State,
                 pos=CAM_POS, size=CAM_SIZE, fov_deg=CAM_FOV_DEG):
        self.model = model
        w, h = size

        flags = model.shape_flags.numpy()
        ground_only = flags.copy()
        types = model.shape_type.numpy()
        labels = list(model.shape_label)
        machine = np.array([types[i] != int(newton.GeoType.PLANE)
                            and "rock" not in (labels[i] or "")
                            for i in range(len(types))])
        ground_only[machine] &= ~int(newton.ShapeFlags.VISIBLE)
        model.shape_flags.assign(ground_only)
        model.bvh_build_shapes(state)
        self.sensor = SensorTiledCamera(model)
        model.shape_flags.assign(flags)

        self.rays = self.sensor.utils.compute_pinhole_camera_rays(w, h, math.radians(fov_deg))
        self.depth_image = self.sensor.utils.create_depth_image_output(w, h)
        # (num_cameras, num_worlds) = (1, 1); straight down, image up = world +x
        target = (pos[0], pos[1], 0.0)
        self.transform = wp.array([[_look_at_transform(pos, target)]], dtype=wp.transform)

        # Depth -> world-space point cloud (drawn in the viewer, 3 cm points).
        self._w, self._h = w, h
        self.points = wp.zeros((h, w, 1), dtype=wp.vec3)
        self.colors = wp.zeros((h, w, 1), dtype=wp.vec3)
        self.valid = wp.zeros((h, w, 1), dtype=wp.int32)
        self.points_flat = self.points.reshape((h * w,))     # views for log_points
        self.colors_flat = self.colors.reshape((h * w,))
        self.point_radius = CAM_POINT_RADIUS

    def update(self, state: newton.State) -> None:
        self.model.bvh_refit_shapes(state)    # ground is static, but keep the contract
        self.sensor.update(state, self.transform, self.rays, depth_image=self.depth_image)
        wp.launch(
            project_depth_to_points_kernel,
            dim=(self._h, self._w, 1),
            inputs=[self.depth_image, self.rays, self.transform,
                    self.points, self.colors, self.valid, CAM_MAX_DEPTH, self._w],
        )

    def depth(self) -> np.ndarray:
        """(H, W) ray-hit distance in meters; negative = no hit."""
        return self.depth_image.numpy()[0, 0]


class ArmGroundSDF:
    """Deep-SDF distance between the arm and the camera point cloud.

    Loads the m445_shovel RobotSDF checkpoint (one DeepSDF per kinematic
    component: 000 static chassis, 001 CABIN, 002 BOOM, 003 STICK, 004 TELE,
    005 shovel) as a function of the 5 joint positions. Components 000/001
    are EXCLUDED from the prediction; the value is a soft min over the
    remaining meshes x all point-cloud points,
        softmin(v) = -logsumexp(-beta * v) / beta,
    i.e. a smooth "distance of the closest arm part to the closest point".

    FK runs on its own newton->mjx model built from the newton-ground URDF
    the checkpoint was trained on, spawned at SPAWN_HEIGHT so the poses live
    in the same world frame as the sim and the camera cloud.
    """

    def __init__(self, ckpt_path: str = SDF_CKPT):
        from importlib.resources import files
        from robot_sdf import RobotSDF

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_urdf(
            str(files("newton_ground") / "assets/robots/m445/m445_shovel_fixed_w_cabin.urdf"),
            xform=wp.transform(wp.vec3(0.0, 0.0, SPAWN_HEIGHT)),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
        )
        # effort="0.0" in the URDF -> invalid MuJoCo actfrcrange on conversion
        for d in range(len(builder.joint_effort_limit)):
            if builder.joint_effort_limit[d] <= 0.0:
                builder.joint_effort_limit[d] = 1.0e6
        self.robot_sdf = RobotSDF(ckpt_path, builder.finalize(), mode="mjx")

        skip, beta = SDF_SKIP_COMPONENTS, SDF_SOFTMIN_BETA

        @jax.jit
        def value(q, points, valid):
            sdf = self.robot_sdf.sdf_per_component(q, points)[skip:]   # (C - skip, N)
            sdf = jnp.where(valid[None, :] > 0, sdf, 1.0e3)            # no-hit pixels drop out
            return -jax.nn.logsumexp(-beta * sdf) / beta

        self._value = value
        self._grad = jax.jit(jax.grad(value, argnums=0))   # d softmin / d q, through FK

        # Compile BOTH jits now, at startup, for the runtime point-cloud shape.
        # Otherwise the grad compiles on the FIRST barrier activation - i.e. a
        # multi-second freeze exactly when the arm reaches the margin. With the
        # persistent compilation cache above, later runs load this from disk.
        n = CAM_SIZE[0] * CAM_SIZE[1]
        q0 = jnp.zeros(len(JOINT_NAMES), jnp.float32)
        p0 = jnp.zeros((n, 3), jnp.float32)
        v0 = jnp.ones(n, jnp.int32)
        t0 = time.perf_counter()
        self._value(q0, p0, v0).block_until_ready()
        self._grad(q0, p0, v0).block_until_ready()
        print(f"ArmGroundSDF: value+grad compiled/loaded in {time.perf_counter() - t0:.1f} s")

    def __call__(self, q, points, valid) -> float:
        """q: (5,) joint positions; points: (N, 3) world; valid: (N,) int."""
        return float(self._value(jnp.asarray(q, jnp.float32),
                                 jnp.asarray(points, jnp.float32),
                                 jnp.asarray(valid, jnp.int32)))

    def grad(self, q, points, valid) -> np.ndarray:
        """(5,) gradient of the softmin wrt joint position: the joint-space
        direction that INCREASES clearance from the point cloud."""
        return np.asarray(self._grad(jnp.asarray(q, jnp.float32),
                                     jnp.asarray(points, jnp.float32),
                                     jnp.asarray(valid, jnp.int32)))


def apply_sdf_barrier(action, sdf_val, sdf_grad):
    """Per-joint hard gate on the point cloud, same shape as the deploy
    apply_barrier (menzi_ws mole_planning_system safe_mbrl action_barrier.py)
    but keyed by the deep-SDF gradient instead of per-joint position limits.

    Inactive while sdf_val >= SDF_BARRIER_MARGIN (sdf_grad may be None then).
    Below the margin, every joint whose OWN commanded motion decreases the
    clearance (a_i * g_i < 0) is set to zero velocity command; joints moving
    along or away from the cloud pass through. All surviving terms then have
    a_i * g_i >= 0, so d(sdf)/dt >= 0 no matter how the plant scales each
    joint's action to velocity - pushing into the barrier STOPS rather than
    creeping (unlike a projection in action space).

    Returns (gated_action, s) with s = action . ghat (None when inactive).
    """
    a = np.asarray(action, np.float32).copy()
    if sdf_val >= SDF_BARRIER_MARGIN or sdf_grad is None:
        return a, None
    g = np.asarray(sdf_grad, np.float32)
    g_norm = float(np.linalg.norm(g))
    if g_norm < 1.0e-8:
        return a, None
    s = float(a @ (g / g_norm))
    a[(a * g < 0.0) & (np.abs(g) > SDF_GRAD_EPS)] = 0.0
    return a, s


class M445NewtonPlant:
    """Learned-ensemble dynamics + Newton FK/rendering for the M445.

    step(action) advances the NN plant one dt and re-poses the Newton bodies;
    ``self.model`` / ``self.state`` are what a viewer needs to draw the scene.
    """

    def __init__(self, sampling: str = "normal", seed: int = 0):
        # Learned dynamics (the actual plant)
        self.ensemble = load_ensemble()
        self.jd = self.ensemble.joint_dim
        self.bd = self.ensemble.buffer_dim
        self.dt = self.ensemble._dt
        if self.jd != len(JOINT_NAMES):
            raise ValueError(f"checkpoint has joint_dim={self.jd}, expected {len(JOINT_NAMES)}")

        # Home pose + hardware polarity from the 5-DOF baseline deploy config.
        base = load_baseline_cfg()
        lo = np.asarray(base["joint_pos_min"], np.float32)
        hi = np.asarray(base["joint_pos_max"], np.float32)
        self.q_home = jnp.asarray(0.5 * (lo + hi), jnp.float32)
        # Hardware output polarity (J_BOOM = -1). NOT applied to the model input:
        # the ensemble acts on pre-polarity actions, see to_hardware_current().
        self.current_sign = np.asarray(base["joint_current_sign"], np.float32)

        # Newton model: FK + rendering only, no solver. The motion envelope is
        # the full URDF range - the model extrapolates outside the trained
        # barrier ranges (boom [-1.38,-0.95], tele [0,0.5], ...), less accurately.
        urdf = _load_patched_urdf()
        self.q_lower, self.q_upper = _joint_limits(urdf)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_urdf(
            ET.tostring(urdf, encoding="unicode"),
            xform=wp.transform(wp.vec3(0.0, 0.0, SPAWN_HEIGHT)),
            floating=False,
            collapse_fixed_joints=True,
        )
        builder.add_ground_plane()
        _add_rocks(builder)
        self.model = builder.finalize()
        self.state = self.model.state()

        # NN joint order -> Newton generalized-coordinate indices
        labels = list(self.model.joint_label)
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        joint_idx = [_find_joint(labels, n) for n in JOINT_NAMES]
        self._q_idx = np.array([q_start[j] for j in joint_idx])
        self._qd_idx = np.array([qd_start[j] for j in joint_idx])
        self._q_host = self.model.joint_q.numpy().copy()
        self._qd_host = self.model.joint_qd.numpy().copy()
        self._joint_q = wp.clone(self.model.joint_q)
        self._joint_qd = wp.clone(self.model.joint_qd)

        self._key = jax.random.key(seed)
        self._step_fn = self._make_step_fn(sampling)
        self.reset()

    def _make_step_fn(self, sampling: str):
        if sampling not in ("normal", "member", "mean"):
            raise ValueError(f"sampling must be normal|member|mean, got {sampling!r}")
        graphdef, self._params = nnx.split(self.ensemble.model)
        input_idx = self.ensemble._input_idx
        mode, dt, jd = self.ensemble.mode, self.dt, self.jd
        num_ensembles = self.ensemble.model.num_ensembles
        lo, hi = self.q_lower, self.q_upper

        # nnx params must enter the jit as arguments (a closure capture would
        # cross trace levels), so they are threaded through like in heap_m445.
        @jax.jit
        def step_fn(params, rs: RobotState, action: jax.Array, key: jax.Array) -> RobotState:
            net = nnx.merge(graphdef, params)
            x = rs.ravel() if input_idx is None else rs.ravel()[input_idx]
            mu, sig = jnp.split(net(x), 2, axis=-1)              # (E, jd) each
            k_member, k_noise = jax.random.split(key)
            m = jax.random.randint(k_member, (), 0, num_ensembles)
            if sampling == "mean":
                v = mu.mean(axis=0)
            elif sampling == "member":                            # TS1, member mean only
                v = mu[m]
            else:                                                 # TS1 + aleatoric sample
                v = mu[m] + sig[m] * jax.random.normal(k_noise, (jd,))

            qd_next = rs.get_qd() + v if mode == "dv" else v
            q_next = jnp.clip(rs.get_q() + qd_next * dt, lo, hi)
            qd_next = (q_next - rs.get_q()) / dt                  # velocity after the limit clamp

            q_buf = jnp.roll(rs.q_buffer, -jd).at[-jd:].set(q_next)
            qd_buf = jnp.roll(rs.qd_buffer, -jd).at[-jd:].set(qd_next)
            act_buf = jnp.roll(rs.act_buffer, -jd).at[-jd:].set(action)
            return rs.replace(q_buffer=q_buf, qd_buffer=qd_buf, act_buffer=act_buf)

        return step_fn

    def reset(self, q0=None) -> np.ndarray:
        if q0 is None:
            q0 = self.q_home                       # 5-DOF baseline start pose
        self.rs = RobotState.create(jnp.asarray(q0, jnp.float32),
                                    buffer_size=self.bd, q_dim=self.jd)
        self._sync_fk()
        return np.asarray(self.rs.get_q())

    def step(self, action) -> np.ndarray:
        """action: (5,) normalized command in [-1, 1] -> joint positions (5,)."""
        action = jnp.clip(jnp.asarray(action, jnp.float32), -1.0, 1.0)
        self._key, key = jax.random.split(self._key)
        self.rs = self._step_fn(self._params, self.rs, action, key)
        self._sync_fk()
        return np.asarray(self.rs.get_q())

    def to_hardware_current(self, action) -> np.ndarray:
        """Model/planner action -> real actuator command polarity.

        The deploy runner multiplies by joint_current_sign only when PUBLISHING
        (J_BOOM wired inverted); the training buffers hold the pre-polarity
        action, so simulation and planning never apply this sign.
        """
        return np.asarray(action, np.float32) * self.current_sign

    def _sync_fk(self):
        """Push the NN joint state into Newton and recompute body poses."""
        self._q_host[self._q_idx] = np.asarray(self.rs.get_q(), np.float32)
        self._qd_host[self._qd_idx] = np.asarray(self.rs.get_qd(), np.float32)
        self._joint_q.assign(self._q_host)
        self._joint_qd.assign(self._qd_host)
        newton.eval_fk(self.model, self._joint_q, self._joint_qd, self.state)


class M445MujocoPlant:
    """MuJoCo physics + joint-velocity PD controller (default plant).

    Same control path as newton-ground's ShovelFloating: each 0.04 s tick the
    action in [-1, 1] maps to a desired joint velocity (per-joint ranges from
    JOINT_VEL_LOWER/UPPER), q_desired tracks the current q, and the
    FeedForwardController computes tau = M @ qdd_des + bias (gravity
    compensated, effort-limited) which SolverMuJoCo integrates over
    SIM_SUBSTEPS substeps. Exposes the same interface as M445NewtonPlant.
    """

    def __init__(self, seed: int = 0):
        del seed                          # deterministic; ctor parity with the NN plant
        self.jd = len(JOINT_NAMES)
        self.dt = 0.04                    # control tick, matches the NN plant / joystick
        self.sim_dt = self.dt / SIM_SUBSTEPS

        # Home pose + hardware polarity from the 5-DOF baseline deploy config.
        base = load_baseline_cfg()
        lo = np.asarray(base["joint_pos_min"], np.float32)
        hi = np.asarray(base["joint_pos_max"], np.float32)
        self.q_home = 0.5 * (lo + hi)
        self.current_sign = np.asarray(base["joint_current_sign"], np.float32)

        urdf = _load_patched_urdf()
        self.q_lower, self.q_upper = _joint_limits(urdf)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        # ke/kd <= 0 -> MuJoCo's default solreflimit handles joint limits
        # (same rationale as newton-ground's excavator config).
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=-1.0, limit_kd=-1.0, friction=0.7)
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1e2
        builder.default_shape_cfg.kf = 1e2
        builder.default_shape_cfg.mu = 0.5
        builder.add_urdf(
            ET.tostring(urdf, encoding="unicode"),
            xform=wp.transform(wp.vec3(0.0, 0.0, SPAWN_HEIGHT)),
            floating=False,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
        )
        _add_rocks(builder)
        # The URDF declares effort="0.0"; MuJoCo rejects actfrcrange [0, 0],
        # so seed real effort limits before the solver converts the model.
        for name, (elo, ehi) in zip(JOINT_NAMES, EFFORT_LIMITS):
            mag = max(abs(elo), abs(ehi))
            j = _find_joint(builder.joint_label, name)
            d0 = builder.joint_qd_start[j]
            lin, ang = builder.joint_dof_dim[j]
            for d in range(lin + ang):
                builder.joint_effort_limit[d0 + d] = mag
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.state = self.model.state()
        self._state_next = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            integrator="implicitfast",
            cone="elliptic",
            impratio=1,
            iterations=100,
            ls_iterations=20,
            nconmax=4000,
            njmax=4000,
            jacobian="dense",             # dense required by FeedForwardController
            use_mujoco_contacts=False,
        )

        joint_idx = [_find_joint(list(self.model.joint_label), n) for n in JOINT_NAMES]
        self.controller = FeedForwardController(
            model=self.model,
            solver=self.solver,
            num_worlds=1,
            kp_world=list(ARM_KP),
            kd_world=list(ARM_KD),
            ctrl_joint_indices=joint_idx,
            effort_limits=[list(e) for e in EFFORT_LIMITS],
        )
        self._q_idx = self.controller.ctrl_to_q.numpy()

        self.q_desired = wp.zeros((1, self.jd), dtype=wp.float32)
        self.qd_desired = wp.zeros((1, self.jd), dtype=wp.float32)
        self._action_wp = wp.zeros((1, self.jd), dtype=wp.float32)
        self.vel_upper = wp.array(list(JOINT_VEL_UPPER), dtype=wp.float32)
        self.vel_lower = wp.array(list(JOINT_VEL_LOWER), dtype=wp.float32)
        self._q_host = self.model.joint_q.numpy().copy()

        self.reset()

    def reset(self, q0=None) -> np.ndarray:
        if q0 is None:
            q0 = self.q_home                       # 5-DOF baseline start pose
        q0 = np.asarray(q0, np.float32)
        self._q_host[self._q_idx] = q0
        self.state.joint_q.assign(self._q_host)
        self.state.joint_qd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.q_desired.assign(q0.reshape(1, -1))
        self.qd_desired.zero_()
        # Teleport: refresh the solver's internal MuJoCo data (as in
        # newton-ground's reset_selected), else it keeps stepping the old pose.
        self.solver._update_mjc_data(self.solver.mjw_data, self.model, self.state)
        return self.q()

    def step(self, action) -> np.ndarray:
        """action: (5,) normalized velocity command in [-1, 1] -> joint positions (5,)."""
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0).reshape(1, self.jd)
        self._action_wp.assign(a)
        self.model.collide(self.state, self.contacts)
        for _ in range(SIM_SUBSTEPS):
            self.state.clear_forces()
            wp.launch(
                integrate_action_kernel,
                dim=(1, self.jd),
                inputs=[
                    self.state.joint_q,
                    self._action_wp,
                    self.q_desired,
                    self.qd_desired,
                    self.controller.ctrl_to_q,
                    self.model.joint_coord_count,
                    1.0,                  # velocity_scale (unused: per-joint ranges above)
                    self.sim_dt,
                    self.vel_lower,
                    self.vel_upper,
                ],
            )
            tau = self.controller.compute_control(
                self.model, self.solver, self.state, self.q_desired, self.qd_desired)
            self.control.joint_f.assign(tau)
            self.solver.step(self.state, self._state_next, self.control, self.contacts, self.sim_dt)
            self.state, self._state_next = self._state_next, self.state
        return self.q()

    def q(self) -> np.ndarray:
        return self.state.joint_q.numpy()[self._q_idx]

    def to_hardware_current(self, action) -> np.ndarray:
        """Model/planner action -> real actuator command polarity (J_BOOM inverted)."""
        return np.asarray(action, np.float32) * self.current_sign


class JoystickController:
    """Maps an XInput gamepad to the 5 joints (layout as in newton-ground)."""

    _AXIS_LX, _AXIS_LY, _AXIS_LT, _AXIS_RX, _AXIS_RY = 0, 1, 2, 3, 4
    _BTN_LB, _BTN_RB = 4, 5

    def __init__(self, deadzone: float = 0.15):
        self.deadzone = deadzone
        self.joystick = None
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Gamepad: {self.joystick.get_name()} "
                  f"(axes={self.joystick.get_numaxes()}, buttons={self.joystick.get_numbuttons()})")
        else:
            print("No gamepad found - actions will be zero.")

    def _axis(self, idx: int) -> float:
        if idx >= self.joystick.get_numaxes():
            return 0.0
        raw = self.joystick.get_axis(idx)
        if abs(raw) < self.deadzone:
            return 0.0
        sign = 1.0 if raw > 0 else -1.0
        return sign * (abs(raw) - self.deadzone) / (1.0 - self.deadzone)

    def _button(self, idx: int) -> int:
        if idx >= self.joystick.get_numbuttons():
            return 0
        return self.joystick.get_button(idx)

    def get_action(self) -> np.ndarray:
        action = np.zeros(5, dtype=np.float32)
        if self.joystick is None or not self.joystick.get_init():
            return action
        pygame.event.pump()
        action[0] = -self._axis(self._AXIS_LX)                    # TURN
        action[1] = -self._axis(self._AXIS_RY)                    # BOOM (push up = positive)
        action[2] = -self._axis(self._AXIS_LY)                    # STICK (inverted)
        action[3] = max(0.0, self._axis(self._AXIS_LT)) - float(self._button(self._BTN_LB))  # TELE
        action[4] = self._axis(self._AXIS_RX)                     # EE_PITCH
        return action

    def reset_pressed(self) -> bool:
        if self.joystick is None or not self.joystick.get_init():
            return False
        return bool(self._button(self._BTN_RB))


def main():
    parser = newton.examples.create_parser()
    parser.add_argument("--plant", choices=("pd", "nn"), default="pd",
                        help="pd: MuJoCo physics + joint-velocity PD (default); "
                             "nn: learned ensemble dynamics")
    parser.add_argument("--sampling", choices=("normal", "member", "mean"), default="mean",
                        help="nn plant only. normal: random member + aleatoric noise; "
                             "member: random member mean; mean: ensemble mean")
    parser.add_argument("--deadzone", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    viewer, args = newton.examples.init(parser)

    if args.plant == "nn":
        plant = M445NewtonPlant(sampling=args.sampling, seed=args.seed)
        detail = f"ensemble dynamics, buffer {plant.bd}, sampling '{args.sampling}'"
    else:
        plant = M445MujocoPlant(seed=args.seed)
        detail = f"MuJoCo + velocity PD, {SIM_SUBSTEPS} substeps of {plant.sim_dt*1000:.1f} ms"
    cam = TopDownCamera(plant.model, plant.state)
    arm_sdf = ArmGroundSDF()
    viewer.set_model(plant.model)
    viewer.set_camera(pos=wp.vec3(10.0, -10.0, 6.0), pitch=-20.0, yaw=135.0)
    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()    # custom loop: GL keeps the "Loading" splash up otherwise

    pad = JoystickController(deadzone=args.deadzone)
    print(f"Plant '{args.plant}': {plant.jd} joints, dt {plant.dt}s ({detail})")
    print("Controls: LX=turn  RY=boom  LY=stick  LT/LB=tele  RX=pitch  |  RB=reset  |  close window to exit")

    sim_time = 0.0
    step_count = 0
    rb_was_down = False
    q = plant.reset()
    cam.update(plant.state)

    # One step + one rendered frame per tick, capped at 1/dt (25 Hz) wall-clock.
    while viewer.is_running():
        tick_start = time.perf_counter()

        action = pad.get_action()
        rb_down = pad.reset_pressed()
        if rb_down and not rb_was_down:
            q = plant.reset()
        rb_was_down = rb_down

        if viewer.should_step():
            # SDF barrier at the CURRENT pose: gradient is only computed when
            # the arm is within the margin, then the approaching component of
            # the action (a . grad < 0) is removed before stepping.
            pts_np = cam.points_flat.numpy()
            valid_np = cam.valid.numpy().ravel()
            sdf_val = arm_sdf(q, pts_np, valid_np)
            sdf_grad = (arm_sdf.grad(q, pts_np, valid_np)
                        if sdf_val < SDF_BARRIER_MARGIN else None)
            action, adotg = apply_sdf_barrier(action, sdf_val, sdf_grad)

            q = plant.step(action)
            cam.update(plant.state)
            sim_time += plant.dt
            step_count += 1
            if step_count % SDF_PRINT_EVERY == 0:
                barrier = "off" if adotg is None else f"ON a.g {adotg:+5.2f}"
                fmt = " ".join(f"{v:+6.2f}" for v in q)
                print(f"\r  q [{fmt}]  act [{' '.join(f'{a:+4.1f}' for a in action)}]  "
                      f"sdf {sdf_val:+7.3f} m  barrier {barrier}  ",
                      end="", flush=True)

        viewer.begin_frame(sim_time)
        viewer.log_state(plant.state)
        viewer.log_points("camera_pcd", cam.points_flat, cam.point_radius, cam.colors_flat)
        viewer.end_frame()

        leftover = plant.dt - (time.perf_counter() - tick_start)
        if leftover > 0.0:
            time.sleep(leftover)

    pygame.quit()


if __name__ == "__main__":
    main()
