"""Collision-aware M445 hammer environment.

The identified real-machine dynamics (IdentifiedPlant) drive the five joints
from valve commands, Newton poses the hammer by forward kinematics only, a
top-down camera raycasts the rocks and the floor into an elevation-map point
cloud, and the env reports the exact clearance between the arm surface and
every rock. No physics solver runs: rocks are fixed, the robot is never hit by
the camera rays.

    env = CollisionAwareEnv()
    env.add_rock((4.5, 0.8, 0.0), radius=0.9, seed=1)
    obs = env.reset()
    for _ in range(100):
        obs = env.step(valve_command)   # (5,) in [-1, 1]: TURN, BOOM, STICK, TELE, EE_PITCH
        env.render()                    # Newton GL viewer: robot, rocks, camera points
    obs.points[obs.valid == 1]          # raycast hits in the world frame
    obs.clearance                       # (n_rocks,) signed distance arm -> rock [m]
"""
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import newton
import warp as wp

from safe_mbrl.envs.collision_aware.camera import PointCloud, TopDownCamera
from safe_mbrl.envs.collision_aware.plant import MODEL_DIR, IdentifiedPlant
from safe_mbrl.envs.collision_aware.rocks import Rock, RockField
from safe_mbrl.m445_hammer_spec import HAMMER_URDF, JOINT_NAMES, SAFE_Q_MAX, SAFE_Q_MIN

# The URDF's meshes/*.dae are not shipped with safe_mbrl; first existing dir wins.
MESH_DIRS = ("/home/ccdonosoo/rsl/newton-ground/newton-ground/assets/robots/m445/meshes",
             "/home/ccdonosoo/menzi/menzi_ws/src/mole_media/robots/m445/meshes")
EE_BODY = "ROTO_BASE"
# Hammer tip in the ROTO_BASE frame: J_EE_ROLL (0.263, 0, 0.3645) + J_EE_YAW (0.1785, 0.0225, 0)
# + HAMMER_TO_ENDEFFECTOR_CONTACT (2.10, 0, 0.2), all fixed joints of the URDF.
TIP_OFFSET = wp.vec3(2.5415, 0.0225, 0.5645)
CLEARANCE_MAX = 10.0           # [m] distance queries stop beyond this


@dataclass
class Observation:
    time: float
    q: np.ndarray               # (5,) joint positions, JOINT_NAMES order
    qd: np.ndarray              # (5,) joint velocities
    tip: np.ndarray             # (3,) hammer tip, world frame
    cloud: PointCloud           # camera raycast of rocks + floor
    clearance: np.ndarray       # (n_rocks,) signed distance arm surface -> rock, < 0 inside
    floor_clearance: float      # lowest arm surface point above the floor
    collided: bool              # any clearance < 0

    @property
    def points(self) -> np.ndarray:
        return self.cloud.points

    @property
    def valid(self) -> np.ndarray:
        return self.cloud.valid


def transform_points(tf, points: np.ndarray) -> np.ndarray:
    """Apply a (x, y, z, qx, qy, qz, qw) transform to (N, 3) points."""
    t, q = np.asarray(tf[:3], np.float32), np.asarray(tf[3:], np.float32)
    v, w = q[:3], q[3]
    c = np.cross(v, points)
    return points + 2.0 * (w * c + np.cross(v, c)) + t


@wp.kernel
def surface_to_world_kernel(local: wp.array(dtype=wp.vec3), body: wp.array(dtype=int),
                            body_q: wp.array(dtype=wp.transform), world: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    world[i] = wp.transform_point(body_q[body[i]], local[i])


@wp.kernel
def clearance_kernel(points: wp.array(dtype=wp.vec3), mesh_ids: wp.array(dtype=wp.uint64), n_meshes: int,
                     floor_z: float, rock_min: wp.array(dtype=float), floor_min: wp.array(dtype=float)):
    i = wp.tid()
    p = points[i]
    wp.atomic_min(floor_min, 0, p[2] - floor_z)
    for k in range(n_meshes):
        q = wp.mesh_query_point_sign_normal(mesh_ids[k], p, CLEARANCE_MAX)
        if q.result:
            closest = wp.mesh_eval_position(mesh_ids[k], q.face, q.u, q.v)
            wp.atomic_min(rock_min, k, q.sign * wp.length(p - closest))


class CollisionAwareEnv:
    def __init__(self, camera: TopDownCamera | None = None, sampling: str = "mean", seed: int = 0,
                 urdf: str = HAMMER_URDF, model_dir: str = MODEL_DIR, floor_z: float = 0.0,
                 surface_samples_per_body: int = 2000, device: str = "cuda:0"):
        self.device, self.floor_z, self.time = device, floor_z, 0.0
        self.joint_names = JOINT_NAMES
        self.safe_q_min, self.safe_q_max = SAFE_Q_MIN.copy(), SAFE_Q_MAX.copy()

        self.spawn_height = 0.0              # set by _build_robot: chassis lifted onto the floor
        self.model = self._build_robot(urdf)
        self.state = self.model.state()
        self._q_idx, self._body_idx = self._index_joints_and_bodies()
        self._joint_q = wp.clone(self.model.joint_q)
        self._joint_qd = wp.clone(self.model.joint_qd)
        self._surface_local, self._surface_body = self._sample_arm_surface(surface_samples_per_body, seed)
        self._surface_world = wp.zeros(len(self._surface_body), dtype=wp.vec3, device=device)

        q_min, q_max = self.model.joint_limit_lower.numpy()[self._q_idx], self.model.joint_limit_upper.numpy()[self._q_idx]
        self.plant = IdentifiedPlant(q_min, q_max, model_dir=model_dir, sampling=sampling, seed=seed)
        self.dt = self.plant.dt
        self.rocks = RockField(device=device, floor_z=floor_z)
        self.camera = camera or TopDownCamera(device=device)
        self._viewer = None
        self.reset()

    # ------------------------------------------------------------------ scene
    def add_rock(self, position, radius: float = 0.8, **generation) -> Rock:
        """Place a procedural rock; see RockField.add for the generation parameters."""
        return self.rocks.add(position, radius, **generation)

    def scatter_rocks(self, n: int, **kwargs) -> list[Rock]:
        """Random natural-looking rock field in front of the machine; see RockField.scatter."""
        return self.rocks.scatter(n, **kwargs)

    def clear_rocks(self) -> None:
        self.rocks.clear()

    # ------------------------------------------------------------ env interface
    def reset(self, q0=None) -> Observation:
        """Start at rest at q0 (default: centre of the safe joint box), empty command history."""
        q0 = 0.5 * (self.safe_q_min + self.safe_q_max) if q0 is None else np.asarray(q0, np.float32)
        self.time = 0.0
        self.plant.reset(q0)
        self._pose_robot()
        return self.observe()

    def step(self, action) -> Observation:
        """Apply one valve command (5,) in [-1, 1] for dt, pose the hammer, observe."""
        self.plant.step(action)
        self.time += self.dt
        self._pose_robot()
        return self.observe()

    def observe(self) -> Observation:
        cloud = self.camera.raycast(self.rocks)
        clearance, floor_clearance = self.clearance()
        return Observation(time=self.time, q=self.plant.q, qd=self.plant.qd, tip=self.tip(), cloud=cloud,
                           clearance=clearance, floor_clearance=floor_clearance,
                           collided=bool((clearance < 0.0).any()))

    # --------------------------------------------------------------- geometry
    def tip(self) -> np.ndarray:
        pose = self.state.body_q.numpy()[self._body_idx[EE_BODY]]
        return np.asarray(wp.transform_point(wp.transform(*pose), TIP_OFFSET), np.float32)

    def link_poses(self) -> dict[str, np.ndarray]:
        """Body label -> (7,) world pose (x, y, z, qx, qy, qz, qw) of every arm body."""
        body_q = self.state.body_q.numpy()
        return {name: body_q[i] for name, i in self._body_idx.items()}

    def surface_points(self) -> np.ndarray:
        """(M, 3) samples of the arm surface in the world frame (for SDF-style checks)."""
        return self._surface_world.numpy()

    def check_pose(self, q) -> tuple[np.ndarray, np.ndarray, float]:
        """(tip, rock clearance, floor clearance) the arm WOULD have at q; the plant is untouched."""
        self._pose_robot(np.asarray(q, np.float32))
        tip, (clearance, floor) = self.tip(), self.clearance()
        self._pose_robot()
        return tip, clearance, floor

    def clearance(self) -> tuple[np.ndarray, float]:
        """Exact (n_rocks,) signed distance arm surface -> each rock, and the floor clearance."""
        n = max(len(self.rocks), 1)
        rock_min = wp.full(n, CLEARANCE_MAX, dtype=float, device=self.device)
        floor_min = wp.full(1, CLEARANCE_MAX, dtype=float, device=self.device)
        wp.launch(clearance_kernel, dim=len(self._surface_body), device=self.device,
                  inputs=[self._surface_world, self.rocks.mesh_ids, len(self.rocks), self.floor_z,
                          rock_min, floor_min])
        return rock_min.numpy()[:len(self.rocks)], float(floor_min.numpy()[0])

    # ---------------------------------------------------------------- viewer
    def render(self, point_radius: float = 0.03) -> bool:
        """Draw the scene in Newton's GL viewer. Returns False once the window is closed."""
        if self._viewer is None:
            self._viewer = newton.viewer.ViewerGL()
            self._viewer.set_model(self.model)
        v = self._viewer
        if not v.is_running():
            return False
        v.begin_frame(self.time)
        v.log_state(self.state)
        for rock in self.rocks.rocks:
            v.log_mesh(rock.label, rock.mesh.points, rock.mesh.indices, color=(0.55, 0.5, 0.45))
        cloud = self.camera.raycast(self.rocks)
        self._log_points(v, "camera_points", cloud.points[cloud.valid == 1], point_radius, (0.2, 0.8, 0.3))
        self._log_points(v, "camera", self.camera.position[None], 0.15, (0.9, 0.3, 0.2))
        v.end_frame()
        return True

    def _log_points(self, viewer, name: str, points: np.ndarray, radius: float, color) -> None:
        if len(points) == 0:
            viewer.log_points(name, None)
            return
        viewer.log_points(name, wp.array(points, dtype=wp.vec3, device=self.device), radii=radius,
                          colors=wp.full(len(points), wp.vec3(*color), dtype=wp.vec3, device=self.device))

    # -------------------------------------------------------------- internals
    def _build_robot(self, urdf: str) -> newton.Model:
        """FK-only Newton model of the machine, raised so the chassis rests on the floor."""
        root = ET.parse(urdf).getroot()
        mesh_dir = next((d for d in MESH_DIRS if os.path.isdir(d)), None)
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
        machine = newton.ModelBuilder(up_axis=newton.Axis.Z)
        machine.add_urdf(ET.tostring(root, encoding="unicode"), floating=False, collapse_fixed_joints=True)
        chassis_bottom = min(transform_points(machine.shape_transform[s],
                                              np.asarray(machine.shape_source[s].vertices) * machine.shape_scale[s])[:, 2].min()
                             for s in range(machine.shape_count)
                             if machine.shape_body[s] == -1 and machine.shape_type[s] == int(newton.GeoType.MESH))
        self.spawn_height = float(self.floor_z - chassis_bottom)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_builder(machine, xform=wp.transform(wp.vec3(0.0, 0.0, self.spawn_height)))
        return builder.finalize(device=self.device)

    def _index_joints_and_bodies(self):
        labels = list(self.model.joint_label)
        q_start = self.model.joint_q_start.numpy()
        q_idx = np.array([q_start[next(i for i, l in enumerate(labels) if l.endswith("/" + n) or l == n)]
                          for n in self.joint_names])
        body_idx = {lbl.split("/")[-1]: i for i, lbl in enumerate(self.model.body_label)}
        return q_idx, body_idx

    def _sample_arm_surface(self, per_body: int, seed: int):
        """Random vertices of every arm body's meshes, in the body frame."""
        rng = np.random.default_rng(seed)
        shape_body, shape_type = self.model.shape_body.numpy(), self.model.shape_type.numpy()
        shape_tf, shape_scale = self.model.shape_transform.numpy(), self.model.shape_scale.numpy()
        local, body = [], []
        for b in range(self.model.body_count):
            verts = []
            for s in np.where((shape_body == b) & (shape_type == int(newton.GeoType.MESH)))[0]:
                v = np.asarray(self.model.shape_source[s].vertices, np.float32) * shape_scale[s]
                verts.append(transform_points(shape_tf[s], v[rng.choice(len(v), per_body)]))
            if verts:
                verts = np.concatenate(verts)
                local.append(verts)
                body.append(np.full(len(verts), b))
        return (wp.array(np.concatenate(local), dtype=wp.vec3, device=self.device),
                wp.array(np.concatenate(body), dtype=int, device=self.device))

    def _pose_robot(self, q_arm=None) -> None:
        """Push joint positions (default: the plant's) into Newton; recompute body poses + arm surface."""
        q, qd = self._joint_q.numpy(), self._joint_qd.numpy()
        q[self._q_idx] = self.plant.q if q_arm is None else q_arm
        qd[self._q_idx] = self.plant.qd if q_arm is None else 0.0
        self._joint_q.assign(q)
        self._joint_qd.assign(qd)
        newton.eval_fk(self.model, self._joint_q, self._joint_qd, self.state)
        wp.launch(surface_to_world_kernel, dim=len(self._surface_body), device=self.device,
                  inputs=[self._surface_local, self._surface_body, self.state.body_q, self._surface_world])
