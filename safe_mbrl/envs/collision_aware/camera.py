"""Top-down raycast camera.

Casts one ray per pixel against the rocks and the floor plane only, so the
returned cloud is an elevation map of the terrain: the robot is never hit.
"""
import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from safe_mbrl.envs.collision_aware.rocks import RockField

NO_HIT, FLOOR = -1, 0          # hit_id codes; rock k is FLOOR + 1 + k


@dataclass
class PointCloud:
    points: np.ndarray         # (N, 3) world hit points (undefined where valid == 0)
    valid: np.ndarray          # (N,) 1 where a ray hit something within max_distance
    depth: np.ndarray          # (H, W) ray length [m], -1 for no hit
    hit_id: np.ndarray         # (N,) NO_HIT, FLOOR or rock index + 1
    points_wp: wp.array        # same points on the device, for the viewer
    valid_wp: wp.array


@wp.kernel
def raycast_kernel(origins: wp.array(dtype=wp.vec3), directions: wp.array(dtype=wp.vec3),
                   mesh_ids: wp.array(dtype=wp.uint64), n_meshes: int, floor_z: float, max_dist: float,
                   points: wp.array(dtype=wp.vec3), depth: wp.array(dtype=float),
                   valid: wp.array(dtype=int), hit_id: wp.array(dtype=int)):
    i = wp.tid()
    o = origins[i]
    d = directions[i]
    t_best = float(max_dist)
    id_best = int(NO_HIT)

    if d[2] < 0.0:                                     # floor plane z = floor_z
        t = (floor_z - o[2]) / d[2]
        if t < t_best:
            t_best = t
            id_best = FLOOR

    for k in range(n_meshes):
        q = wp.mesh_query_ray(mesh_ids[k], o, d, t_best)   # only hits closer than t_best
        if q.result:
            t_best = float(q.t)
            id_best = FLOOR + 1 + k

    hit = id_best != NO_HIT
    points[i] = o + d * t_best
    depth[i] = wp.where(hit, t_best, -1.0)
    valid[i] = wp.where(hit, 1, 0)
    hit_id[i] = id_best


class TopDownCamera:
    """Pinhole camera at `position` looking at `look_at` (default: straight down)."""

    def __init__(self, position=(3.0, 0.0, 7.0), look_at=None, width: int = 64, height: int = 64,
                 fov_deg: float = 60.0, max_distance: float = 20.0, device: str = "cuda:0"):
        self.position = np.asarray(position, np.float32)
        self.width, self.height, self.max_distance, self.device = width, height, max_distance, device
        look_at = self.position + (0.0, 0.0, -1.0) if look_at is None else np.asarray(look_at, np.float32)
        directions = self._pinhole_rays(look_at, math.radians(fov_deg))

        n = width * height
        self._origins = wp.array(np.tile(self.position, (n, 1)), dtype=wp.vec3, device=device)
        self._directions = wp.array(directions, dtype=wp.vec3, device=device)
        self._points = wp.zeros(n, dtype=wp.vec3, device=device)
        self._depth = wp.zeros(n, dtype=float, device=device)
        self._valid = wp.zeros(n, dtype=int, device=device)
        self._hit_id = wp.zeros(n, dtype=int, device=device)

    def _pinhole_rays(self, look_at, fov) -> np.ndarray:
        """Unit ray directions, row-major (H, W): forward = look direction, image up ~ world +x."""
        forward = look_at - self.position
        forward /= np.linalg.norm(forward)
        up_hint = np.array([1.0, 0.0, 0.0]) if abs(forward[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up_hint)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        half = math.tan(fov / 2.0)
        u = (2.0 * (np.arange(self.width) + 0.5) / self.width - 1.0) * half * self.width / self.height
        v = (1.0 - 2.0 * (np.arange(self.height) + 0.5) / self.height) * half
        uu, vv = np.meshgrid(u, v)                                        # (H, W)
        rays = uu[..., None] * right + vv[..., None] * up + forward
        return (rays / np.linalg.norm(rays, axis=-1, keepdims=True)).reshape(-1, 3).astype(np.float32)

    def raycast(self, rocks: RockField) -> PointCloud:
        wp.launch(raycast_kernel, dim=self.width * self.height, device=self.device,
                  inputs=[self._origins, self._directions, rocks.mesh_ids, len(rocks), rocks.floor_z,
                          self.max_distance, self._points, self._depth, self._valid, self._hit_id])
        return PointCloud(points=self._points.numpy(), valid=self._valid.numpy(),
                          depth=self._depth.numpy().reshape(self.height, self.width),
                          hit_id=self._hit_id.numpy(), points_wp=self._points, valid_wp=self._valid)
