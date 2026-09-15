"""Static, raycastable rocks.

A rock is a procedural convex hull (newton-ground's WarpRockGenerator) placed in
the world and kept as a Warp mesh. Rocks never move and are not simulated: the
camera raycasts them and the env measures the robot's clearance to them.
"""
from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class Rock:
    label: str
    position: np.ndarray       # (3,) world position of the generator origin
    radius: float
    vertices: np.ndarray       # (V, 3) world-frame vertices
    faces: np.ndarray          # (F, 3)
    mesh: wp.Mesh              # world-frame Warp mesh (raycast + distance queries)

    @property
    def top(self) -> float:
        return float(self.vertices[:, 2].max())


class RockField:
    """Container of rocks with the mesh-id array the Warp kernels consume."""

    def __init__(self, device: str = "cuda:0", floor_z: float = 0.0):
        self.device = device
        self.floor_z = floor_z
        self.rocks: list[Rock] = []
        self._generator = None
        self._mesh_ids = wp.zeros(1, dtype=wp.uint64, device=device)

    def __len__(self) -> int:
        return len(self.rocks)

    @property
    def mesh_ids(self) -> wp.array:
        return self._mesh_ids

    def add(self, position, radius: float = 0.8, *, seed: int | None = None, rest_on_floor: bool = True,
            sink: float = 0.2,
            subdivisions: int = 2, warp_vector=(1.0, 1.0, 0.7), noise_scale: float = 0.4,
            noise_frequency: float = 0.4, octaves: int = 5, persistence: float = 0.5,
            n_plane_clips: int = 16, normalize_size: bool = True, label: str | None = None) -> Rock:
        """Generate a rock and place it at `position` (x, y, z).

        rest_on_floor=True ignores z and buries the rock by `sink` x its height, so it
        sits in the ground like a field stone instead of balancing on the floor.
        normalize_size=True rescales the hull so its largest extent is 2 * radius
        (the generator's superellipsoid warp is randomly oriented per seed, so the
        raw size varies). The remaining keywords are WarpRockGenerator parameters:
        `warp_vector` stretches the base sphere per axis, `noise_*`/`octaves`/
        `persistence` set the Perlin surface roughness, `n_plane_clips` cuts facets.
        """
        if self._generator is None:
            from newton_ground.procedural_generation.rock_generator import WarpRockGenerator
            self._generator = WarpRockGenerator(device=self.device)

        hull = self._generator.generate_rock(
            radius=float(radius), subdivisions=subdivisions, warp_vector=tuple(warp_vector),
            noise_scale=noise_scale, noise_frequency=noise_frequency, octaves=octaves,
            persistence=persistence, n_plane_clips=n_plane_clips, seed=seed,
        ).convex_hull
        vertices = np.asarray(hull.vertices, np.float32)
        faces = np.asarray(hull.faces, np.int32)
        if normalize_size:
            vertices *= 2.0 * radius / (vertices.max(0) - vertices.min(0)).max()

        position = np.asarray(position, np.float32).copy()
        if rest_on_floor:
            height = vertices[:, 2].max() - vertices[:, 2].min()
            position[2] = self.floor_z - vertices[:, 2].min() - sink * height
        vertices = vertices + position

        mesh = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device=self.device),
                       indices=wp.array(faces.flatten(), dtype=wp.int32, device=self.device))
        rock = Rock(label or f"rock_{len(self.rocks)}", position, float(radius), vertices, faces, mesh)
        self.rocks.append(rock)
        self._refresh_ids()
        return rock

    def scatter(self, n: int, area=((2.5, 6.5), (-3.0, 3.0)), radius=(0.4, 0.9), min_spacing: float = 0.5,
                seed: int = 0, **generation) -> list[Rock]:
        """Drop n rocks at random in the (x, y) area with random sizes and shapes.

        Positions are rejection-sampled so rocks keep at least `min_spacing` of
        free ground between them; extra keywords go to add() for every rock.
        """
        rng = np.random.default_rng(seed)
        (x_lo, x_hi), (y_lo, y_hi) = area
        placed, rocks = [], []
        for i in range(n):
            r = float(rng.uniform(*radius))
            for _ in range(200):
                xy = rng.uniform((x_lo, y_lo), (x_hi, y_hi))
                if all(np.linalg.norm(xy - q) >= r + rq + min_spacing for q, rq in placed):
                    break
            placed.append((xy, r))
            rocks.append(self.add((xy[0], xy[1], 0.0), r, seed=int(rng.integers(1 << 30)),
                                  warp_vector=tuple(rng.uniform(0.6, 1.0, 3)),
                                  noise_scale=float(rng.uniform(0.3, 0.5)),
                                  n_plane_clips=int(rng.integers(6, 20)), **generation))
        return rocks

    def clear(self) -> None:
        self.rocks.clear()
        self._refresh_ids()

    def _refresh_ids(self) -> None:
        ids = [r.mesh.id for r in self.rocks] or [0]       # kernels loop over n = len(self)
        self._mesh_ids = wp.array(ids, dtype=wp.uint64, device=self.device)
