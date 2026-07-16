"""Batched strike IK on newton.ik: (point, normal) in BASE frame -> q in the
safe box, with the strike axis (EE +x) aligned to -n. Each target is expanded
into S in-box seeds solved as independent LM problems; select_best filters and
ranks with an injected FK (no jax imports here)."""
import numpy as np
import warp as wp

import newton
import newton.ik as ik

from safe_mbrl.m445_hammer_spec import (
    EE_BODY, HAMMER_URDF, JOINT_NAMES, SAFE_Q_MAX, SAFE_Q_MIN)


def canonicalize_normal(normals: np.ndarray) -> np.ndarray:
    """Unit normals pointing up out of the surface (rows with n_z < 0 flipped)."""
    n = np.array(normals, np.float32).reshape(-1, 3)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n[n[:, 2] < 0.0] *= -1.0
    return n


def _find_joint(labels, name):
    for i, lbl in enumerate(labels):
        if lbl == name or lbl.endswith("/" + name):
            return i
    raise KeyError(f"joint {name!r} not found")


@wp.kernel
def _axis_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    target_axes: wp.array(dtype=wp.vec3),
    link_index: int,
    axis_local: wp.vec3,
    start_idx: int,
    weight: float,
    problem_idx_map: wp.array(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    row = wp.tid()
    base = problem_idx_map[row]
    tf = body_q[row, link_index]
    a = wp.quat_rotate(wp.transform_get_rotation(tf), axis_local)
    err = target_axes[base] - a
    residuals[row, start_idx + 0] = weight * err[0]
    residuals[row, start_idx + 1] = weight * err[1]
    residuals[row, start_idx + 2] = weight * err[2]


@wp.kernel
def _jac_fill(
    q_grad: wp.array2d(dtype=wp.float32),
    n_dofs: int,
    start_idx: int,
    component: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    row = wp.tid()
    for j in range(n_dofs):
        jacobian[row, start_idx + component, j] = q_grad[row, j]


class AxisAlignObjective(ik.IKObjective):
    """3-row residual weight * (target_axis - R_link @ axis_local), autodiff."""

    def __init__(self, link_index: int, target_axes: wp.array,
                 axis_local=(1.0, 0.0, 0.0), weight: float = 1.0):
        super().__init__()
        self.link_index = link_index
        self.target_axes = target_axes
        self.axis_local = wp.vec3(*axis_local)
        self.weight = weight
        self.e_arrays = None

    def residual_dim(self):
        return 3

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        if jacobian_mode != ik.IKJacobianType.ANALYTIC:
            self.e_arrays = []
            for component in range(3):
                e = np.zeros((self.n_batch, self.total_residuals), np.float32)
                e[:, self.residual_offset + component] = 1.0
                self.e_arrays.append(
                    wp.array(e.flatten(), dtype=wp.float32, device=self.device))

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx,
                          problem_idx):
        wp.launch(
            _axis_residuals,
            dim=body_q.shape[0],
            inputs=[body_q, self.target_axes, self.link_index, self.axis_local,
                    start_idx, self.weight, problem_idx],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        for component in range(3):
            tape.backward(grads={tape.outputs[0]: self.e_arrays[component].flatten()})
            q_grad = tape.gradients[dq_dof]
            wp.launch(
                _jac_fill,
                dim=self.n_batch,
                inputs=[q_grad, model.joint_dof_count, start_idx, component],
                outputs=[jacobian],
                device=self.device,
            )
            tape.zero()


class NewtonBatchIK:
    """One LM solve over B targets x S seeds; returns all candidates (B, S, 5)."""

    def __init__(self, n_targets: int, n_seeds: int = 64,
                 urdf_path: str = HAMMER_URDF,
                 q_min=SAFE_Q_MIN, q_max=SAFE_Q_MAX,
                 w_pos: float = 1.0, w_axis: float = 0.7, w_limit: float = 10.0,
                 iterations: int = 32):
        self.n_targets, self.n_seeds = n_targets, n_seeds
        self.q_min = np.asarray(q_min, np.float32)
        self.q_max = np.asarray(q_max, np.float32)
        self.iterations = iterations

        # floating=False -> world == BASE; collapse=False keeps the EE body.
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_urdf(urdf_path, floating=False, collapse_fixed_joints=False)
        self.model = builder.finalize()
        nq = self.model.joint_coord_count
        if nq != len(JOINT_NAMES):
            raise ValueError(f"expected {len(JOINT_NAMES)} coords, got {nq}")
        labels = list(self.model.joint_label)
        q_start = self.model.joint_q_start.numpy()
        self.q_idx = np.array([q_start[_find_joint(labels, n)] for n in JOINT_NAMES])
        if not (self.q_idx == np.arange(nq)).all():
            raise ValueError(f"unexpected joint coord order: {self.q_idx}")
        self.ee_index = next(i for i, b in enumerate(self.model.body_label)
                             if b.endswith(EE_BODY))

        # safe box into the model BEFORE IKSolver snapshots the limits
        self.model.joint_limit_lower.assign(self.q_min)
        self.model.joint_limit_upper.assign(self.q_max)

        P = n_targets * n_seeds
        self.target_pos = wp.zeros(P, dtype=wp.vec3)
        self.target_axes = wp.zeros(P, dtype=wp.vec3)
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index, link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=self.target_pos, weight=w_pos)
        self.axis_obj = AxisAlignObjective(
            self.ee_index, self.target_axes, weight=w_axis)
        self.limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper, weight=w_limit)
        self.solver = ik.IKSolver(
            self.model, n_problems=P,
            objectives=[self.pos_obj, self.axis_obj, self.limit_obj],
            optimizer="lm", jacobian_mode="autodiff")
        self.joint_q = wp.zeros((P, nq), dtype=wp.float32)

    def solve(self, points, normals, q_now=None, rng_seed: int = 0,
              iterations: int | None = None) -> np.ndarray:
        """points/normals (B, 3) in BASE frame -> candidates (B, S, 5)."""
        B, S = self.n_targets, self.n_seeds
        points = np.asarray(points, np.float32).reshape(B, 3)
        axes = -canonicalize_normal(normals).reshape(B, 3)

        self.target_pos.assign(np.repeat(points, S, axis=0))
        self.target_axes.assign(np.repeat(axes, S, axis=0))

        rng = np.random.default_rng(rng_seed)
        seeds = rng.uniform(self.q_min, self.q_max, (B, S, len(self.q_min)))
        if q_now is not None:
            seeds[:, 0] = np.asarray(q_now, np.float32)
        self.joint_q.assign(seeds.reshape(B * S, -1).astype(np.float32))

        self.solver.reset()
        self.solver.step(self.joint_q, self.joint_q,
                         iterations=iterations or self.iterations)
        return self.joint_q.numpy().reshape(B, S, -1)


def select_best(q_cand, points, normals, q_now, fk_pos_axis,
                q_min=SAFE_Q_MIN, q_max=SAFE_Q_MAX,
                pos_tol: float = 0.02, axis_tol_deg: float = 8.0,
                criterion: str = "ee_path", n_path: int = 24,
                box_eps: float = 1.0e-4):
    """Filter (pos/axis tol + in-box) and rank; fk_pos_axis: (N,5)->(pos,axis).

    criterion "ee_path" = EE arc length of the joint path from q_now;
    "joint_dist" = box-normalized joint distance.
    """
    B, S, jd = q_cand.shape
    flat = q_cand.reshape(-1, jd)
    points = np.asarray(points, np.float32).reshape(B, 3)
    tgt_axis = -canonicalize_normal(normals).reshape(B, 3)
    q_now = np.asarray(q_now, np.float32)

    pos, axis = fk_pos_axis(flat)
    pos = np.asarray(pos).reshape(B, S, 3)
    axis = np.asarray(axis).reshape(B, S, 3)

    pos_err = np.linalg.norm(pos - points[:, None], axis=-1)
    cosang = np.clip(np.sum(axis * tgt_axis[:, None], axis=-1), -1.0, 1.0)
    axis_err = np.degrees(np.arccos(cosang))
    in_box = ((flat >= q_min - box_eps) & (flat <= q_max + box_eps)).all(-1)
    feasible = (pos_err < pos_tol) & (axis_err < axis_tol_deg) \
        & in_box.reshape(B, S)

    span = (q_max - q_min).astype(np.float32)
    if criterion == "joint_dist":
        score = np.linalg.norm((flat - q_now) / span, axis=-1).reshape(B, S)
    elif criterion == "ee_path":
        s = np.linspace(0.0, 1.0, n_path, dtype=np.float32)
        path = q_now + (flat[:, None] - q_now) * s[None, :, None]
        xyz, _ = fk_pos_axis(path.reshape(-1, jd))
        xyz = np.asarray(xyz).reshape(B * S, n_path, 3)
        score = np.linalg.norm(np.diff(xyz, axis=1), axis=-1).sum(-1).reshape(B, S)
    else:
        raise ValueError(f"unknown criterion {criterion!r}")

    score = np.where(feasible, score, np.inf)
    best = score.argmin(axis=1)
    take = (np.arange(B), best)
    return {
        "q": q_cand[take],
        "ok": feasible.any(axis=1),
        "pos_err": pos_err[take],
        "axis_err_deg": axis_err[take],
        "score": score[take],
        "n_feasible": feasible.sum(axis=1),
    }
