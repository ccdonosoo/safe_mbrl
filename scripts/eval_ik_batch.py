"""Batched strike-IK eval: targets from FK of random in-box poses, checks
success rate, residuals (via the mjx FK), and timing per batch size."""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import time

import numpy as np

from safe_mbrl.envs.m445_hammer import M445HammerFK, make_fk_batch
from safe_mbrl.ik.newton_ik import NewtonBatchIK, select_best
from safe_mbrl.m445_hammer_spec import SAFE_Q_MAX, SAFE_Q_MIN


def sample_strike_targets(fk_pos_axis, n, rng, axis_z_max=-0.05):
    """In-box configs with a downward-ish strike axis -> (q_true, p, normal)."""
    qs, ps, axes = [], [], []
    while sum(len(q) for q in qs) < n:
        q = rng.uniform(SAFE_Q_MIN, SAFE_Q_MAX, (4 * n, 5)).astype(np.float32)
        pos, axis = fk_pos_axis(q)
        keep = axis[:, 2] < axis_z_max
        qs.append(q[keep]); ps.append(pos[keep]); axes.append(axis[keep])
    q = np.concatenate(qs)[:n]
    p = np.concatenate(ps)[:n]
    a = np.concatenate(axes)[:n]
    return q, p, -a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--rng", type=int, default=0)
    args = ap.parse_args()

    fk = M445HammerFK()
    fk_pos_axis, _ = make_fk_batch(fk)
    rng = np.random.default_rng(args.rng)
    q_now = 0.5 * (SAFE_Q_MIN + SAFE_Q_MAX)

    print(f"{'B':>4} {'S':>4} | {'ok %':>6} | {'pos err mm':>18} | "
          f"{'axis err deg':>18} | {'feas/tgt':>8} | {'solve ms':>8} {'select ms':>9}")
    for B in args.batches:
        q_true, points, normals = sample_strike_targets(fk_pos_axis, B, rng)
        flip = rng.random(B) < 0.5  # canonicalization must absorb sign flips
        normals_in = normals.copy()
        normals_in[flip] *= -1.0

        ik = NewtonBatchIK(n_targets=B, n_seeds=args.seeds, iterations=args.iters)
        cand = ik.solve(points, normals_in, q_now=q_now)
        select_best(cand, points, normals_in, q_now, fk_pos_axis,
                    criterion="ee_path")
        t0 = time.perf_counter()
        cand = ik.solve(points, normals_in, q_now=q_now)
        t_solve = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        best = select_best(cand, points, normals_in, q_now, fk_pos_axis,
                           criterion="ee_path")
        t_select = (time.perf_counter() - t0) * 1e3

        ok = best["ok"]
        pe = best["pos_err"][ok] * 1e3
        ae = best["axis_err_deg"][ok]
        fmt = lambda v: (f"{v.mean():6.2f}/{np.percentile(v, 95):6.2f}/{v.max():6.2f}"
                         if len(v) else "     -")
        print(f"{B:>4} {args.seeds:>4} | {100 * ok.mean():5.1f}% | {fmt(pe):>18} | "
              f"{fmt(ae):>18} | {best['n_feasible'].mean():8.1f} | "
              f"{t_solve:8.1f} {t_select:9.1f}")

        jd = select_best(cand, points, normals_in, q_now, fk_pos_axis,
                         criterion="joint_dist")
        agree = (np.abs(jd["q"] - best["q"]).max(-1) < 1e-6)[ok].mean() if ok.any() else 0
        print(f"{'':>11}| ee_path vs joint_dist picked same solution: "
              f"{100 * agree:.0f}% of solved targets")


if __name__ == "__main__":
    main()
