"""Post-hoc model evaluation for heap benchmark runs that saved raw data + checkpoints.

Scores every saved ensemble checkpoint on an arbitrary slice of the logged training
data, so the model-error curve can be defined after the fact - e.g. excluding the
initial exploration rounds, or on a fixed late-training slice:

  python scripts/eval_heap_model.py --run logs/heap-eetracking/tuned_full/1 \
      --data-rounds 11-400          # per-checkpoint MSE on all post-exploration data
  python scripts/eval_heap_model.py --run ... --data-rounds 390-400   # fixed late slice
  python scripts/eval_heap_model.py --run ... --data-rounds 11-400 --cumulative
      # growing eval set: checkpoint i is scored on rounds 11..i, i.e. everything it has
      # been trained on so far except the initial exploration
  python scripts/eval_heap_model.py --run ... --online
      # prequential/test error: checkpoint i is scored ONLY on round i+1 - the data that
      # was collected while planning with checkpoint i, which it has never trained on

Writes <run>/model_eval.csv with: round, nll, mse (discounted H-step joint-position
MSE), mse_vel (one-step velocity MSE, the MBPO "Model Err" definition), n_windows.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import csv
import glob
import json
import pickle
import re

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from safe_mbrl.models.robot_ensemble import RobotEnsemble
from safe_mbrl.models.online_trainer import (rollout_loss, rollout_loss_mse,
                                             _featurize, _input_to_state)


def parse_range(spec, avail):
    """'a-b' inclusive round range, filtered to what exists."""
    a, b = (int(x) for x in spec.split("-"))
    return [r for r in avail if a <= r <= b]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="run dir containing data/, ckpt/, config.json")
    p.add_argument("--data-rounds", default="11-400",
                   help="inclusive round range of data to evaluate on (default: skip rounds 1-10 exploration)")
    p.add_argument("--ckpt-rounds", default=None, help="inclusive round range of checkpoints (default: all)")
    p.add_argument("--cumulative", action="store_true",
                   help="score checkpoint i on data rounds [start..i] instead of the fixed slice")
    p.add_argument("--online", action="store_true",
                   help="score checkpoint i on round i+1 only (the data collected while planning with it)")
    p.add_argument("--stride", type=int, default=8, help="window start stride within each 150-step episode")
    p.add_argument("--batch", type=int, default=4096, help="windows per jit call")
    p.add_argument("--out", default=None, help="output csv (default <run>/model_eval.csv)")
    args = p.parse_args()

    cfg = json.load(open(os.path.join(args.run, "config.json")))
    jd, bd = 4, cfg["buffer_dim"]
    H, gamma = cfg["bptt_horizon"], 0.95
    model = RobotEnsemble(joint_dim=jd, buffer_dim=bd, model_type="PE", mode="v",
                          features=tuple(cfg["features"]), num_ensembles=cfg["num_ensembles"],
                          key=jax.random.key(0))
    graphdef, _ = nnx.split(model.model)
    dt, mode = model._dt, model.mode

    data_avail = sorted(int(re.search(r"(\d+)", os.path.basename(f)).group(1))
                        for f in glob.glob(os.path.join(args.run, "data", "round_*.npy")))
    ckpt_avail = sorted(int(re.search(r"(\d+)", os.path.basename(f)).group(1))
                        for f in glob.glob(os.path.join(args.run, "ckpt", "params_*.pkl")))
    data_rounds = parse_range(args.data_rounds, data_avail)
    ckpt_rounds = parse_range(args.ckpt_rounds, ckpt_avail) if args.ckpt_rounds else ckpt_avail
    if not data_rounds or not ckpt_rounds:
        raise SystemExit(f"nothing to do: {len(data_rounds)} data rounds, {len(ckpt_rounds)} checkpoints")

    # deterministic strided windows over every episode of the selected rounds,
    # kept in round order with cumulative bounds so --cumulative can slice prefixes
    wins, bounds, spans = [], {}, {}
    for r in data_rounds:
        arr = np.load(os.path.join(args.run, "data", f"round_{r:03d}.npy"))  # (E, T, 3*jd*bd)
        n0 = len(wins)
        for e in range(arr.shape[0]):
            for st in range(0, arr.shape[1] - (H + 1), args.stride):
                wins.append(arr[e, st:st + H + 1])
        bounds[r] = len(wins)
        spans[r] = (n0, len(wins))
    wins = np.asarray(wins, np.float32)
    n_total = len(wins)
    print(f"{n_total} windows from rounds {data_rounds[0]}-{data_rounds[-1]} "
          f"({len(data_rounds)} rounds, stride {args.stride})")

    B = args.batch
    pad = (-n_total) % B
    if pad:
        wins = np.concatenate([wins, np.zeros((pad,) + wins.shape[1:], np.float32)])
    wins_d = jnp.asarray(wins)                       # resident on device once

    @nnx.jit
    def score(ens, chunk, k):
        w = (jnp.arange(chunk.shape[0]) < k).astype(jnp.float32)
        states = jax.vmap(jax.vmap(lambda x: _input_to_state(x, jd, bd)))(chunk)
        actions = states.act_buffer[:, :, -jd:]
        f_nll = lambda e, s, a: rollout_loss(e, s, a, jd, mode, dt, gamma, H, input_idx=None)
        f_mse = lambda e, s, a: rollout_loss_mse(e, s, a, jd, mode, dt, gamma, H, input_idx=None)
        def f_vel(e, s, a):
            s0 = jax.tree_util.tree_map(lambda x: x[0], s)
            mu, _ = jnp.split(e(_featurize(s0, None)), 2, axis=-1)
            mu = jnp.mean(mu, axis=0) if mu.ndim > 1 else mu
            pred_qd = s0.get_qd() + mu if mode == "dv" else mu
            return jnp.mean((pred_qd - s.qd_buffer[1, -jd:]) ** 2)
        vm = lambda f: nnx.vmap(f, in_axes=(None, 0, 0))(ens, states, actions)
        return jnp.stack([jnp.sum(vm(f_nll) * w), jnp.sum(vm(f_mse) * w), jnp.sum(vm(f_vel) * w)])

    out = args.out or os.path.join(args.run,
                                   "model_eval_online.csv" if args.online
                                   else "model_eval_cum.csv" if args.cumulative else "model_eval.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "nll", "mse", "mse_vel", "n_windows", "data_rounds"])
        for r in ckpt_rounds:
            if args.online:
                lo, hi = spans.get(r + 1, (0, 0))
                n, tag = hi - lo, (f"{r + 1}" if hi > lo else "none")
            elif args.cumulative:
                past = [x for x in data_rounds if x <= r]
                lo, hi = 0, (bounds[past[-1]] if past else 0)
                n = hi
                tag = f"{data_rounds[0]}-{past[-1]}" if past else "none"
            else:
                lo, hi, n = 0, n_total, n_total
                tag = f"{data_rounds[0]}-{data_rounds[-1]}"
            if n == 0:
                w.writerow([r, "nan", "nan", "nan", 0, tag]); f.flush()
                continue
            params = pickle.load(open(os.path.join(args.run, "ckpt", f"params_{r:03d}.pkl"), "rb"))
            ens = nnx.merge(graphdef, jax.device_put(params))
            acc = np.zeros(3)
            if args.online:                                  # one small constant-shape set per round
                acc += np.asarray(jax.device_get(score(ens, wins_d[lo:hi], n)))
            else:
                for i in range(lo, hi, B):
                    acc += np.asarray(jax.device_get(score(ens, wins_d[i:i + B], min(B, hi - i))))
            nll, mse, mse_vel = acc / n
            w.writerow([r, f"{nll:.6e}", f"{mse:.6e}", f"{mse_vel:.6e}", n, tag])
            f.flush()
            print(f"ckpt {r:3d}: nll {nll:9.3f}  mse {mse:.3e}  mse_vel {mse_vel:.3e}  (n={n})", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
