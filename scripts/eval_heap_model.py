"""Post-hoc model evaluation for heap benchmark runs that saved raw data + checkpoints.

Scores every saved ensemble checkpoint on an arbitrary slice of the logged training
data, so the model-error curve can be defined after the fact - e.g. excluding the
initial exploration rounds, or on a fixed late-training slice:

  python scripts/eval_heap_model.py --run logs/heap-eetracking/tuned_full/1 \
      --data-rounds 11-400          # per-checkpoint MSE on all post-exploration data
  python scripts/eval_heap_model.py --run ... --data-rounds 390-400   # fixed late slice

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

    # deterministic strided windows over every episode of the selected rounds
    wins = []
    for r in data_rounds:
        arr = np.load(os.path.join(args.run, "data", f"round_{r:03d}.npy"))  # (E, T, 3*jd*bd)
        for e in range(arr.shape[0]):
            for st in range(0, arr.shape[1] - (H + 1), args.stride):
                wins.append(arr[e, st:st + H + 1])
    wins = np.asarray(wins, np.float32)
    print(f"{len(wins)} windows from rounds {data_rounds[0]}-{data_rounds[-1]} "
          f"({len(data_rounds)} rounds, stride {args.stride})")

    B = args.batch
    pad = (-len(wins)) % B
    weights = np.concatenate([np.ones(len(wins)), np.zeros(pad)]).astype(np.float32)
    wins = np.concatenate([wins, np.repeat(wins[-1:], pad, axis=0)]) if pad else wins
    wins_d = jnp.asarray(wins).reshape(-1, B, H + 1, wins.shape[-1])
    weights = jnp.asarray(weights).reshape(-1, B)

    @nnx.jit
    def score(ens, chunk, w):
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
        n = jnp.sum(w)
        return (jnp.sum(vm(f_nll) * w) / n, jnp.sum(vm(f_mse) * w) / n, jnp.sum(vm(f_vel) * w) / n)

    out = args.out or os.path.join(args.run, "model_eval.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "nll", "mse", "mse_vel", "n_windows",
                    f"data_rounds={data_rounds[0]}-{data_rounds[-1]}"])
        for r in ckpt_rounds:
            params = pickle.load(open(os.path.join(args.run, "ckpt", f"params_{r:03d}.pkl"), "rb"))
            ens = nnx.merge(graphdef, jax.device_put(params))
            acc = np.zeros(3)
            for chunk, wt in zip(wins_d, weights):
                acc += np.asarray(jax.device_get(score(ens, chunk, wt))) * float(jnp.sum(wt))
            nll, mse, mse_vel = acc / float(weights.sum())
            w.writerow([r, f"{nll:.6e}", f"{mse:.6e}", f"{mse_vel:.6e}", len(wins) - pad, ""])
            f.flush()
            print(f"ckpt {r:3d}: nll {nll:9.3f}  mse {mse:.3e}  mse_vel {mse_vel:.3e}", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
