"""Aggregate the heap benchmark suite: mean±std across seeds per planner objective.

Reads logs/heap-eetracking/{objective}/{seed}/train.csv (+ eval.csv) and writes
comparison plots with x = episode index (1 round = 10 rollouts = 1500 transitions,
the same unit as the TD-MPC2 / DreamerV3 benchmark plots).
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = [
    ("episode_reward", "episode reward (sim, benchmark coefs)", "linear"),
    ("model_loss", "model train NLL", "linear"),
    ("err_term_err_ee_pos", "err_ee_pos [m^2]", "log"),
    ("err_term_err_ee_rot", "err_ee_rot [rad^2]", "log"),
    ("err_term_err_ee_vel", "err_ee_vel", "log"),
    ("err_term_err_j_pos", "err_j_pos [rad^2]", "log"),
    ("err_term_err_j_vel", "err_j_vel", "log"),
]


def load(base, objective, seeds, category):
    dfs = []
    expected = 400 if category == "train" else 80
    for s in seeds:
        path = os.path.join(base, objective, str(s), f"{category}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if len(df) < expected:
                # seed being re-collected for eval trajectories: fall back to the
                # archived complete first-pass run until the rerun finishes
                alt = os.path.join(base, os.pardir, f"_archive_{objective}1_firstpass",
                                   str(s), f"{category}.csv")
                if os.path.exists(alt):
                    df = pd.read_csv(alt)
            df["seed"] = s
            dfs.append(df)
    return dfs


def band(ax, dfs, key, xkey, label, color):
    if not dfs:
        return
    n = min(len(d) for d in dfs)
    x = dfs[0][xkey].values[:n]
    y = np.stack([d[key].values[:n] for d in dfs])
    mu, sd = y.mean(0), y.std(0)
    ax.plot(x, mu, label=f"{label} (n={len(dfs)})", color=color)
    ax.fill_between(x, mu - sd, mu + sd, alpha=0.25, color=color, linewidth=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "logs", "heap-eetracking"))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--objectives", nargs="+", default=["tuned", "matching"])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.join(args.base, "comparison.png")

    colors = {"tuned": "tab:blue", "matching": "tab:orange"}
    train = {o: load(args.base, o, args.seeds, "train") for o in args.objectives}
    evals = {o: load(args.base, o, args.seeds, "eval") for o in args.objectives}

    ncols = 4
    nrows = int(np.ceil((len(METRICS) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, title, scale) in zip(axes, METRICS):
        for o in args.objectives:
            band(ax, train[o], key, "episode", o, colors.get(o))
        ax.set_title(title)
        ax.set_xlabel("episode (1500 transitions)")
        ax.set_yscale(scale)
        if scale == "log":
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ax = axes[len(METRICS)]
    for o in args.objectives:
        dfs = [d.assign(episode=d["step"] / 150) for d in evals[o]]
        band(ax, dfs, "episode_reward", "episode", o, colors.get(o))
    ax.set_title("eval reward (fixed 3320-step trajectory)")
    ax.set_xlabel("episode (1500 transitions)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    for ax in axes[len(METRICS) + 1:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
