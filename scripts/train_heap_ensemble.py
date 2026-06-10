"""Iterative online MBRL on HeapEnv: collect rollouts -> train the RobotEnsemble
a few epochs -> collect more -> train again, and plot train/val curves.

Each env step we read the joint position/velocity from HeapEnv, maintain rolling
(q, qd, act) buffers in the RobotState layout, and store ravel([q,qd,act]) as a
Dataset row. One episode == one contiguous Dataset (so rollout windows never cross
a reset). The model is trained with OnlineTrainer.train_model_bptt.
"""
import os
# Let JAX coexist with torch on the GPU instead of grabbing all the memory.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from types import SimpleNamespace
import numpy as np
import jax
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from safe_mbrl.envs.heap_env.heap_example_env import HeapEnv
from safe_mbrl.models.robot_ensemble import RobotEnsemble
from safe_mbrl.models.online_trainer import OnlineTrainer
from safe_mbrl.utils.type_aliases import Dataset


class ListLogger:
    """Minimal logger matching OnlineTrainer.log_training's scalar_summary calls."""
    def __init__(self):
        self.data = {}

    def scalar_summary(self, tag, value, step):
        self.data.setdefault(tag, []).append((int(step), float(value)))


def collect(env, n_steps, jd, bd, min_len):
    """Run episodes until `n_steps` env steps are gathered. Returns a list of
    Datasets, one per episode (rows are consecutive timesteps)."""
    datasets, collected = [], 0
    while collected < n_steps:
        env.reset()
        q0 = env.dof_pos_history.squeeze(0).cpu().numpy()[:, 0]          # current joints
        q_buf = np.tile(q0, bd).astype(np.float32)
        qd_buf = np.zeros(jd * bd, np.float32)
        act_buf = np.zeros(jd * bd, np.float32)

        rows, terminated = [], False
        while not terminated and collected < n_steps:
            a = np.clip(0.5 * np.random.randn(1, jd), -1.0, 1.0).astype(np.float32)
            _, _, terminated, _, _ = env.step(a)
            q_t = env.dof_pos_history.squeeze(0).cpu().numpy()[:, 0]
            qd_t = env.dof_vel_history.squeeze(0).cpu().numpy()[:, 0]

            q_buf = np.roll(q_buf, -jd); q_buf[-jd:] = q_t
            qd_buf = np.roll(qd_buf, -jd); qd_buf[-jd:] = qd_t
            act_buf = np.roll(act_buf, -jd); act_buf[-jd:] = a[0]
            rows.append(np.concatenate([q_buf, qd_buf, act_buf]))
            collected += 1

        rows = np.asarray(rows, np.float32)
        if len(rows) > min_len:                                          # usable for a rollout window
            target = np.zeros((len(rows), jd), np.float32)               # unused by BPTT
            datasets.append(Dataset(input=jax.numpy.asarray(rows), target=jax.numpy.asarray(target)))
    return datasets


def main():
    N_ROUNDS = 100
    STEPS_PER_ROUND = 600
    EPOCHS_PER_ROUND = 4
    HORIZON, HORIZON_VAL = 10, 15

    cfg = SimpleNamespace(step_penalty_coef=0, action_penalty_coef=1,
                          accel_penalty_coef=0, accel_sign_penalty_coef=0)
    env = HeapEnv(n_envs=1, use_act_net=True, n_history_steps=10,
                  n_ref_steps=15, t_step=0.04, t_traj=6.0, cfg=cfg)

    model = RobotEnsemble(joint_dim=4, buffer_dim=10, model_type="PE",
                          mode="v", num_ensembles=5)
    jd, bd = model.joint_dim, model.buffer_dim

    logger = ListLogger()
    train_ds, val_ds = [], []
    trainer = OnlineTrainer(model, optax.adam(1e-3), train_ds, val_ds,
                            batch_size=64, horizon=HORIZON, horizon_val=HORIZON_VAL,
                            nb_epochs=EPOCHS_PER_ROUND, early_stopping_patience=10_000,
                            logger=logger)

    round_marks = []
    for r in range(N_ROUNDS):
        print(f"\n=== Round {r}: collecting {STEPS_PER_ROUND} env steps ===")
        new = collect(env, STEPS_PER_ROUND, jd, bd, min_len=HORIZON_VAL + 2)
        val_ds.append(new[-1])              # last episode -> validation
        train_ds.extend(new[:-1])           # rest -> training (in-place; trainer shares the refs)
        print(f"  episodes: +{len(new)} | train={len(train_ds)} val={len(val_ds)} "
              f"| train steps={sum(len(d) for d in train_ds)}")

        round_marks.append(trainer.epoch)
        trainer.train_model_bptt(seed=r, verbose=True)
        trainer.epoch += 1                  # continue epoch numbering on the next round

    # ---- plot training / validation curves --------------------------------- #
    tr = logger.data["train/train_loss"]
    vl = logger.data["train/val_loss"]
    plt.figure(figsize=(9, 5))
    plt.plot(*zip(*tr), "-o", ms=3, label="train NLL")
    plt.plot(*zip(*vl), "-o", ms=3, label="val NLL")
    for m in round_marks:
        plt.axvline(m, color="gray", ls=":", alpha=0.5)
    plt.xlabel("epoch"); plt.ylabel("rollout NLL")
    plt.title("HeapEnv online MBRL: collect -> train, per round")
    plt.legend(); plt.grid(True, alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
